# Motion-defined digit classification on Common-Fate Moving MNIST

Show a model `T` context frames and ask which digit is there. Nothing in any
single frame says: the figure and the background are drawn from the same noise,
and the digit exists only as a region whose **velocity** differs from its
surroundings. A per-frame model has nothing to classify.

The claim under test is that a model's **velocity structure** exposes the digit —
`felstm`'s fixed lattice of transported copies, `melstm`'s tracked slots — while
a plain ConvLSTM has nowhere to put it.

---

## Running it

```bash
# cluster (one model per job)
sbatch --job-name=cf_lstm   submit_classification.sbatch lstm
sbatch --job-name=cf_felstm submit_classification.sbatch felstm
sbatch --job-name=cf_melstm submit_classification.sbatch melstm

# locally (lstm is comfortable, melstm slow, felstm impractical)
bash run_classification.sh lstm
```

`run_classification.sh` holds the hyperparameters and is shared by both paths.
`submit_classification.sbatch` is only the Slurm wrapper — submitting
`run_classification.sh` directly fails with *"No partition specified"* because it
carries no `#SBATCH` directives.

Extra flags pass straight through:

```bash
sbatch --job-name=cf_k6 submit_classification.sbatch melstm --num_vel_modes 6
```

---

## Read `lstm` first

`lstm` has no transport structure. **It should sit at chance (10%).** If it
climbs meaningfully above chance, the dataset is leaking a per-frame cue and
neither of the other two numbers means anything.

This is not hypothetical — it already happened once, and the cause is worth
knowing (see [The seam leak](#the-seam-leak-why-corr_len-must-be-0)).

---

## The data

`CommonFateMovingMNISTDataset`, which subclasses `TDMovingMNISTDataset` and so
inherits its whole motion vocabulary (`constant` / `piecewise` / `stochastic` /
`accelerate`, `transition_mode`, `motion_difficulty`, `freeze_after`).

| setting | default | why |
|---|---|---|
| `--image_size` | 36 | a 28px digit still has room to travel on the torus; felstm carries 25 copies of the state at every timestep, so this is where its cost is decided |
| `--seq_len` | 15 | context `T` |
| `--corr_len` | **0.0** | **leave it there** — see below |
| `--data_v_range` | 2 | figure max speed; felstm needs `--v_range >= this` |
| `--bg_mode` | `opposite` | background separation, see below |
| `--motion_mode` | `piecewise` | figure velocity held 3–6 frames, then changes |
| `--num_figures` | 1 | one digit and one background = two motions |

Every layer is band-limited noise on a torus with integer velocities, so `roll()`
is the exact group action and a warp-based model is exactly equivariant rather
than approximately.

### Two variants

| variant | construction | in the frame co-moving at `v_fg` |
|---|---|---|
| `moving_mask` (default) | mask **and** its texture travel at `v_fg` | figure is **static** → transport recovers the **shape** |
| `static_mask` | mask fixed, textures scroll underneath | nothing is static → transport recovers **texture**, not shape |

### How the background is kept distinguishable

`--bg_mode opposite` (default) keeps the background **on the shared velocity
grid** and requires it to travel in an opposing direction to the digit at `t=0`
(strictly negative dot product), plus a max-norm gap of `--min_dv` at every step.

`--bg_mode disjoint` instead gives the background a *faster* grid of its own. It
guarantees they never coincide, but it puts the background beyond every lattice
copy felstm has — a motion felstm structurally **cannot represent** while
melstm's tracked slots can. That is a difference in expressive power confounded
with the effect being measured, so it warns when paired with felstm.

### The seam leak: why `corr_len` must be 0

All layers share the same texture statistics, so no *intensity* cue can leak the
figure. But with `corr_len > 0` the **seam** can: pixels *within* a region are
correlated while pixels *across* the boundary are independent, so the digit's
outline is a local-statistics discontinuity present in **every frame**.

A single-frame CNN, given no temporal information at all:

| `corr_len` | 0.0 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|
| accuracy (chance 10%) | **11%** | 13% | 24% | 36% |

At the old `corr_len=1.0` default a per-frame model could already do much of the
job, which is why `lstm` reached high accuracy — it was reading the seam, not the
motion. `corr_len=0` is also simply better for the experiment: transport IoU
0.661 vs 0.522, `slot_hit_fig` 100% vs 96%.

An earlier version of this repo's docstring said "keep `corr_len <= 1.0`", based
on a hand-crafted gradient detector that scored only AUC 0.53 there. That
underestimated the leak: a trained CNN extracts far more from the same cue.

---

## The model

`MotionDigitClassifier` = backbone **encoder** → velocity-axis pool → conv+MLP head.

All three backbones end the encoder at `(B, V, C, H, W)` — `V` is 1 for `lstm`,
`(2R+1)²` for `felstm`, `K` slots for `melstm` — so one head serves all three and
**trained parameter counts are identical across them**. The models differ in
transport structure, not capacity. The run header prints the table so an
accidental capacity difference is visible immediately:

```
architecture : melstm  V=2 velocity slots  velocity_source=bootstrap

  submodule            params   note
  backbone.cell        38,144   recurrent, run on every velocity copy
  pool                  2,113   velocity reduce (attention)
  head.conv            92,544   3 blocks, circular, stride 2
  head.mlp             17,802   avg+max pool -> logits
  backbone.decoder      9,537   UNUSED in this task, excluded from the optimizer
  TRAINED             150,603   what the optimizer updates
```

The backbone's decoder is built (so the recurrent cell matches the prediction
experiments exactly) but never run, and is excluded from the optimizer.

### Velocity pooling — why not `max`

`--velocity_pool attention` (default). Max-pooling over the velocity axis is the
right tool when the digit is bright on black: the correctly transported copy
simply has the largest activations.

Here every copy carries **equal-amplitude noise**. What distinguishes the right
velocity is not magnitude but **coherence** — the copy transported at the
figure's velocity adds the figure in register frame after frame and develops
spatial structure, while every other copy averages a drifting texture into mush.
Their maxima stay comparable, so `max` reduces to picking the luckiest spike.

The attention score therefore reads each copy's per-channel spatial **mean and
standard deviation**, the std being the coherence signal, and softmaxes over the
velocity axis. `max` / `mean` / `concat` remain available so that claim stays
testable.

(Spatial max *inside* the head is a different thing and is used: after learned
convolutions, over space, it is right for an object covering a few percent of the
frame, where a pure average would drown it.)

### `--velocity_source` — melstm only, and it matters

Where slot velocities come from. Measured `slot_hit_fig` at `K=2` on this data:

| source | `slot_hit_fig` | what it does |
|---|---|---|
| `bootstrap` **(default)** | **97.9%** | explains the dominant motion away, re-correlates the residual |
| `frame_pair` | 68.8% | top-K peaks of the raw frame pair |
| `tracked` | **0.0%** | MEConvLSTM's own protocol |

`tracked` correlates each slot's hidden state against the next frame. That needs
a clean template and has none here — `h` is a smoothed function of mostly
background noise. Measured: after 15 frames **no slot held either velocity**, and
the slots collapsed from 4 distinct velocities onto 2.0. Collapse is
self-sustaining: two slots at the same velocity get the same warp, the same input
and the same weights, so they never separate again.

`bootstrap` is a batched torch port of `residual_bootstrap` in
`common_fate_diagnostics.py`. `Seq2SeqMEConvLSTM` itself is deliberately
unchanged — this is a property of *this data*, not a correction to the model.

### `--num_vel_modes` — why 2

The scene contains exactly two motions: the digit and the background. With
`bootstrap`, `slot_hit_fig` is 97.9% at `K=2` and **identically 97.9% at K=3 and
K=4** — extra slots buy nothing and cost compute linearly.

(With the weaker `frame_pair` source, `K=2` reaches only 69% and extra slots *do*
help — a defect of the peak selection, not evidence the scene has more motions.)

---

## What to read in wandb

### Scalars, per epoch

| key | meaning |
|---|---|
| `train_loss` / `train_acc` | cross-entropy and accuracy on the training split |
| `val_loss` / `val_acc` | same on validation; `val_acc` drives selection and the LR scheduler |
| `lr`, `time` | learning rate, seconds per epoch |

**Chance is 10%.**

### Velocity diagnostics — melstm only

These matter *more than the loss curve*, because an accuracy number from slots
that never transported at the figure's velocity is measuring nothing.

| key | meaning | healthy |
|---|---|---|
| `val_slot_hit_fig` | fraction of samples where some slot ends the encoder holding the **ground-truth figure** velocity | ~0.95+ |
| `val_slot_hit_bg` | same for the **background** velocity | ~1.0 |
| `val_attn_on_fig` | share of the attention pool's mass sitting on a copy matching the figure | should **rise** during training |

If `slot_hit_fig` is near zero, the model never represented the figure — fix that
before interpreting anything else. If it is high but `attn_on_fig` stays flat at
`1/K`, the slots found the figure but the head never learned to attend to it.

### The state images

`val_states_sample0`, `val_states_sample1` — logged every `--log_states_every`
epochs (default 1) on a **fixed** set of sequences, so the wandb step slider
shows the *same* sample developing as training proceeds.

Each figure is, top to bottom:

- **one row per velocity copy**, labelled with its velocity and marked `<- FIGURE`
  or `<- bg` when it matches ground truth. For `felstm` the informative copies are
  selected (the one matching the figure, the one matching the background, plus
  controls) because its whole lattice is unreadable; for `melstm` all `K` slots;
  for `lstm` the single untransported state.
- **`frame (input)`** — what the model saw. This is noise. The digit is genuinely
  not in it.
- **`figure (GT mask)`** — the answer key: where the figure actually was.

**What you are looking for:** the copy marked `<- FIGURE` should develop the
digit's shape over time while the others stay textureless. That is the entire
experiment in one picture.

If the title says *"not on this model's velocity lattice"*, that copy does not
exist for this model — expected for `felstm` under `--bg_mode disjoint`, and the
reason `opposite` is the default.

### Summary (final values)

`test_acc`, `test_loss`, `best_val_acc`, `best_epoch` are written to
`wandb.summary` rather than the history, so they do not disturb the step counter.

> **A wandb gotcha worth knowing.** `wandb.log()` without an explicit `step`
> commits and advances wandb's internal counter. This script logs at
> `step=epoch`, so anything logged without a step would push the counter past the
> epoch and the *next* epoch's metrics would be silently dropped with
> *"Tried to log to step N that is less than the current step"*. Every
> `wandb.log` here passes an explicit step, and a test enforces it.

---

## Files

| file | role |
|---|---|
| `moving_mnist/common_fate_moving_mnist_dataset.py` | the dataset and the numpy generator |
| `moving_mnist/common_fate_diagnostics.py` | phase correlation, co-moving accumulation, residual bootstrap |
| `moving_mnist/motion_classification_model.py` | classifier: pool + head + velocity sources |
| `moving_mnist/train_classification.py` | training loop, diagnostics, checkpointing |
| `moving_mnist/mps_integer_warp.py` | makes melstm trainable on Apple GPU |
| `moving_mnist/visualization.py` | `log_motion_classification_states` |
| `run_classification.sh` | hyperparameters, shared by local and cluster |
| `submit_classification.sbatch` | Slurm wrapper with auto-resume and self-chaining |
| `moving_mnist/tests/test_motion_classification.py` | 35 tests |
| `test_common_fate_moving_mnist.ipynb` | visual tour of the dataset (GIFs) |

### Cost

Measured at batch 16, 64px, hidden 32, `T=15` on an M-series GPU:

| model | V | s/batch |
|---|---|---|
| `lstm` | 1 | 0.30 |
| `melstm` | 4 | 14.4 |
| `felstm` | 25 | OOM at 20 GB |

The cost is the recurrent cell's **convolution** (164 ms/step), not the warp
(5 ms) or the velocity estimation (<1 ms) — so the only levers are
`--hidden_size`, `V`, and batch size. `--image_size 36` rather than 64 is the
single biggest saving for felstm. Batch size is set per model in
`run_classification.sh`; felstm gets the smaller one.

On Apple GPU, MPS implements no `aten::grid_sampler_2d_backward`, so melstm's
warp cannot backpropagate. `mps_integer_warp.py` swaps it for a gather-based
integer shift that is bit-exact against `torch.roll` for every integer velocity
(`grid_sample` is the *less* accurate of the two, ~1.5e-6 at unit scale), and
falls back to `grid_sample` for fractional velocities.
