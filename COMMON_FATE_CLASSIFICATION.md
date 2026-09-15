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

## What the controls actually mean

**`lstm` above chance is expected, and is not by itself evidence of a leak.**
An earlier version of this document said it "should sit at chance"; that was
wrong, and believing it will send you hunting for bugs that are not there.

A plain ConvLSTM is not motion-blind. Its cell convolves the input together with
the previous hidden state, which is enough to build local spatiotemporal
correlations — Reichardt-style motion detectors. It can notice that a region
moves differently from its surroundings, segment it roughly, and classify the
shape. What it **cannot** do is transport its hidden state, so it cannot
accumulate the figure coherently in a co-moving frame over many steps.

So the claim under test is **melstm and felstm beat lstm**, not that lstm fails.

### The real leak test

A model with no access to motion at all: one frame, no recurrence.

```bash
python moving_mnist/leak_probe.py                      # current defaults
python moving_mnist/leak_probe.py --corr_len 0 0.5 1 2
```

If that beats chance, a per-frame cue exists and every recurrent number is
contaminated. This is what caught the texture-seam leak (see
[The seam leak](#the-seam-leak-why-corr_len-must-be-0)); at the current
`corr_len=0` default it reports 11% against 10% chance.

### And `val_state_shape_iou`

The second control, and the one that separates "learned the motion" from "found
a shortcut": a model solving the task the intended way has the digit's shape in
the transported copy of its hidden state. `lstm` scores **below** chance there
(0.019 vs 0.062) because it has no transport. High accuracy together with a flat,
near-chance shape IoU is the signature of a shortcut.

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
| `--epochs` | 50 | early stopping is **off** (`--early_stop_patience 0`), so all three run the full 50 and the curves are comparable end to end |
| `--max_train_samples` | 50000 | **not a dataset size** — see below |
| `--val_curve_interval` | 25 | fixed-set val loss every N optimizer steps |
| `--val_curve_size` | 256 | how many fixed sequences behind that curve |

### The splits are disjoint at the GLYPH level

MNIST's 60k training images are split **54000 train / 6000 val**, with no shared
glyphs, and test draws from MNIST's own 10k test split — three disjoint sets.
Velocities and textures are re-drawn on every access, so an epoch revisits the
same digits under fresh motion; only the glyph identity is held fixed by the
index.

This matters for classification and is easy to get wrong. Without
`digit_indices` the dataset **ignores its index** and samples a random glyph on
every access, so a `random_split` hands both halves the same 60k pool and val is
measured on digits the model already trained on. That is harmless for next-frame
prediction — what the parent class was built for — and wrong here.

`--max_train_samples` therefore caps **distinct digits per epoch**, not just
epoch length: lowering it genuinely reduces data diversity. Leave it unset to use
all 54000.

Measured cost at 36px, batch 32, `T=15`, hidden 32 on an M-series GPU (the A100 is
much faster; the ratios are what transfer):

| model | V | s/batch | 50 epochs @ 20k | @ 50k |
|---|---|---|---|---|
| `lstm` | 1 | 0.19 | 1.7 h | 4.1 h |
| `melstm` | 2 | 0.44 | 3.8 h | 9.5 h |
| `felstm` | 25 | — | OOM at 20 GB (cluster-only) |

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

### Normalisation — BatchNorm, with its statistics recomputed

`--head_norm batch` (default) and `--precise_bn_batches 50`. Both are measured
choices; getting either wrong produces a failure that looks like a data problem.

**BatchNorm is load-bearing here, not a convenience.** Train accuracy over 14
epochs — the one number every variant reports identically:

| norm | lr | ep0 → ep13 |
|---|---|---|
| **batch** | 1e-3 | 0.109 → **0.230** |
| group | 1e-3 | 0.104 → 0.109 (flat) |
| group | 3e-3 | 0.103 → 0.096 (flat) |
| none | 1e-3 | 0.110 → 0.106 (flat) |

Only BatchNorm learns at all. The likely reason is *what* it normalises over: per
channel **across the batch**, which removes the component common to every sample
— here the background noise field, which dominates the small, low-contrast
structure the figure contributes. GroupNorm divides each sample by its own
standard deviation, dominated by that same background, leaving the ratio
untouched.

**But its running statistics lag badly.** They are an EMA collected while the
weights were still moving, and this head sits on a *recurrent* state whose
distribution shifts as both the cell and the attention weights train, so the EMA
never catches up. Left alone, val accuracy oscillates between chance and its true
value **between epochs** while training accuracy rises smoothly — which is what
it looked like in practice:

| ep | train_acc | val (stale EMA) | val (**recomputed**) | val (batch stats) |
|---|---|---|---|---|
| 3 | 0.196 | 0.087 | **0.225** | 0.209 |
| 5 | 0.203 | 0.103 | **0.235** | 0.240 |
| 7 | 0.216 | 0.127 | **0.230** | 0.231 |
| 9 | 0.211 | 0.122 | **0.246** | 0.229 |

`recompute_bn_stats` re-estimates every BatchNorm from the training distribution
immediately before each evaluation, using `momentum=None` so PyTorch accumulates
a cumulative average — the exact mean and variance for the *current* weights. It
is gradient-free and changes no parameters. Cost is `--precise_bn_batches`
forward passes per epoch (~20s for melstm at 50).

**If you see val bouncing between chance and a high value, check this first.** The
diagnostic is to evaluate the same model twice, once with `model.eval()` and once
with `model.train()` (batch statistics): if the batch-statistics number tracks
training accuracy and the eval one does not, it is BatchNorm — not the data.

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

### `val_state_shape_iou` — transport coherence, NOT a predictor of accuracy

Takes the velocity copy transported at the figure's velocity, takes the local
variance of its channel mean, and scores it against the true mask (area-matched
IoU). `val_state_shape_iou_best` is the same over the best copy;
`val_state_shape_iou_chance` is the mask's area fraction.

Read it as a measure of **transport coherence**. Three limits, all measured, and
worth knowing before you draw conclusions from it:

**It does not predict accuracy.** felstm reached 0.976 val accuracy with a
matched IoU sitting at chance — its information is spread across copies and over
time rather than concentrated in one copy at the end.

**It penalises fixed velocity lattices under time-varying motion.** Under
`--motion_mode constant` melstm and felstm score *identically*; only under
`piecewise` does felstm fall behind:

| model | constant | piecewise | chance |
|---|---|---|---|
| melstm | 0.521 | 0.494 | ~0.10 |
| felstm | **0.521** | **0.317** | ~0.10 |

felstm's copies sit at fixed lattice velocities, so a copy only accumulates
coherently while the figure's velocity matches it; melstm re-estimates its slots
every step and stays locked on through a change. That is a real architectural
difference, not a defect in either — and it is arguably the most interesting
thing this metric shows.

**It reads the channel MEAN**, so it measures raw accumulated texture. A trained
cell may encode the figure in particular channels that cancel in the mean, which
is why the number tends to fall rather than rise during training.

`best` is the fairer cross-architecture number — "is the digit anywhere in the
state" — but it favours models with more copies, since taking the best of V gets
more chances as V grows.

### Velocity diagnostics — melstm only

These matter *more than the loss curve*, because an accuracy number from slots
that never transported at the figure's velocity is measuring nothing.

| key | meaning | healthy |
|---|---|---|
| `val_slot_hit_fig` | fraction of samples where some slot ends the encoder holding the **ground-truth figure** velocity | ~0.95+ |
| `val_slot_hit_bg` | same for the **background** velocity | ~1.0 |
| `val_attn_on_fig` | share of the attention pool's mass on a copy matching the figure | should **rise** |
| `val_attn_on_bg` | same for the background copy | should fall as the head commits |
| `val_attn_entropy` | entropy of the attention distribution, in nats | should **fall** from `log(K)` |
| `val_attn_entropy_max` | `log(K)`, the hedging ceiling, for reference | constant |

If `slot_hit_fig` is near zero, the model never represented the figure — fix that
before interpreting anything else. If it is high but `attn_on_fig` stays flat at
`1/K` and `attn_entropy` stays at `log(K)`, the slots found the figure but the
head never learned to select it, and the pool is doing nothing.

### The state images

`val_states_sample0 … sample29` — the paper figure. Logged every
`--log_states_every` epochs (default 1) on a **random but fixed** set of
sequences, so the panels span different digits and speeds while the wandb slider
still shows the *same* sequences developing across epochs. Every timestep
`0 … T-1` is a column.

Rows, top to bottom:

- **one row per velocity copy**, the FIGURE one in bold. For `felstm` the
  informative copies are selected (matching the figure's velocity, matching the
  background's, plus controls) because its whole lattice is unreadable; for
  `melstm` all `K` slots; for `lstm` the single untransported state, labelled as
  such — it has no lattice, so no lattice vocabulary appears on it.
- **`input frame (ground truth)`** — what the model saw. Noise. The digit is
  genuinely not in it.
- **`figure mask (ground truth)`** — where the figure actually was.

**What to look for:** the copy marked FIGURE should develop the digit's shape
over time while the others stay textureless.

Ground-truth velocities are deliberately **not** printed. The motion is
piecewise-constant, so a single `v_fig` for the whole sequence would be wrong.
Velocities appear on a row only where they are genuinely constant — felstm's
fixed lattice copies. A melstm slot re-estimates every step, so its row is
labelled by which motion it followed, not by a number.

`--states_fig_dir DIR` also writes each panel as PNG and PDF for the paper.

### `val_readout_states_*` — diagnostic, not for the paper

A few extra panels (`--log_states_readout_samples`, default 3) that add a
**local variance (figure copy)** row: the picture form of
`val_state_shape_iou`, showing where that copy accumulated coherent structure.
Kept under a separate key, and never written to `--states_fig_dir`, because it
is a diagnostic about the metric rather than about the model — and the metric
measures transport coherence rather than anything predictive of accuracy.

### `test_confusion`### `test_confusion`

A 10x10 confusion matrix on the test set, logged at the end, plus
`test_acc_digit0 … digit9` in the summary. The *structure* of the errors says
more about mechanism than the scalar does: a model reading a per-frame cue tends
to confuse digits by stroke statistics, one reading motion by shape.

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
| `moving_mnist/leak_probe.py` | single-frame CNN — the dataset's leak test |
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
