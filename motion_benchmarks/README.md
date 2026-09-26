# motion_benchmarks — experiments beyond Moving MNIST

Everything the "third experiment" discussion proposed, implemented on one branch
(`real_motion_experiments`) and **additive**: the models and trainers in `moving_mnist/` are
reused, and the only edits to existing files are opt-in flags whose defaults reproduce the old
behaviour bit-for-bit (checked by tests).

| # | experiment | kind | what it shows | data |
|---|---|---|---|---|
| E1 | STEPS-style synthetic radar | prediction | transport pays in proportion to the rain's Lagrangian lifetime; ground-truth wind | generated |
| E2 | real radar (SEVIR VIL, pySTEPS archives) | prediction | the application ConvLSTM was built for, vs. the operational baseline | download |
| E3 | rendered real trajectories (rounD / inD / exiD / highD) | prediction | real, continuous, correlated velocities (braking, turning) | levelX application; synthetic stand-in included |
| E4 | fluids in a moving frame (Rayleigh–Bénard 3D, Swift–Hohenberg, The Well) | prediction | Galilean = flow, non-inertial = motion; exact mean-flow connection; head-to-head with Fromme et al. | simulate (Dedalus / Oceananigans) or The Well |
| E5 | GOES satellite clouds | prediction | long-lived clouds, several wind layers (K slots) | AWS open data |
| E6 | common-fate real silhouettes (Weizmann) + articulation sweep | classification | every frame is noise; transport is the enabling mechanism; how far outside G it holds | Weizmann masks; synthetic stand-in included |
| E7 | two-photon calcium imaging with injected brain motion | prediction | real rigid motion + intrinsic change, the Figure 1 setup | Suite2p / Neurofinder / Allen; synthetic stand-in included |

Every dataset without public data ships a synthetic stand-in, so every pipeline runs end to end
today; the real-data readers are tested on small fakes in the exact file formats.

---

## Quick start

```bash
# from the repo root, in the same environment as moving_mnist/
pip install h5py scipy pandas                # pysteps, netCDF4, tifffile, dedalus: optional
python -m pytest motion_benchmarks/tests -q   # ~1-2 min on CPU, 80+ tests

# 2-minute end-to-end check of any dataset / model
python -m motion_benchmarks.train_motion --dataset radar_synthetic --model melstm --smoke_test
```

`python -m motion_benchmarks.train_motion --help` lists every option; dataset options carry the
dataset's prefix (`--radar_*`, `--fluid_*`, `--sh_*`, `--rbc_*`, `--well_*`, `--traj_*`, `--sat_*`,
`--ca_*`).

---

## Three things found while building this — read these first

**1. The rollout is not exactly equivariant with the original decoder input.** In
`Seq2SeqMEConvLSTM.forward` (and the FE decoder) each rollout step warps the state to the *next*
frame's position and then feeds the cell the *previous* frame, unwarped. Input and state are
misaligned by the step velocity `v`, and after a frame motion `u` by `v + u`, so the decoder is
not exactly equivariant — the encoder is (to 1e-6). `tests/test_models.py` demonstrates both.
`--decoder_input warped` feeds every slot the previous prediction warped with *its own* velocity
(aligned, exactly equivariant; the input is then the Lagrangian-persistence guess of the next
frame). `slurm/run_suite.sh` uses `warped` for the new experiments; the default stays `previous`
so the existing Moving MNIST runs reproduce. Worth a sentence in the paper, and possibly a re-run
of Moving MNIST with `warped` to see whether it matters in practice.

**2. Fixed benchmark sets are duplicated across DataLoader workers.** `TDMovingMNISTDataset` with
`random=False` (test, len-gen) keeps a private RNG and ignores the index, so each of the
`num_workers` workers regenerates the same stream: with 4 workers the 10,000-sequence test set
has ~2,500 distinct sequences. Means are unbiased, error bars too small by ~2×. Training
(`random=True`) is not affected. `train.py` / `train_classification.py --distinct_worker_streams`
fixes it (opt-in, because it changes which sequences the benchmark sets contain). The datasets
here seed item *i* from `(seed, i)`, so they are immune (tested).

**3. Two corrections to earlier notes.**
- Fromme et al.'s `Ra = 2500` is in units where the layer height is 2; the Dedalus runs here give
  `Nu ≈ 3` and a well-mixed interior, i.e. a height-based `Ra ≈ 2×10⁴` — well supercritical, not
  "just above onset". Plumes do evolve; they still mostly overturn in place (no frame motion).
- Swift–Hohenberg *rolls* have an aperture problem: motion along a straight roll is unobservable
  (and harmless for prediction). Hexagons (`--sh_g2 1`, the default) are fully observable
  (0.02 px phase-correlation error). Earlier SH numbers used a grid-scaled `k_c` that selects the
  domain mode for few rolls; the solver here fixes `k_c = 1` and sizes the box.

---

## The shared protocol (`train_motion.py`)

Models (`--model`):

| name | what |
|---|---|
| `lstm` | ConvLSTM (`FEConvLSTMPlus`, `v_range 0`) |
| `felstm` | FEConvLSTM, integer lattice `[-v_range, v_range]²` |
| `melstm` | MEConvLSTM (`MEConvLSTMPlus`), `--velocity_source` below |
| `melstm_oracle` | MEConvLSTM fed the **true** velocity — separates "transport helps" from "estimation is hard" |
| `lstm_stabilized` | ConvLSTM run in the frame co-moving with the **true** motion (undo → predict → redo): the best a static-symmetry model (e.g. Fromme's) can do when *told* the frame. MEConvLSTM should match it without being told |
| `persistence_eulerian` / `persistence_lagrangian` / `persistence_oracle` | copy the last frame / move it with the estimated / true velocity |

The three persistence baselines are also evaluated inside **every** run on the same test sets,
so each `results.json` carries its own reference and `headroom_eulerian_over_lagrangian`.

MEConvLSTM velocity sources (`--velocity_source`):

| source | velocity from | use for |
|---|---|---|
| `track` | the original protocol: bootstrap K peaks from (X0, X1), each slot correlates its hidden state with the next frame; optional gate `--track_gate_radius/--track_gate_min_conf/--track_gate_erode` | objects on a plain background |
| `frame_pair` | K peaks of phase correlation on consecutive raw frames (optionally `--frame_pair_mode residual`), slot identity by nearest-previous matching | textures, fluids, radar |
| `external` | the dataset's ground truth (`melstm_oracle`) | upper bound |
| `mean_flow` | the fluid's own spatial-mean horizontal velocity — the exact Galilean connection `⟨u_h⟩[ψ▷X] = ⟨u_h⟩[X] + v_t`; the network sees `u − ⟨u⟩` (zero-momentum gauge) | E4 with `--fluid_action galilean` |

Other model options: `--pc_alpha` (whitening exponent; datasets recommend 1 for textures,
0.5 for physics/rain, 0.25 with a window for calcium), `--pc_subpixel`, `--pc_search_radius`
(default: the dataset's speed bound), `--pc_window` (Hann taper for non-periodic crops),
`--pc_channels`, `--residual {none,eulerian,lagrangian}` (Lagrangian = "Lagrangian persistence +
learned evolution"; on `lstm` it is the key ablation: same transported skip, untransported
memory), `--decoder_input {previous,warped,zeros}`, `--forget_bias`, `--forget_bias_long`,
`--long_fraction` (two-timescale memory, `models/gates.py`).

Protocol details kept from `moving_mnist/train.py`: fresh training sequences every epoch; fixed
val / test / gen_test; MEConvLSTM's decoder velocity tracked against the target in training and
**frozen** at evaluation (`--eval_velocity_mode both` also logs the tracked oracle); the future
velocity freezes at the last context transition (`freeze_after = input_frames`) unless
`--no_freeze_future`; length generalisation via `--gen_pred_frames`.

Outputs in `--save_dir/<run_name>/` (default `./experiments/motion_benchmarks/`, git-ignored):
`checkpoint_best.pth`, `checkpoint_last.pth`, `results.json` (per-lead MSE / MAE / RMSE,
CSI / POD / FAR / bias and FSS at the dataset's thresholds, encoder velocity EPE, Nusselt and
KE-spectrum errors for fluids, extra test sets, baselines), `examples.png`.

```bash
# one comparison = identical data and budget for every model (Slurm)
bash motion_benchmarks/slurm/run_suite.sh radar_synthetic --radar_lifetime 6,48
python -m motion_benchmarks.scripts.plot_results "experiments/motion_benchmarks/radar_synthetic_*/results.json" --out radar.png
```

---

## E1 — STEPS-style synthetic radar  (`--dataset radar_synthetic`)

Power-law field (`--radar_beta 2.6`), intermittent (`--radar_wet_fraction 0.35`), every Fourier
mode AR(1) with a scale-dependent Lagrangian lifetime `τ(k) = lifetime·(k_ref/k)^0.8` (small
scales die faster), advected by a time-varying wind (`--radar_schedule piecewise|ou|rotating`,
`--radar_max_speed`). `--radar_layers 2` superposes independent layers with independent winds
(wind shear, a K-slot case). Model input `log1p(R)/4`; CSI/FSS thresholds R = 0.5, 2, 8.

Headroom before training anything (`scripts/headroom.py`, 96×96, 12-frame context):

| lifetime | lead 1 | lead 3 | lead 6 | lead 12 |
|---|---|---|---|---|
| 3 | 1.13 | 1.07 | 1.03 | 1.05 |
| 12 | 1.97 | 1.69 | 1.49 | 1.40 |
| 48 | 5.17 | 4.10 | 3.20 | 2.60 |

```bash
python -m motion_benchmarks.scripts.headroom --dataset radar_synthetic \
    --sweep radar_lifetime=3,6,12,24,48 --n 96 --image_size 96 --out headroom_radar.json
bash motion_benchmarks/slurm/run_suite.sh radar_synthetic --image_size 96 --radar_lifetime 6,48 \
    --radar_test_lifetimes 3,6,12,24,48 --radar_test_fast_speed 4 --residual lagrangian
```
Report: MSE/CSI/FSS vs lead for each model against Lagrangian persistence; the trained-model
gain per lifetime (the `extra_lifetime_*` test sets) next to the headroom curve;
`fast_wind` for velocity generalisation. FE is affordable here (`FE_V=3`, 49 copies); at real
radar speeds it is not — that is the feasibility argument.

## E2 — real radar  (`--dataset radar_real`)

```bash
# SEVIR VIL (1 km, 5 min, 384 px, 49 frames per event) -> 96 px at 4 km
aws s3 cp --no-sign-request s3://sevir/data/vil/2019/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5 .
python -m motion_benchmarks.scripts.prepare_radar --source sevir \
    --inputs "SEVIR_VIL_*.h5" --downsample 4 --out data/radar/sevir_vil96.h5
# or pySTEPS example cases (<= 24 frames each: a qualitative figure only)
python -m motion_benchmarks.scripts.prepare_radar --source pysteps --pysteps_case fmi \
    --pysteps_data_dir data/pysteps --tile 96 --out data/radar/fmi.h5

# no ground truth -> no oracle models
MODELS="lstm felstm melstm_fp" bash motion_benchmarks/slurm/run_suite.sh radar_real \
    --radar_file data/radar/sevir_vil96.h5
python -m motion_benchmarks.scripts.pysteps_baseline --dataset radar_real \
    --radar_file data/radar/sevir_vil96.h5 --n 1000 --out pysteps_lk.json   # operational baseline
```
Events are split train/val/test by event (`--radar_split 0.8,0.1,0.1`), windows with less than
`--radar_min_wet` rain are resampled, the crop is not periodic (phase correlation uses a Hann
window automatically). SEVIR CSI thresholds are the standard VIL levels 16, 74, 133, 160, 181,
219 (/255).

## E3 — rendered real trajectories  (`--dataset trajectories`)

```bash
# after the levelX academic application (rounD / inD / exiD / uniD; highD for motorways):
python -m motion_benchmarks.scripts.prepare_trajectories --dataset round \
    --raw_dir ~/data/rounD-dataset-v1.0/data --out_dir data/trajectories/round
bash motion_benchmarks/slurm/run_suite.sh trajectories --traj_source round \
    --traj_dir data/trajectories/round --traj_split "00-17;18-20;21-23" --traj_fps 5 \
    --traj_meters_per_px 0.5 --traj_n_motions 2 --num_vel_modes 2
# without the data: synthetic roundabout / intersection / highway kinematics
python -m motion_benchmarks.train_motion --dataset trajectories --traj_synthetic_kind roundabout ...
```
Sprites: `box` (oriented, SE(2)) or `gaussian` (isotropic, exactly ℝ²). `--traj_camera piecewise`
adds a drifting drone (a global time-dependent motion on top of the objects);
`--traj_background photo` samples the site's orthophoto. rounD/inD at 0.5 m/px and 5 fps give
~2 px per step with the heading rotating on the ring; highD needs ~1 m/px and 10 fps. The motion
is real, so it is **not** frozen in the future — the frozen-velocity rollout meets real braking
and turning; report `--eval_velocity_mode both`.

## E4 — fluids in a moving frame  (`--dataset rbc3d | swift_hohenberg | the_well`)

`--fluid_action regular` translates the fields (exactly the paper's action; velocity by phase
correlation, α = 0.5); `galilean` also adds the observer velocity to the horizontal velocity
channels (a non-inertial observer; `--velocity_source mean_flow` is then exact and
parameter-free); `none` is the lab frame. Frame motion: `--fluid_schedule`, `--fluid_max_speed`
(px per snapshot); `--fluid_test_fast_speed` for velocity generalisation;
`--fluid_extra_actions none` adds a lab-frame test set.

**Rayleigh–Bénard 3D (Fromme et al., the head-to-head).**
```bash
# data: 100 runs, 48x48x32, Ra 2500, Pr 0.7, t in [100, 300] every 0.5 (Dedalus, 1 core per run)
sbatch --array=0-99 motion_benchmarks/slurm/rbc3d_dedalus.sbatch
python -m motion_benchmarks.physics.merge_runs --inputs "rbc_runs/run_*.h5" --out data/rbc3d_ra2500.h5
#   (or Oceananigans.jl, their solver: physics/rbc3d_oceananigans.jl + convert_oceananigans.py)
# regular action, phase correlation on the mid-plane temperature (the dataset's default channel)
bash motion_benchmarks/slurm/run_suite.sh rbc3d --rbc_file data/rbc3d_ra2500.h5 --rbc_z_stride 4 \
    --fluid_action regular --residual lagrangian --input_frames 25 --pred_frames 25
# Galilean action with the physics connection
MODELS="lstm melstm_fp melstm_oracle" bash motion_benchmarks/slurm/run_suite.sh rbc3d \
    --rbc_file data/rbc3d_ra2500.h5 --rbc_z_stride 4 --fluid_action galilean
python -m motion_benchmarks.train_motion --dataset rbc3d --rbc_file data/rbc3d_ra2500.h5 \
    --fluid_action galilean --model melstm --velocity_source mean_flow ...
```
Height is treated as channels (4 fields × 32/`z_stride` heights) on the native 48×48 periodic
grid — do not put the recurrence behind a stride-8 autoencoder (a 1-px frame motion becomes
1/8 px in latent space). Fluids results add `nusselt_abs_err` and `ke_spectrum_lsd` per lead.
The D4-steerable variant (testing `𝒢_mot ⋊ 𝒢_glob`) is not included; it needs escnn and vector
field types for the velocity channels.

**Swift–Hohenberg** (`physics/swift_hohenberg.py`, cheap warm-up with a Goldstone mode):
hexagons by default, `--sh_g2 0` for rolls; banks are cached under `--sh_cache`.

**The Well** (`rayleigh_benard`: 2D, 512×128, periodic in x, walls in y, Ra 1e6–1e10):
```bash
pip install the_well && the-well-download --base-path data/the_well --dataset rayleigh_benard --split train
python -m motion_benchmarks.train_motion --dataset the_well --well_train <file>.hdf5 \
    --well_val <val file> --well_test <test file> --well_downsample 4,4 --fluid_action regular ...
```
Motion is horizontal only (walls in y); the models' circular padding is wrong across the walls —
a limitation to state. `--well_y_periodic` for fully periodic sets such as `shear_flow`.
`python -c "from motion_benchmarks.datasets.fluids import inspect_h5; inspect_h5('<file>')"`
prints a file's layout.

## E5 — GOES satellite clouds  (`--dataset satellite`)

```bash
python -m motion_benchmarks.scripts.download_goes --satellite 19 --band 13 \
    --start 2025-07-01T12 --hours 24 --crop 512 --downsample 4 --n_crops 8 \
    --event_frames 36 --out data/satellite/goes19_c13.h5
MODELS="lstm felstm melstm_fp" bash motion_benchmarks/slurm/run_suite.sh satellite \
    --sat_file data/satellite/goes19_c13.h5
```
Band 13 (10.3 µm) brightness temperature, 2 km → 8 km, every 5 min, day and night; model input
`(320 K − BT)/120 K`; CSI thresholds 260 / 235 / 210 K (colder = higher cloud). Different cloud
layers move with different winds: try `--num_vel_modes 2..3` and `--frame_pair_mode residual`.

## E6 — common-fate real silhouettes  (`train_cf_video.py`)

Every frame is band-limited noise; the subject exists only as a region moving coherently
(mask and texture share one velocity, sub-pixel; both layers move, never opposite, equal speed,
variance-renormalised blending, figure area ≈ 30%).
```bash
wget https://www.wisdom.weizmann.ac.il/~vision/VideoAnalysis/Demos/SpaceTimeActions/DB/classification_masks.mat
python -m motion_benchmarks.train_cf_video --dataset weizmann --weizmann_mat classification_masks.mat \
    --model melstm --num_vel_modes 4 --velocity_source frame_pair --readout_steps 6 \
    --forget_bias 0.44 --forget_bias_long 2.97 --test_subjects daria --val_subjects denis --control shuffle
bash motion_benchmarks/slurm/cf_weizmann_folds.sh classification_masks.mat    # leave-one-subject-out
# the articulation sweep: how far outside G = R^2 transport keeps paying
for amp in 0 1 2 4 6; do python -m motion_benchmarks.train_cf_video --dataset deform_mnist \
    --deform_amp $amp --dilate 2 --model melstm --num_vel_modes 4 ...; done
```
`K = 3–5` slots (piecewise-rigid body parts reach dense-flow parity), short + long memory
(two timescales), and a readout over the last `readout_steps` transported states (the
articulation *is* the label). Controls: `lstm` (in place, should sit near chance) and the
frame-shuffled test (`--control shuffle`). Without the .mat file a synthetic walker stands in.

## E7 — calcium imaging with injected motion  (`--dataset calcium`)

```bash
python -m motion_benchmarks.scripts.prepare_calcium --inputs suite2p/plane0 more_movies/*.tif --out data/calcium.h5
python -m motion_benchmarks.train_motion --dataset calcium --ca_file data/calcium.h5 \
    --ca_motion real --model melstm --velocity_source frame_pair ...
# or physiological motion (respiration, heartbeat, drift, locomotion bursts), or synthetic movies:
python -m motion_benchmarks.train_motion --dataset calcium --model melstm ...
```
Motion-corrected movies get real registration traces (Suite2p `xoff/yoff`, transplanted between
movies) or a physiological model re-applied, cropped so shifted content never wraps. The velocity
changes every frame (respiration is a sinusoid), so the frozen-velocity rollout is a poor
assumption here — frozen Lagrangian persistence is *worse* than Eulerian beyond a few frames;
report short leads and the tracked oracle. Phase correlation needs a Hann window and α ≈ 0.25
(set by the dataset).

---

## Tools

| script | purpose |
|---|---|
| `scripts/headroom.py` | Eulerian/Lagrangian persistence headroom per lead, with `--sweep option=v1,v2,...` |
| `scripts/pc_benchmark.py` | phase-correlation error vs α / window on any dataset with ground truth |
| `scripts/plot_results.py` | per-lead curves of several runs + baselines |
| `scripts/pysteps_baseline.py` | dense optical flow (LK) + semi-Lagrangian extrapolation |
| `scripts/prepare_radar.py`, `download_goes.py`, `prepare_trajectories.py`, `prepare_calcium.py` | data preparation |
| `physics/rbc3d_dedalus.py`, `rbc3d_oceananigans.jl`, `convert_oceananigans.py`, `merge_runs.py`, `swift_hohenberg.py` | simulations |

## Changes to existing files (all opt-in, defaults unchanged)

* `moving_mnist/velocity_predictor_model.py`: `PhaseCorrelation(alpha, subpixel, search_radius,
  suppress_radius)`. Defaults run the original code path (bit-identical, tested).
* `moving_mnist/train.py`, `train_classification.py`: `--distinct_worker_streams`.
* `moving_mnist/worker_seeding.py`: new, used by that flag.

## Layout

```
motion_benchmarks/
  common/     shifts, phase_correlation, schedules, fields, metrics, baselines, seeding
  models/     melstm_plus, felstm_plus, stabilizer, persistence, cf_classifier, gates, factory
  datasets/   base, registry, specs, radar_synthetic, archive (radar/satellite), fluids,
              trajectories, calcium, common_fate_video
  physics/    swift_hohenberg, rbc3d_dedalus, rbc3d_oceananigans.jl, convert_oceananigans, merge_runs
  scripts/    headroom, pc_benchmark, plot_results, pysteps_baseline, prepare_*, download_goes
  slurm/      train_motion.sbatch, run_suite.sh, rbc3d_dedalus.sbatch, cf_weizmann_folds.sh
  tests/      pytest suite
  train_motion.py, train_cf_video.py
```

Not verified end to end here (no network access to the data hosts from the build machine): the
GOES and SEVIR downloads themselves and the Oceananigans script. Their readers and converters
are tested on files in the same formats; the Dedalus script was run (small resolution, and
24×24×16 to t = 80: Nu ≈ 3, mean flow ≈ 0).
