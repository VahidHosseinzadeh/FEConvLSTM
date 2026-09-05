#!/usr/bin/env python3
"""
Motion-generalization experiment: how does rollout error behave when the
*motion distribution* differs from the one a model was trained on?

Run AFTER training, against the checkpoints train.py wrote. For each requested
model and each motion preset it rebuilds the length-generalisation benchmark
with that motion, rolls the model out autoregressively, and stores the full
per-sequence per-timestep error matrix.

Every preset keeps freeze_after = gen_input_frames, exactly as train.py's
length-generalisation benchmark does, so the ROLLOUT is always constant-velocity
extrapolation -- the same kind of task for every model. What the presets vary is

  (a) how hard the frozen velocity is to INFER from the context
      (ctx_constant -> ctx_jumpy -> ctx_late_change -> ctx_churn), and
  (b) how large that velocity is once inferred (speed_1 .. speed_4).

Architecture and data geometry are read from the run's own
results/history_<model>_<runid>.json, so this cannot rebuild a different
network than was trained.

Layout (mirrors train.py's --model_save_dir):
    <save_dir>/models/<model>_best_model_<runid>.pth     read
    <save_dir>/results/history_<model>_<runid>.json      read
    <save_dir>/motion_gen/motion_gen_<model>_<runid>.npz written

Example
-------
    python moving_mnist/motion_generalization.py \
        --model_save_dir ./experiments --n_sequences 10000 --batch_size 100
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from time_dependent_moving_mnist_dataset import TDMovingMNISTDataset
from train_eval_utils import build_model

# --------------------------------------------------------------------------
# Motion presets. Each dict lists ONLY the fields it changes; anything absent
# keeps the run's own trained value, so "as_trained" is the reference.
#
# Only parameters the chosen motion_mode actually reads are listed, because the
# generator ignores the rest and listing them suggests an effect that does not
# exist:
#     min_segment/max_segment : piecewise, accelerate
#     p_change                : stochastic only
#     transition_mode         : piecewise, stochastic
#     smooth_probability      : only when transition_mode == "smooth"
#                               ("smooth" at probability 0.0 == "uniform")
#
# Every preset keeps freeze_after = gen_input_frames, so the ROLLOUT is always
# constant-velocity extrapolation. The presets vary two independent things:
#   (a) how hard the frozen velocity is to INFER from the context
#   (b) how large that velocity is, i.e. whether an enumerated model can
#       represent it at all
# --------------------------------------------------------------------------
MOTION_PRESETS = {
    "as_trained":         {},                    # reference: the trained distribution

    # ---- (a) inference difficulty, magnitude held at the trained range ----
    "static_velocity":    dict(motion_mode="constant"),
    "abrupt_jumps":       dict(transition_mode="uniform"),
    "change_at_boundary": dict(min_segment=1, max_segment=2),
    "rapid_changes":      dict(motion_mode="stochastic", transition_mode="uniform",
                               p_change=0.6),

    # ---- (b) magnitude, context behaviour held as trained ----------------
    "slow_v1":            dict(data_v_range=1),
    "fast_v3":            dict(data_v_range=3),
    "fast_v4":            dict(data_v_range=4),

    # ---- crossed: large velocity that is trivial to infer -----------------
    # Separates "the velocity is unrepresentable" from "the velocity is hard to
    # read off the context" -- constant context removes the inference problem
    # entirely while keeping the magnitude far outside a |v| <= 2 lattice.
    "fast_v4_static":     dict(motion_mode="constant", data_v_range=4),

    # ---- changing speed: |v| ramps toward the extremes during the context --
    "accelerating_v4":    dict(motion_mode="accelerate", data_v_range=4),

    # ======================================================================
    # Extended sweep. The ten above are the paper set (PAPER_MOTIONS below);
    # everything here fills in the axes between them. All of it is opt-in per
    # run via --motions, because each preset costs a full pass over
    # --n_sequences for every model.
    # ======================================================================

    # ---- (a) inference difficulty, finer grained -------------------------
    # How often the velocity changes, from "almost never" to "every frame",
    # holding the magnitude at the trained range.
    #
    # slow_changes: segments long enough to change only about once inside a
    # 15-frame context. Longer ones (15-30) never change there at all and just
    # duplicate static_velocity.
    "slow_changes":       dict(min_segment=7, max_segment=14),
    "change_every_step":  dict(min_segment=1, max_segment=1),
    "stochastic_rare":    dict(motion_mode="stochastic", p_change=0.05),
    "stochastic_mid":     dict(motion_mode="stochastic", p_change=0.30),

    # How far each change jumps: "smooth" at probability 1.0 only ever steps
    # to an adjacent velocity, which is the gentlest non-constant context.
    "always_smooth":      dict(smooth_probability=1.0),
    "half_smooth":        dict(smooth_probability=0.5),

    # ---- (b) magnitude, finer grained ------------------------------------
    "fast_v5":            dict(data_v_range=5),
    "fast_v6":            dict(data_v_range=6),

    # ---- crossed: magnitude x inference difficulty -----------------------
    # The paper set only crosses "trivial to infer" with v4. These fill the
    # rest of the square, separating "cannot represent this velocity" from
    # "cannot read this velocity off the context" at several magnitudes.
    "slow_v1_static":     dict(motion_mode="constant", data_v_range=1),
    "fast_v3_static":     dict(motion_mode="constant", data_v_range=3),
    "fast_v6_static":     dict(motion_mode="constant", data_v_range=6),
    "fast_v4_jumpy":      dict(transition_mode="uniform", data_v_range=4),
    "fast_v4_boundary":   dict(min_segment=1, max_segment=2, data_v_range=4),
    "fast_v4_churn":      dict(motion_mode="stochastic", transition_mode="uniform",
                               p_change=0.6, data_v_range=4),

    # ---- changing speed at other magnitudes ------------------------------
    "accelerating_v2":    dict(motion_mode="accelerate"),
    "accelerating_v6":    dict(motion_mode="accelerate", data_v_range=6),
}

# The presets the paper figures use. --motions defaults to this rather than to
# every preset above, so adding to the extended sweep never silently multiplies
# the cost of a default run.
PAPER_MOTIONS = [
    "as_trained", "static_velocity", "abrupt_jumps", "change_at_boundary",
    "rapid_changes", "slow_v1", "fast_v3", "fast_v4", "fast_v4_static",
    "accelerating_v4",
]


def find_runs(save_dir, models):
    """Locate one (config, run_id, checkpoint) triple per requested model."""
    results_dir = os.path.join(save_dir, "results")
    models_dir = os.path.join(save_dir, "models")
    runs = {}
    for name in models:
        hits = sorted(glob.glob(os.path.join(results_dir, f"history_{name}_*.json")))
        if not hits:
            raise SystemExit(
                f"no history_{name}_*.json in {results_dir}\n"
                f"Train first, or pass --model_save_dir pointing at the run output.")
        if len(hits) > 1:
            print(f"WARNING: {len(hits)} histories for {name}; using the newest "
                  f"({os.path.basename(hits[-1])})")
        cfg = json.loads(open(hits[-1]).read())["config"]
        run_id = os.path.basename(hits[-1])[len(f"history_{name}_"):-len(".json")]

        ckpt = os.path.join(models_dir, f"{name}_best_model_{run_id}.pth")
        if not os.path.exists(ckpt):
            alt = os.path.join(models_dir, f"{name}_best.pth")
            if not os.path.exists(alt):
                raise SystemExit(f"no checkpoint for {name}: tried {ckpt} and {alt}")
            print(f"WARNING: {os.path.basename(ckpt)} missing, falling back to "
                  f"{os.path.basename(alt)} (may be from a different run)")
            ckpt = alt
        runs[name] = (cfg, run_id, ckpt)
    return runs


def make_dataset(cfg, preset, args):
    """The length-gen benchmark generator with one motion preset applied.

    freeze_after = gen_input_frames matches train.py's gen_test_dataset, so the
    rollout is constant-velocity for every preset.
    """
    m = dict(
        motion_mode=cfg.get("motion_mode", "piecewise"),
        transition_mode=cfg.get("transition_mode", "smooth"),
        min_segment=cfg.get("min_segment", 3),
        max_segment=cfg.get("max_segment", 6),
        p_change=cfg.get("p_change", 0.25),
        smooth_probability=cfg.get("smooth_probability", 0.8),
        data_v_range=cfg.get("data_v_range", 2),
    )
    m.update(MOTION_PRESETS[preset])

    return TDMovingMNISTDataset(
        root=args.root, train=False,
        seq_len=args.gen_seq_len, num_digits=2,
        image_size=cfg["image_size"], max_speed=m.pop("data_v_range"),
        motion_difficulty=None, freeze_after=args.gen_input_frames,
        min_center_distance=20, reject_overlap=True,
        require_distinct_velocities=True,
        return_motion=False, return_positions=False,
        transform=None, download=True,
        random=False, seed=args.data_seed, max_tries=300, **m)


@torch.no_grad()
def rollout_error(model, seqs, ctx, pred_len, device, batch_size):
    """(N, seq_len, ...) -> (N, pred_len) per-sequence per-timestep MSE.

    target_seq is never passed, i.e. the deployable protocol: MEConvLSTM
    freezes its last encoder velocity rather than tracking future frames.
    """
    out = []
    for b in range(0, seqs.shape[0], batch_size):
        chunk = seqs[b:b + batch_size].to(device)
        pred = model(chunk[:, :ctx], pred_len=pred_len)
        if isinstance(pred, tuple):
            pred = pred[0]
        out.append(((pred - chunk[:, ctx:]) ** 2)
                   .mean(dim=(2, 3, 4)).float().cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    p = argparse.ArgumentParser(
        description="Motion-generalization sweep over trained checkpoints")
    p.add_argument('--model_save_dir', type=str, default='./experiments',
                   help="Run output root written by train.py (models/, results/); "
                        "motion_gen/ is created alongside them")
    p.add_argument('--models', nargs='+', default=['lstm', 'felstm', 'melstm'],
                   choices=['lstm', 'felstm', 'melstm'])
    p.add_argument('--motions', nargs='+', default=PAPER_MOTIONS,
                   choices=['all'] + list(MOTION_PRESETS),
                   help='Motion presets to sweep. Defaults to the paper set; '
                        'pass "all" for every preset in MOTION_PRESETS.')
    p.add_argument('--n_sequences', type=int, default=10000,
                   help='Benchmark sequences per motion preset')
    p.add_argument('--batch_size', type=int, default=100,
                   help='Rollout batch; lower it if the GPU runs out of memory')
    p.add_argument('--root', type=str, default='./data', help='MNIST download root')
    p.add_argument('--data_seed', type=int, default=42,
                   help='Fixed benchmark seed; keep it constant across models')
    p.add_argument('--gen_seq_len', type=int, default=None,
                   help='Default: the value the run was evaluated at')
    p.add_argument('--gen_input_frames', type=int, default=None,
                   help='Default: the value the run was evaluated at')
    p.add_argument('--device', type=str, default=None,
                   help="cuda / mps / cpu (default: best available)")
    args = p.parse_args()
    if 'all' in args.motions:
        args.motions = list(MOTION_PRESETS)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    runs = find_runs(args.model_save_dir, args.models)
    out_dir = os.path.join(args.model_save_dir, "motion_gen")
    os.makedirs(out_dir, exist_ok=True)

    # Geometry is shared across models so every model sees identical sequences;
    # take it from the first run and verify the rest agree.
    ref_cfg = runs[args.models[0]][0]
    if args.gen_seq_len is None:
        args.gen_seq_len = ref_cfg["gen_seq_len"]
    if args.gen_input_frames is None:
        args.gen_input_frames = ref_cfg["gen_input_frames"]
    for name, (cfg, _, _) in runs.items():
        for k in ("image_size", "gen_seq_len", "gen_input_frames"):
            if cfg[k] != ref_cfg[k]:
                print(f"WARNING: {name} was evaluated with {k}={cfg[k]} but this "
                      f"sweep uses {ref_cfg[k]}; results are not paired")

    ctx = args.gen_input_frames
    pred_len = args.gen_seq_len - ctx
    trained_pred = ref_cfg["seq_len"] - ref_cfg["input_frames"]

    print(f"device            : {device}")
    print(f"models            : {', '.join(args.models)}")
    print(f"motions           : {', '.join(args.motions)}")
    print(f"sequences/motion  : {args.n_sequences}")
    print(f"rollout           : {ctx} context -> {pred_len} predicted "
          f"(trained horizon {trained_pred}, {pred_len / trained_pred:.1f}x)")
    print(f"output            : {out_dir}\n")

    loaded = {}
    for name, (cfg, run_id, ckpt) in runs.items():
        m = build_model(cfg)
        m.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=True)
        loaded[name] = m.to(device).eval()
        print(f"loaded {name:<7} {os.path.basename(ckpt)}  "
              f"({sum(q.numel() for q in m.parameters()):,} params)")
    print()

    store = {name: {} for name in args.models}
    t0 = time.time()
    for preset in args.motions:
        # One dataset per preset, materialised once so every model is scored on
        # the identical sequences (paired comparison).
        ds = make_dataset(ref_cfg, preset, args)
        ds.reset_rng()
        seqs = torch.stack([ds[i][0] for i in range(args.n_sequences)])

        for name in args.models:
            err = rollout_error(loaded[name], seqs, ctx, pred_len,
                                device, args.batch_size)
            store[name][preset] = err
            print(f"{preset:<16} {name:<7} mean={err.mean():.4e}  "
                  f"final={err[:, -1].mean():.4e}  [{(time.time()-t0)/60:5.1f} min]",
                  flush=True)
        del seqs

    for name in args.models:
        cfg, run_id, _ = runs[name]
        path = os.path.join(out_dir, f"motion_gen_{name}_{run_id}.npz")
        np.savez_compressed(
            path,
            model=name, run_id=run_id,
            motions=np.array(args.motions),
            n_sequences=args.n_sequences,
            gen_input_frames=ctx, gen_seq_len=args.gen_seq_len,
            train_input_frames=cfg["input_frames"],
            train_pred_frames=trained_pred,
            hidden_size=cfg["hidden_size"],
            data_seed=args.data_seed,
            # one (n_sequences, pred_len) matrix per preset -- full per-sequence
            # detail so mean/std or median/IQR can be chosen when plotting
            **{f"err_{p}": store[name][p] for p in args.motions})
        print(f"\nSaved motion-generalization results to {path}")

    print(f"\n{'motion':<16}" + "".join(f"{n:>14}" for n in args.models))
    for preset in args.motions:
        print(f"{preset:<16}" + "".join(f"{store[n][preset].mean():>14.3e}"
                                        for n in args.models))
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
