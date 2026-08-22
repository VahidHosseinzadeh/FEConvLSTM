#!/usr/bin/env python
"""
Motion-difficulty sweep over trained checkpoints. Evaluation only.

Reuses eval_len_generalization -- the same function that produced the
len_gen_*.npz files -- so the rollout protocol, the copy-last baseline, the
boundary-age definition and the output schema are identical to the existing
results by construction rather than by re-implementation. Only the dataset
config varies between cells.

MELSTM runs the DEPLOYED protocol: _run_model is called with target_seq=None
and the model in eval(), so track_decoder_velocity resolves to False and the
final encoder velocity is frozen for the whole rollout. No oracle.

Sweep A  motion_difficulty in {0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0}, max_speed=2
Sweep B  named regimes at fixed parameters, off the d axis

Every cell freezes the velocity at the context boundary, so the rollout is
constant-velocity everywhere and the sweep isolates how well the context was
ENCODED under that motion regime.

    python moving_mnist/motion_difficulty_sweep.py --model_save_dir ./experiments
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from time_dependent_moving_mnist_dataset import TDMovingMNISTDataset  # noqa: E402
import train_eval_utils as teu                                        # noqa: E402
from train_eval_utils import build_model, eval_len_generalization      # noqa: E402

D_GRID = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)

# Fixed-parameter regimes. Not on the d axis: each pins one corner of the
# rate x jump-size plane that the scalar sweep only crosses diagonally.
NAMED_REGIMES = {
    # Fixed-parameter regimes pinning the corners of the (switching rate x jump
    # size) plane that the scalar d axis only crosses diagonally. Chosen so that
    # net context displacement stays coherent -- a regime whose digit vibrates in
    # place is an easier task, not a harder one, and would confound the result.
    #
    #                      rate   |dv|  net disp (px, of 27.5 at constant)
    "constant":       dict(motion_mode="constant"),
    #                     0.000   0.00   27.5   the floor: a pure flow
    "rare_teleport":  dict(motion_mode="stochastic", p_change=0.05,
                           transition_mode="uniform"),
    #                     0.049   2.42   23.7   low rate, max jump
    "regular_fast":   dict(motion_mode="piecewise", min_segment=2, max_segment=2,
                           transition_mode="smooth", smooth_probability=0.8),
    #                     0.337   1.40   19.1   REGULAR timing at a raised rate.
    #   Rate sits between the d=0.65 and d=0.80 cells, so comparing it against
    #   them separates "how often the velocity changes" from "how unpredictable
    #   the timing of those changes is".
    "random_walk":    dict(motion_mode="stochastic", p_change=1.0,
                           transition_mode="smooth", smooth_probability=1.0),
    #                     0.668   1.00   21.7   max rate, min jump
    "churn_half":     dict(motion_mode="stochastic", p_change=0.5,
                           transition_mode="uniform"),
    #                     0.500   2.43   10.9   high rate AND max jump -- the
    #   corner left empty by the others. Travel is reduced but not collapsed;
    #   p_change=1.0 here would drop it to 6.4 px, i.e. a digit vibrating in
    #   place, which is the confound Fix 1 removed from the d axis.
    "piecewise@0.5":  dict(motion_mode="piecewise", transition_mode="smooth",
                           min_segment=3, max_segment=6, smooth_probability=0.8),
    #   the ACTUAL training config. Not a panel-(b) category -- it is the
    #   validation point marked against the d curve, and should land near d=0.5.
}

REFERENCE_REGIME = "d=0.00"


def cells(which):
    """(regime, difficulty, dataset kwargs) for every cell in the sweep.

    Every cell uses neighbor_kernel="symmetric": the legacy kernel is
    degree-biased, so mean |v| drifts with the switching rate and difficulty
    gets confounded with speed. Sweep B needs it too -- random_walk routes every
    change through the neighbour kernel, so under the legacy kernel its realised
    rate (1.00) is not comparable with the d axis (0.67 at the same p_change).
    """
    out = []
    if which in ("all", "a"):
        for d in D_GRID:
            out.append((f"d={d:.2f}", float(d),
                        dict(motion_difficulty=d, neighbor_kernel="symmetric")))
    if which in ("all", "b"):
        for name, kw in NAMED_REGIMES.items():
            out.append((name, None, dict(kw, neighbor_kernel="symmetric")))
    return out


def slug(regime):
    """Filesystem- and glob-safe form of a regime label."""
    return re.sub(r"[^A-Za-z0-9]+", "_", regime).strip("_")


def find_runs(save_dir, models):
    """One (config, run_id, checkpoint) per model -- same lookup as the sweep."""
    results_dir = os.path.join(save_dir, "results")
    models_dir = os.path.join(save_dir, "models")
    runs = {}
    for name in models:
        hits = sorted(glob.glob(os.path.join(results_dir, f"history_{name}_*.json")))
        if not hits:
            raise SystemExit(f"no history_{name}_*.json in {results_dir}")
        cfg = json.loads(open(hits[-1]).read())["config"]
        run_id = os.path.basename(hits[-1])[len(f"history_{name}_"):-len(".json")]
        ckpt = os.path.join(models_dir, f"{name}_best_model_{run_id}.pth")
        if not os.path.exists(ckpt):
            alt = os.path.join(models_dir, f"{name}_best.pth")
            if not os.path.exists(alt):
                raise SystemExit(f"no checkpoint for {name}: tried {ckpt} and {alt}")
            print(f"WARNING: {os.path.basename(ckpt)} missing, falling back to "
                  f"{os.path.basename(alt)}")
            ckpt = alt
        runs[name] = (cfg, run_id, ckpt)
    return runs


def make_dataset(kwargs, args):
    """The benchmark generator with one cell's motion config applied.

    max_speed is fixed for every cell: difficulty must never turn into the
    trivial 'the velocity left the lattice' result, which is a separate axis.
    """
    return TDMovingMNISTDataset(
        root=args.root, train=False,                 # MNIST TEST split
        seq_len=args.gen_seq_len, num_digits=2, image_size=args.image_size,
        max_speed=args.max_speed,
        freeze_after=args.gen_input_frames,
        min_center_distance=20, reject_overlap=True,
        require_distinct_velocities=True,
        return_motion=True, return_positions=False,
        transform=None, download=args.download,
        random=False, seed=args.data_seed, max_tries=300,
        **kwargs)


def materialise(ds, n):
    """Draw n sequences into tensors.

    With random=False the dataset RNG is stateful and every __getitem__ advances
    it, so scoring three models by iterating the dataset three times would hand
    each of them DIFFERENT sequences. Materialising once removes the trap
    structurally: all three models are then scored on the identical tensors, and
    the paired statistics are valid by construction rather than by discipline.
    """
    ds.reset_rng()
    seqs, labels, motions = [], [], []
    for i in range(n):
        s, l, m = ds[i][:3]
        seqs.append(s)
        labels.append(torch.as_tensor(l))
        motions.append(m)
    return torch.stack(seqs), torch.stack(labels), torch.stack(motions)


def motion_stats(motion_schedule, input_frames, max_speed):
    """Realised statistics of the velocity process over the CONTEXT transitions.

    _apply_freeze makes motions[f-1:] constant, so only t = 1 .. f-2 are free.
    Boundary age counts from the last change to the boundary the same way
    eval_len_generalization does.
    """
    m = np.asarray(motion_schedule)                       # (N, T, D, 2)
    f = input_frames
    last = f - 2
    grid = [(vx, vy) for vx in range(-max_speed, max_speed + 1)
            for vy in range(-max_speed, max_speed + 1) if (vx, vy) != (0, 0)]
    index = {v: i for i, v in enumerate(grid)}
    joint = np.zeros((len(grid), len(grid)), dtype=np.int64)

    ages, jumps, n_change, n_trans = [], [], 0, 0
    for seq in m:
        for dgt in range(seq.shape[1]):
            v = seq[:, dgt, :]
            last_change = 0
            for t in range(1, last + 1):
                a, b = (int(v[t - 1][0]), int(v[t - 1][1])), (int(v[t][0]), int(v[t][1]))
                if a in index and b in index:
                    joint[index[a], index[b]] += 1
                n_trans += 1
                if a != b:
                    n_change += 1
                    last_change = t
                    jumps.append(max(abs(b[0] - a[0]), abs(b[1] - a[1])))
            ages.append(last - last_change + 1)

    row = joint.sum(axis=1)
    total = row.sum()
    entropy = 0.0
    if total:
        for k in np.flatnonzero(row):
            p = joint[k][joint[k] > 0] / row[k]
            entropy += (row[k] / total) * float(-(p * np.log2(p)).sum())
    p_emp = n_change / n_trans if n_trans else 0.0
    return dict(
        p_change=p_emp,
        mean_gap=(1.0 / p_emp) if p_emp > 0 else float("inf"),
        mean_dv=float(np.mean(jumps)) if jumps else 0.0,
        mean_age=float(np.mean(ages)) if ages else float("nan"),
        entropy=entropy,
    )


def summarise(per_seq_err, short_end):
    e_short = per_seq_err[:, :short_end].mean(axis=1)
    e_long = per_seq_err.mean(axis=1)
    q = lambda v: (float(np.median(v)),
                   float(np.percentile(v, 25)), float(np.percentile(v, 75)))
    return dict(short=q(e_short), long=q(e_long), e_short=e_short, e_long=e_long)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model_save_dir', default='./experiments')
    p.add_argument('--models', nargs='+', default=['lstm', 'felstm', 'melstm'],
                   choices=['lstm', 'felstm', 'melstm'])
    p.add_argument('--sweep', default='all', choices=['all', 'a', 'b'])
    p.add_argument('--n_sequences', type=int, default=2000)
    p.add_argument('--batch_size', type=int, default=100)
    p.add_argument('--root', default='./data')
    p.add_argument('--data_seed', type=int, default=42)
    p.add_argument('--gen_seq_len', type=int, default=100)
    p.add_argument('--gen_input_frames', type=int, default=15)
    p.add_argument('--image_size', type=int, default=36)
    p.add_argument('--max_speed', type=int, default=2)
    p.add_argument('--download', action='store_true', default=False)
    p.add_argument('--device', default=None)
    p.add_argument('--out_name', default='motion_sweep')
    args = p.parse_args()

    # eval_len_generalization logs qualitative panels to wandb on its first
    # batch. Disable the run and stub the loggers: this touches the logging
    # side-channel only, never the numerical path.
    os.environ.setdefault("WANDB_MODE", "disabled")
    teu.log_sequence_predictions_new = lambda *a, **k: None
    teu.log_state_evolution = lambda *a, **k: None
    teu.log_velocity_report = lambda *a, **k: None

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')   # MPS measured ~2x slower here

    runs = find_runs(args.model_save_dir, args.models)
    out_dir = os.path.join(args.model_save_dir, args.out_name)
    os.makedirs(out_dir, exist_ok=True)

    ref_cfg = runs[args.models[0]][0]
    trained_pred = ref_cfg["seq_len"] - ref_cfg["input_frames"]
    ctx = args.gen_input_frames
    pred_len = args.gen_seq_len - ctx
    short_end = trained_pred          # t = 16 .. 16+trained_pred-1

    loaded = {}
    for name, (cfg, run_id, ckpt) in runs.items():
        m = build_model(cfg)
        m.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=True)
        loaded[name] = m.to(device).eval()

    todo = cells(args.sweep)
    print(f"device            : {device}", flush=True)
    print(f"models            : {', '.join(args.models)}")
    print(f"cells             : {len(todo)}  ({', '.join(r for r, _, _ in todo)})")
    print(f"sequences/cell    : {args.n_sequences}")
    print(f"rollout           : {ctx} context -> {pred_len} predicted "
          f"(trained horizon {trained_pred})")
    print(f"short horizon     : t = {ctx + 1} .. {ctx + short_end}")
    print(f"output            : {out_dir}\n", flush=True)

    rows = []
    t0 = time.time()
    for regime, difficulty, kwargs in todo:
        ds = make_dataset(kwargs, args)
        seqs, labels, motions = materialise(ds, args.n_sequences)

        # The paired statistics require every model to see identical frames.
        # Regenerate and assert, the way the filmstrip loader asserts on strip_gt.
        chk_seqs, _, chk_mot = materialise(make_dataset(kwargs, args), args.n_sequences)
        assert torch.equal(seqs, chk_seqs) and torch.equal(motions, chk_mot), (
            f"{regime}: regeneration is not reproducible; the cell's sequences "
            f"are not paired across models")

        loader = DataLoader(TensorDataset(seqs, labels, motions),
                            batch_size=args.batch_size, shuffle=False, num_workers=0)
        stats = motion_stats(motions.numpy(), ctx, args.max_speed)

        for name in args.models:
            cfg, run_id, _ = runs[name]
            mean, std, details = eval_len_generalization(
                loaded[name], loader, device, ctx, n_strip_sequences=3)

            payload = dict(
                mean=mean, std=std,
                model=name, run_id=run_id,
                hidden_size=cfg["hidden_size"],
                gen_input_frames=ctx,
                gen_seq_len=args.gen_seq_len,
                train_input_frames=ref_cfg["input_frames"],
                train_pred_frames=trained_pred,
                n_sequences=args.n_sequences,
                max_speed=args.max_speed,
                regime=regime,
                neighbor_kernel=kwargs.get("neighbor_kernel", "legacy"),
                # npz has no None; NaN encodes "not on the d axis"
                difficulty=np.float64(difficulty if difficulty is not None else np.nan),
                motion_schedule=motions.numpy().astype(np.int16),
                motion_p_change=stats["p_change"],
                motion_mean_gap=stats["mean_gap"],
                motion_mean_dv=stats["mean_dv"],
                motion_mean_age=stats["mean_age"],
                motion_entropy=stats["entropy"],
            )
            payload.update(details)
            path = os.path.join(
                out_dir, f"{args.out_name}_{name}_{slug(regime)}_{run_id}.npz")
            np.savez_compressed(path, **payload)

            s = summarise(details["per_seq_err"], short_end)
            rows.append(dict(regime=regime, model=name, n=args.n_sequences,
                             short=s["short"], long=s["long"], **stats))
            print(f"  [{time.time()-t0:6.0f}s] {regime:<14} {name:<7} "
                  f"short {s['short'][0]:.3e}  long {s['long'][0]:.3e}",
                  flush=True)      # unbuffered: this is a multi-hour cluster job

    report(rows, args)


def report(rows, args):
    by = {(r["regime"], r["model"]): r for r in rows}
    regimes = list(dict.fromkeys(r["regime"] for r in rows))
    models = list(dict.fromkeys(r["model"] for r in rows))

    print("\n" + "=" * 108)
    print("MOTION SWEEP  --  median per-sequence MSE, and each model normalised "
          f"by ITS OWN {REFERENCE_REGIME} cell")
    print("=" * 108)
    hdr = (f"{'regime':<15}{'n':>6}{'gap':>8}{'|dv|':>7}{'age':>7}{'H bits':>8}  ")
    hdr += "".join(f"{m + ' short':>14}{m + ' norm':>12}" for m in models)
    print(hdr)
    print("-" * len(hdr))
    for reg in regimes:
        any_row = next(r for r in rows if r["regime"] == reg)
        gap = "   inf" if np.isinf(any_row["mean_gap"]) else f"{any_row['mean_gap']:6.2f}"
        line = (f"{reg:<15}{any_row['n']:>6}{gap:>8}{any_row['mean_dv']:>7.2f}"
                f"{any_row['mean_age']:>7.2f}{any_row['entropy']:>8.3f}  ")
        for m in models:
            r = by.get((reg, m))
            ref = by.get((REFERENCE_REGIME, m))
            if r is None:
                line += f"{'-':>14}{'-':>12}"
                continue
            norm = (r["short"][0] / ref["short"][0]) if ref else float("nan")
            line += f"{r['short'][0]:>14.3e}{norm:>12.2f}"
        print(line)
    print("\nnormalised = this model's median at this regime / its own median at "
          f"{REFERENCE_REGIME}; 1.00 means the regime costs it nothing.")


if __name__ == "__main__":
    main()
