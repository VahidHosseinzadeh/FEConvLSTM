#!/usr/bin/env python
"""
Three one-factor motion sweeps over trained checkpoints. Evaluation only.

Separate from motion_difficulty_sweep.py: that experiment varied one scalar and a
set of named regimes; this one crosses three orthogonal axes through a single
shared centre, so each panel isolates one factor.

Reuses eval_len_generalization -- the function that produced the len_gen_*.npz
files -- so the rollout protocol, copy-last baseline, boundary-age definition and
output schema match the existing results by construction. MELSTM runs the
DEPLOYED protocol (target_seq=None in eval(), final encoder velocity frozen).

    Axis A  switching RATE   p_change = d ** 2.170, s = 0.80, max_speed = 2
    Axis B  JUMP SIZE        s swept, p_change compensated to hold the realised
                             rate fixed, max_speed = 2
    Axis C  SPEED RANGE      max_speed swept, s = 0.80, p_change compensated

Every cell freezes velocity at the context boundary, so the rollout is
constant-velocity throughout and the sweep isolates context ENCODING.

    python moving_mnist/motion_factors_sweep.py --model_save_dir ./experiments
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

D_GRID = (0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00)
S_GRID = (1.0, 0.9, 0.6, 0.4, 0.2, 0.0)          # 0.8 is the shared centre
N_GRID = (1, 3, 4, 5)                            # 2 is the shared centre

TRAIN_D = 0.50          # the training regime on the rate axis
TRAIN_S = 0.80
TRAIN_N = 2
UNIT_OFFSETS = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]


def velocity_grid(max_speed):
    return [(a, b) for a in range(-max_speed, max_speed + 1)
            for b in range(-max_speed, max_speed + 1) if (a, b) != (0, 0)]


def acceptance(max_speed):
    """P(a uniform unit-offset proposal stays on the grid), averaged over states.

    The symmetric neighbour kernel HOLDS on a proposal that leaves the grid, so a
    held proposal is not a velocity change. This is the factor that makes the
    realised switching rate lower than p_change.
    """
    g = set(velocity_grid(max_speed))
    return float(np.mean([sum((v[0] + o[0], v[1] + o[1]) in g for o in UNIT_OFFSETS) / 8
                          for v in g]))


def realised_rate(p_change, s, max_speed):
    """p_change x P(the update actually changes v).

    With probability s the update is a neighbour proposal (accepted with
    probability `acceptance`); otherwise it is uniform over the grid excluding
    the current velocity, which always changes.
    """
    return p_change * (1.0 - s * (1.0 - acceptance(max_speed)))


def compensated_p(target, s, max_speed):
    """The p_change that yields `target` realised rate at this (s, max_speed)."""
    return target / (1.0 - s * (1.0 - acceptance(max_speed)))


# The centre of all three axes is the TRAINING configuration itself, so the
# target rate is derived from it rather than being a round number: this makes
# axis A's d=0.50 cell, axis B's s=0.80 cell and axis C's N=2 cell literally the
# SAME cell, generated once and shared by all three panels.
TARGET_RATE = realised_rate(TRAIN_D ** 2.170, TRAIN_S, TRAIN_N)


def cells(which="all"):
    """(regime, axis, level, dataset kwargs) for every cell.

    The shared centre carries axis="centre"; the plotting code places it at
    d=0.50 / s=0.80 / N=2 in the three panels respectively.
    """
    base = dict(neighbor_kernel="symmetric")
    out = [("centre", "centre", float("nan"),
            dict(base, motion_mode="stochastic", transition_mode="smooth",
                 p_change=TRAIN_D ** 2.170, smooth_probability=TRAIN_S,
                 max_speed=TRAIN_N))]
    if which in ("all", "a"):
        for d in D_GRID:
            if d == TRAIN_D:
                continue                      # that is the centre
            out.append((f"A:d={d:.2f}", "rate", float(d),
                        dict(base, motion_mode="stochastic", transition_mode="smooth",
                             p_change=d ** 2.170, smooth_probability=TRAIN_S,
                             max_speed=TRAIN_N)))
    if which in ("all", "b"):
        for s in S_GRID:
            out.append((f"B:s={s:.2f}", "jump", float(s),
                        dict(base, motion_mode="stochastic", transition_mode="smooth",
                             p_change=compensated_p(TARGET_RATE, s, TRAIN_N),
                             smooth_probability=s, max_speed=TRAIN_N)))
    if which in ("all", "c"):
        for n in N_GRID:
            out.append((f"C:N={n}", "speed", float(n),
                        dict(base, motion_mode="stochastic", transition_mode="smooth",
                             p_change=compensated_p(TARGET_RATE, TRAIN_S, n),
                             smooth_probability=TRAIN_S, max_speed=n)))
    return out


def slug(regime):
    return re.sub(r"[^A-Za-z0-9]+", "_", regime).strip("_")


def find_runs(save_dir, models):
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
                raise SystemExit(f"no checkpoint for {name}")
            print(f"WARNING: falling back to {os.path.basename(alt)}")
            ckpt = alt
        runs[name] = (cfg, run_id, ckpt)
    return runs


def make_dataset(kwargs, args):
    return TDMovingMNISTDataset(
        root=args.root, train=False,                  # MNIST TEST split
        seq_len=args.gen_seq_len, num_digits=2, image_size=args.image_size,
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
    it, so iterating three times would hand each model DIFFERENT sequences.
    Materialising once makes the paired comparison true by construction.
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
    """Realised statistics over the CONTEXT transitions (motions[0 .. f-2])."""
    v = np.asarray(motion_schedule, dtype=np.int64)
    f = input_frames
    ctx = v[:, :f - 1]
    changed = (ctx[:, 1:] != ctx[:, :-1]).any(axis=-1)
    dv = np.abs(ctx[:, 1:] - ctx[:, :-1]).max(axis=-1)
    speed = np.sqrt((ctx ** 2).sum(axis=-1))
    net = np.sqrt((ctx.sum(axis=1) ** 2).sum(axis=-1))
    idx = np.arange(1, f - 1).reshape(1, -1, 1)
    age = (f - 2 - np.where(changed, idx, 0).max(axis=1)) + 1

    grid = velocity_grid(max_speed)
    index = {u: i for i, u in enumerate(grid)}
    joint = np.zeros((len(grid), len(grid)), dtype=np.int64)
    a = ctx[:, :-1].reshape(-1, 2)
    b = ctx[:, 1:].reshape(-1, 2)
    for pa, pb in zip(map(tuple, a), map(tuple, b)):
        if pa in index and pb in index:
            joint[index[pa], index[pb]] += 1
    row = joint.sum(axis=1)
    total = row.sum()
    entropy = 0.0
    if total:
        for k in np.flatnonzero(row):
            q = joint[k][joint[k] > 0] / row[k]
            entropy += (row[k] / total) * float(-(q * np.log2(q)).sum())

    rate = float(changed.mean())
    return dict(rate=rate,
                mean_gap=(1.0 / rate) if rate > 0 else float("inf"),
                mean_dv=float(dv[changed].mean()) if changed.any() else 0.0,
                mean_speed=float(speed.mean()),
                net_disp=float(net.mean()),
                mean_age=float(age.mean()),
                entropy=entropy)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model_save_dir', default='./experiments')
    p.add_argument('--models', nargs='+', default=['lstm', 'felstm', 'melstm'],
                   choices=['lstm', 'felstm', 'melstm'])
    p.add_argument('--sweep', default='all', choices=['all', 'a', 'b', 'c'])
    p.add_argument('--n_sequences', type=int, default=5000)
    p.add_argument('--batch_size', type=int, default=100)
    p.add_argument('--root', default='./data')
    p.add_argument('--data_seed', type=int, default=42)
    p.add_argument('--gen_seq_len', type=int, default=100)
    p.add_argument('--gen_input_frames', type=int, default=15)
    p.add_argument('--image_size', type=int, default=36)
    p.add_argument('--rate_tol', type=float, default=0.005,
                   help='allowed drift of the realised rate on axes B and C')
    p.add_argument('--download', action='store_true', default=False)
    p.add_argument('--device', default=None)
    p.add_argument('--out_name', default='motion_factors')
    args = p.parse_args()

    # eval_len_generalization logs qualitative panels to wandb on its first batch.
    # Stub the loggers: this touches the logging side-channel only.
    os.environ.setdefault("WANDB_MODE", "disabled")
    teu.log_sequence_predictions_new = lambda *a, **k: None
    teu.log_state_evolution = lambda *a, **k: None
    teu.log_velocity_report = lambda *a, **k: None

    device = (torch.device(args.device) if args.device else
              torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    runs = find_runs(args.model_save_dir, args.models)
    out_dir = os.path.join(args.model_save_dir, args.out_name)
    os.makedirs(out_dir, exist_ok=True)

    ref_cfg = runs[args.models[0]][0]
    trained_pred = ref_cfg["seq_len"] - ref_cfg["input_frames"]
    ctx = args.gen_input_frames
    short_end = trained_pred

    loaded = {}
    for name, (cfg, run_id, ckpt) in runs.items():
        m = build_model(cfg)
        m.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=True)
        loaded[name] = m.to(device).eval()

    todo = cells(args.sweep)
    print(f"device            : {device}", flush=True)
    print(f"cells             : {len(todo)}", flush=True)
    print(f"sequences/cell    : {args.n_sequences}", flush=True)
    print(f"target realised rate (all of axis B and C, and the centre): "
          f"{TARGET_RATE:.4f}", flush=True)
    print(f"short horizon     : t = {ctx + 1} .. {ctx + short_end}", flush=True)
    print(f"output            : {out_dir}\n", flush=True)

    rows, t0 = [], time.time()
    for regime, axis, level, kwargs in todo:
        ds = make_dataset(kwargs, args)
        seqs, labels, motions = materialise(ds, args.n_sequences)
        chk, _, chk_m = materialise(make_dataset(kwargs, args), args.n_sequences)
        assert torch.equal(seqs, chk) and torch.equal(motions, chk_m), (
            f"{regime}: regeneration not reproducible; cells are not paired")

        stats = motion_stats(motions.numpy(), ctx, kwargs["max_speed"])

        # Axes B and C are only one-factor sweeps if the realised rate holds.
        # The tolerance is widened by the Monte-Carlo error of the estimate, so a
        # reduced-size run does not fail spuriously while a systematic error in
        # the compensation still trips it.
        if axis in ("jump", "speed", "centre"):
            n_trans = args.n_sequences * 2 * (ctx - 2)
            se = float(np.sqrt(TARGET_RATE * (1 - TARGET_RATE) / n_trans))
            allowed = args.rate_tol + 3 * se
            drift = abs(stats["rate"] - TARGET_RATE)
            assert drift <= allowed, (
                f"{regime}: realised rate {stats['rate']:.4f} drifted {drift:.4f} "
                f"from the target {TARGET_RATE:.4f} (allowed {allowed:.4f} = tol "
                f"{args.rate_tol} + 3 x SE {se:.4f}). The compensation is wrong "
                f"and this axis would confound {axis} with switching rate.")

        loader = DataLoader(TensorDataset(seqs, labels, motions),
                            batch_size=args.batch_size, shuffle=False, num_workers=0)
        for name in args.models:
            cfg, run_id, _ = runs[name]
            mean, std, details = eval_len_generalization(
                loaded[name], loader, device, ctx, n_strip_sequences=3)
            payload = dict(
                mean=mean, std=std, model=name, run_id=run_id,
                hidden_size=cfg["hidden_size"], gen_input_frames=ctx,
                gen_seq_len=args.gen_seq_len,
                train_input_frames=ref_cfg["input_frames"],
                train_pred_frames=trained_pred,
                n_sequences=args.n_sequences,
                regime=regime, axis=axis, level=np.float64(level),
                max_speed=kwargs["max_speed"],
                smooth_probability=kwargs["smooth_probability"],
                p_change_nominal=kwargs["p_change"],
                target_rate=TARGET_RATE,
                neighbor_kernel=kwargs.get("neighbor_kernel", "legacy"),
                motion_schedule=motions.numpy().astype(np.int16),
                **{f"motion_{k}": v for k, v in stats.items()})
            payload.update(details)
            np.savez_compressed(
                os.path.join(out_dir,
                             f"{args.out_name}_{name}_{slug(regime)}_{run_id}.npz"),
                **payload)
            med = float(np.median(details["per_seq_err"][:, :short_end].mean(axis=1)))
            rows.append((regime, axis, name, med, stats))
            print(f"  [{time.time()-t0:6.0f}s] {regime:<12} {name:<7} "
                  f"rate={stats['rate']:.4f} short={med:.3e}", flush=True)

    report(rows, args)


def report(rows, args):
    by = {(r, m): (v, s) for r, a, m, v, s in rows}
    models = list(dict.fromkeys(m for _, _, m, _, _ in rows))
    ref = {m: by[("centre", m)][0] for m in models if ("centre", m) in by}

    for axis, title, xname in (("rate", "AXIS A -- switching rate", "realised rate"),
                               ("jump", "AXIS B -- jump size", "mean |dv|"),
                               ("speed", "AXIS C -- speed range", "max_speed")):
        sel = [r for r in dict.fromkeys(r for r, a, *_ in rows if a in (axis, "centre"))]
        if not sel:
            continue
        print(f"\n{'=' * 104}\n{title}   (median MSE over t=16..25, normalised by the "
              f"shared centre)\n{'=' * 104}")
        hdr = (f"{'cell':<12}{'rate':>8}{'|dv|':>7}{'mean|v|':>9}{'net':>7}"
               f"{'age':>7}{'H':>7}  " + "".join(f"{m:>12}{'norm':>7}" for m in models))
        print(hdr); print("-" * len(hdr))
        for r in sel:
            s = by[(r, models[0])][1]
            line = (f"{r:<12}{s['rate']:>8.4f}{s['mean_dv']:>7.2f}"
                    f"{s['mean_speed']:>9.3f}{s['net_disp']:>7.1f}"
                    f"{s['mean_age']:>7.2f}{s['entropy']:>7.3f}  ")
            for m in models:
                v = by[(r, m)][0]
                line += f"{v:>12.3e}{v / ref[m]:>7.2f}" if m in ref else f"{v:>12.3e}{'-':>7}"
            print(line)
    print("\nnorm = this model's median at this cell / its own median at the shared "
          "centre (the training config).")


if __name__ == "__main__":
    main()
