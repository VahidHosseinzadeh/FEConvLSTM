#!/usr/bin/env python
"""
Score trained KTH checkpoints with the camera moving under the Moving MNIST motion laws.
Evaluation only: the KTH counterpart of moving_mnist/motion_sweep_classification.py.

    python -m kth.motion_sweep_kth --runs kth_lstm_shk_s1 kth_felstm_shk_s1 \
        kth_melstm_shkhov4_s1 --out experiments_kth/motion_sweep

The grids and the named regimes are IMPORTED from the Moving MNIST sweeps, not restated, so a
cell here is the same motion law as the cell of the same name there:

  rate    stochastic, smooth transitions (s = 0.8), p_change = d ** 2.170 for d in D_GRID
  jump    stochastic, s in S_GRID, p_change compensated to hold the realised rate of d = 0.50
          (s = 0.8 is the rate axis's d = 0.50 cell, shared as in Moving MNIST)
  named   the NAMED_REGIMES of motion_difficulty_sweep.py
          All three at max speed 2 with the symmetric neighbour kernel and (0, 0) off the grid,
          exactly as in Moving MNIST.
  speed   constant drift, one velocity per clip from V_R (R = 1..5) with (0, 0), Keller's draws:
          R = 1 and R = 2 are the trainer's 'constant' and 'constant_v2' test sets.
  reference
          'none' (static camera) and 'train' (the checkpoint's own training shake).

Every cell uses the 472 test clips and windows of the trainer's test sets and of sweep_kth.py
(--data_seed 42), so the cells differ only in the camera. A clip's trajectory is drawn once per
cell from (data_seed, split), the same for every checkpoint. BatchNorm: the statistics saved with
each checkpoint, as in sweep_kth.py.

What places a cell on its axis -- switching rate, mean jump |dv|_inf over the switches,
H(v_t | v_{t-1}) -- is a property of the motion law, so it is measured on --stats_draws fresh
trajectories rather than on the 472 test ones (a plug-in entropy needs many transitions).

Output, in --out:
  motion_sweep_<run>.json  per cell: accuracy, loss, locomotion / in-place accuracy and the
                           per-clip predictions (bootstrap intervals, per-clip analyses)
  motion_cells.json        per cell: axis, level, motion law, realised statistics; test labels
  motion_cells.npz         per cell: the test trajectories, (clips, T, 2) int8
"""
import argparse
import json
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from kth.camera_motion import CameraMotion  # noqa: E402  (puts moving_mnist/ on sys.path)
from kth.kth_dataset import KTHClips, KTHStore  # noqa: E402
from kth.sweep_kth import load_run  # noqa: E402
from kth.train_kth import class_report, run_epoch  # noqa: E402
from motion_difficulty_sweep import (NAMED_REGIMES, RATE_TOL, REGIME_CHECKS,  # noqa: E402
                                     SPEED_TOL)
from motion_factors_sweep import (D_GRID, S_GRID, TARGET_RATE, TRAIN_N,  # noqa: E402
                                  TRAIN_S, compensated_p)

SPEED_GRID = (1, 2, 3, 4, 5)

# NAMED_REGIMES keys -> CameraMotion keywords. A regime leaves out only what its law never reads.
_REGIME_KEYS = {"motion_mode": "mode", "transition_mode": "transition",
                "min_segment": "min_segment", "max_segment": "max_segment",
                "smooth_probability": "smooth_probability", "p_change": "p_change"}


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="run names (models/<run>_best.pth)")
    p.add_argument("--save_dir", default="./experiments_kth")
    p.add_argument("--root", default=str(Path(__file__).resolve().parent.parent / "data" / "kth"))
    p.add_argument("--data_seed", type=int, default=42)
    p.add_argument("--stats_draws", type=int, default=20000,
                   help="trajectories per motion law for its realised statistics")
    p.add_argument("--max_clips", type=int, default=None,
                   help="score only the first N test clips (quick checks)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="./experiments_kth/motion_sweep")
    return p.parse_args(argv)


# ------------------------------------------------------------------- the cells
def train_law(cfg):
    """The checkpoint's own training camera, as CameraMotion keywords."""
    if cfg.get("camera", "none") != "shake" or cfg.get("camera_mix", 1.0) < 1.0:
        raise SystemExit(f"{cfg.get('run_name')}: the 'train' cell expects a run trained on "
                         f"the shake alone, not camera={cfg.get('camera')!r} "
                         f"(mix {cfg.get('camera_mix', 1.0)})")
    return dict(mode="shake", shake_amp=tuple(cfg["shake_amp"]),
                shake_period=tuple(cfg["shake_period"]), shake_axes=cfg["shake_axes"],
                shake_vmax=cfg["shake_vmax"])


def cells(cfg):
    """(name, axis, level, CameraMotion keywords) for every cell."""
    mm = dict(v_range=TRAIN_N, neighbor_kernel="symmetric", include_zero=False)
    out = [("none", "reference", 0.0, dict(mode="none")),
           ("train", "reference", float("nan"), train_law(cfg))]
    for d in D_GRID:
        out.append((f"A:d={d:.2f}", "rate", float(d),
                    dict(mm, mode="stochastic", transition="smooth", p_change=d ** 2.170,
                         smooth_probability=TRAIN_S)))
    for s in S_GRID:
        out.append((f"B:s={s:.2f}", "jump", float(s),
                    dict(mm, mode="stochastic", transition="smooth",
                         p_change=compensated_p(TARGET_RATE, s, TRAIN_N),
                         smooth_probability=s)))
    for name, kw in NAMED_REGIMES.items():
        unknown = set(kw) - set(_REGIME_KEYS)
        if unknown:
            raise SystemExit(f"regime {name}: no camera equivalent for {sorted(unknown)}")
        out.append((f"C:{name}", "named", float("nan"),
                    dict(mm, **{_REGIME_KEYS[k]: v for k, v in kw.items()})))
    for R in SPEED_GRID:
        out.append((f"E:R={R}", "speed", float(R), dict(mode="constant", v_range=R)))
    return out


# ------------------------------------------------------------------- the stats
def law_stats(v):
    """
    Realised statistics of (n, T, 2) camera velocities, over the T - 1 steps between a clip's
    T frames (the last row is the unused step past the clip):

      rate       fraction of transitions v_{t-1} -> v_t that change the velocity
      jump       mean |v_t - v_{t-1}|_inf over those changes
      entropy    H(v_t | v_{t-1}) in bits per frame, plug-in, pooled over clips and transitions
      speed      mean |v_t|_inf (px per frame)
      mean_speed mean |v_t|_2, and net_disp the mean |d(T-1)|_2: Moving MNIST's motion_stats
      travel     mean |d(T-1)|_inf, how far the last frame has moved (px)
      excursion  mean max_t |d(t)|_inf, the farthest the camera gets from the first frame (px)
    """
    u = np.asarray(v, np.int64)[:, :-1]
    dv = np.abs(u[:, 1:] - u[:, :-1]).max(-1)
    changed = dv > 0
    R = int(np.abs(u).max())
    K = (2 * R + 1) ** 2
    code = (u[..., 0] + R) * (2 * R + 1) + (u[..., 1] + R)
    joint = np.bincount((code[:, :-1] * K + code[:, 1:]).ravel(), minlength=K * K)
    P = joint / joint.sum()
    prev = P.reshape(K, K).sum(1)
    h = lambda q: float(-(q[q > 0] * np.log2(q[q > 0])).sum())  # noqa: E731
    d = np.cumsum(u, axis=1)                                   # frames 1 .. T-1
    return dict(rate=float(changed.mean()),
                jump=float(dv[changed].mean()) if changed.any() else 0.0,
                entropy=h(P) - h(prev),
                speed=float(np.abs(u).max(-1).mean()),
                mean_speed=float(np.sqrt((u ** 2).sum(-1)).mean()),
                net_disp=float(np.sqrt((d[:, -1] ** 2).sum(-1)).mean()),
                travel=float(np.abs(d[:, -1]).max(-1).mean()),
                excursion=float(np.abs(d).max(-1).max(-1).mean()))


def check_regimes(stats):
    """The named regimes must reproduce Moving MNIST's realised rate and mean speed |v|_2,
    with motion_difficulty_sweep.py's own tolerances."""
    ok = True
    for name, want in REGIME_CHECKS.items():
        got = stats[f"C:{name}"]
        bad = (abs(got["rate"] - want["rate"]) > RATE_TOL
               or abs(got["mean_speed"] - want["speed"]) > SPEED_TOL)
        ok &= not bad
        print(f"  check {name:10s} rate {got['rate']:.3f} (Moving MNIST {want['rate']:.3f})  "
              f"|v|_2 {got['mean_speed']:.3f} ({want['speed']:.3f})  {'FAIL' if bad else 'ok'}")
    return ok


# ------------------------------------------------------------------- scoring
@torch.no_grad()
def score_run(run, sets, a, device):
    model, cfg, epoch = load_run(run, a.save_dir, device)
    criterion = nn.CrossEntropyLoss()
    out, labels = {}, None
    for name, ds in sets.items():
        loader = DataLoader(ds, batch_size=a.batch_size, num_workers=a.num_workers)
        r = run_epoch(model, loader, device, criterion, collect=True)
        y_true, y_pred = r.pop("_y_true"), r.pop("_y_pred")
        r.pop("_global_step", None)
        rep = class_report(y_true, y_pred)
        out[name] = {**{k: v for k, v in r.items() if isinstance(v, (int, float))},
                     "acc_locomotion": rep["acc_locomotion"], "acc_inplace": rep["acc_inplace"],
                     "pred": [int(q) for q in y_pred]}
        labels = labels or [int(t) for t in y_true]
    return cfg, epoch, out, labels


def main(argv=None):
    a = get_args(argv)
    device = torch.device(a.device) if a.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    store = KTHStore(a.root)

    # Every run must share the training camera: the 'train' cell is built once.
    cfgs = {run: torch.load(Path(a.save_dir) / "models" / f"{run}_best.pth", map_location="cpu",
                            weights_only=False)["config"] for run in a.runs}
    laws = {json.dumps(train_law(c), sort_keys=True) for c in cfgs.values()}
    if len(laws) > 1:
        raise SystemExit(f"the runs were trained on different shakes: {sorted(laws)}")
    first = next(iter(cfgs.values()))
    T = int(first["seq_len"])

    todo = cells(first)
    sets, meta, trajectories = {}, {}, {}
    rng_stats = np.random.RandomState(12345)
    for name, axis, level, kw in todo:
        cam = CameraMotion(seq_len=T, **kw)
        ds = KTHClips(store, "test", scheme=first["split_scheme"], camera=cam, seq_len=T,
                      step=int(first["step"]), seed=a.data_seed)
        trajectories[name] = ds.trajectories.astype(np.int8)
        if a.max_clips:
            ds = torch.utils.data.Subset(ds, list(range(min(a.max_clips, len(ds)))))
        sets[name] = ds
        stats = law_stats(np.stack([cam.draw(rng_stats) for _ in range(a.stats_draws)]))
        meta[name] = {"axis": axis, "level": level, "law": cam.describe(),
                      "kwargs": {k: (list(v) if isinstance(v, tuple) else v)
                                 for k, v in kw.items()},
                      "stats": stats, "test_stats": law_stats(trajectories[name])}
    n_clips = len(next(iter(sets.values())))
    print(f"{len(sets)} cells x {n_clips} clips, device {device}; motion laws:")
    for name, m in meta.items():
        s = m["stats"]
        print(f"  {name:20s} rate {s['rate']:.3f}  jump {s['jump']:.2f}  H {s['entropy']:.2f} "
              f"bits  speed {s['speed']:.2f}  travel {s['travel']:5.1f} px   {m['law']}")
    if not check_regimes({k: m["stats"] for k, m in meta.items()}):
        raise SystemExit("the named regimes do not reproduce Moving MNIST's statistics")

    np.savez_compressed(out / "motion_cells.npz", **{k.replace(":", "_").replace("=", "_"): v
                                                     for k, v in trajectories.items()})
    labels_all = None
    for run in a.runs:
        cfg, epoch, res, labels = score_run(run, sets, a, device)
        labels_all = labels_all or labels
        line = "  ".join(f"{k} {res[k]['acc']:.3f}" for k in ("none", "train", "E:R=1", "E:R=2"))
        print(f"{run:28s} best epoch {epoch:3d}  {line}")
        with open(out / f"motion_sweep_{run}.json", "w") as f:
            json.dump({"run": run, "config": cfg, "epoch": epoch, "cells": res}, f)
    with open(out / "motion_cells.json", "w") as f:
        json.dump({"cells": meta, "labels": labels_all, "data_seed": a.data_seed,
                   "n_clips": n_clips, "seq_len": T,
                   "npz_keys": {k: k.replace(":", "_").replace("=", "_") for k in meta}},
                  f, indent=1)
    print(f"wrote {out}/motion_sweep_*.json, motion_cells.json, motion_cells.npz")


if __name__ == "__main__":
    main()
