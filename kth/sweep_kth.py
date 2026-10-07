#!/usr/bin/env python
"""
Score trained KTH checkpoints on a grid of camera shakes -- amplitude A x period P -- and on the
drift motions, without retraining: the generalisation maps.

    python -m kth.sweep_kth --runs kth_lstm_v0_s1 kth_felstm_v0_s1 kth_melstm_v0hov4_s1 \
        --out experiments_kth/sweep

A shake cell moves the camera, on both axes, by d(t) = A sin(2 pi t / P + phi): whole pixels,
independent random phases per axis and clip, no speed cap. P is in model frames (12.5 frames/s,
so 12.5 / P Hz); A = 0 is the static camera. The fastest step is 2 A sin(pi / P) px per frame,
and FEConvLSTM-V_1 covers the cells where that is <= 1 px (the staircase in the figure).

Every cell uses the same test clips and windows, so the cells differ only in the camera.

BatchNorm: each checkpoint is scored with the statistics it was saved with. The trainer recomputes
them on the training distribution before every epoch's val and test pass, and saves the best-val
model right after. So the A = 0 cell of a run trained without camera motion reproduces that run's
logged test accuracy at its best epoch exactly -- the check this script prints.
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

from kth.camera_motion import CameraMotion  # noqa: E402
from kth.kth_dataset import KTHClips, KTHStore  # noqa: E402
from kth.kth_model import build_kth_classifier  # noqa: E402
from kth.train_kth import class_report, run_epoch  # noqa: E402


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="run names (models/<run>_best.pth)")
    p.add_argument("--save_dir", default="./experiments_kth")
    p.add_argument("--root", default=str(Path(__file__).resolve().parent.parent / "data" / "kth"))
    p.add_argument("--amps", type=float, nargs="+", default=[0, 1, 1.5, 2, 3, 4, 6, 8],
                   help="amplitudes, px. Not 0.5: with whole-pixel rounding it never moves the "
                        "camera (measured: that row equals the static one for every model)")
    p.add_argument("--periods", type=float, nargs="+", default=[4, 6, 8, 12, 16, 24, 32])
    p.add_argument("--extra", default="constant,piecewise,constant:2",
                   help="drift motions scored as well ('mode:R' sets the velocity range)")
    p.add_argument("--data_seed", type=int, default=42)
    p.add_argument("--max_clips", type=int, default=None,
                   help="score only the first N test clips (quick checks; the check line is "
                        "then meaningless)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="./experiments_kth/sweep")
    return p.parse_args(argv)


def test_sets(store, a):
    """{name: (KTHClips, meta)} for every grid cell and drift motion; built once, shared."""
    sets = {}

    def add(name, cam, meta):
        ds = KTHClips(store, "test", camera=cam, seed=a.data_seed)
        if a.max_clips:
            ds = torch.utils.data.Subset(ds, list(range(min(a.max_clips, len(ds)))))
        sets[name] = (ds, meta)

    add("none", CameraMotion("none"), {"A": 0.0})
    for A in a.amps:
        if A == 0:
            continue
        for P in a.periods:
            add(f"shake_A{A:g}_P{P:g}", CameraMotion("shake", shake_amp=(A, A),
                                                     shake_period=(P, P), shake_vmax=0),
                {"A": A, "P": P})
    for tok in [t for t in a.extra.split(",") if t]:
        mode, _, r = tok.partition(":")
        R = int(r) if r else 1
        add(tok, CameraMotion(mode, v_range=R), {"mode": mode, "R": R})
    return sets


@torch.no_grad()
def score_run(run, sets, a, device):
    ck = torch.load(Path(a.save_dir) / "models" / f"{run}_best.pth", map_location=device,
                    weights_only=False)
    cfg = ck["config"]
    model = build_kth_classifier(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    model.x_track_p = 0.0
    criterion = nn.CrossEntropyLoss()
    out = {}
    for name, (ds, meta) in sets.items():
        loader = DataLoader(ds, batch_size=a.batch_size, num_workers=a.num_workers)
        r = run_epoch(model, loader, device, criterion, collect=True)
        rep = class_report(r.pop("_y_true"), r.pop("_y_pred"))
        out[name] = {"acc": r["acc"], "loss": r["loss"], **meta,
                     "acc_locomotion": rep["acc_locomotion"], "acc_inplace": rep["acc_inplace"]}
    return cfg, int(ck.get("epoch", -1)), out


def logged_test_at_best(save_dir, run, epoch):
    p = Path(save_dir) / "results" / f"history_{run}.json"
    if not p.exists():
        return None
    rows = json.load(open(p))["epochs"]
    row = next((r for r in rows if r["epoch"] == epoch), None)
    return None if row is None else row.get("test_acc")


def training_region(cfg):
    """What the run was trained on, for the outline: ('none',) or ('shake', A range, P range)."""
    cam = cfg.get("camera", "none")
    if cam == "shake":
        return ("shake", tuple(cfg.get("shake_amp", (1, 3))), tuple(cfg.get("shake_period", (6, 16))),
                cfg.get("camera_mix", 1.0), cfg.get("shake_vmax", 1))
    return (cam,)


def plot(results, a, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    amps = list(a.amps)
    periods = sorted(a.periods, reverse=True)                 # frequency increases to the right
    n = len(results)
    cols = min(n, 4)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 3.9 * rows + 0.6), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, (run, res) in zip(axes.flat, results.items()):
        grid = np.zeros((len(amps), len(periods)))
        for i, A in enumerate(amps):
            for j, P in enumerate(periods):
                key = "none" if A == 0 else f"shake_A{A:g}_P{P:g}"
                grid[i, j] = res["cells"][key]["acc"]
        ax.imshow(grid, origin="lower", cmap="viridis", vmin=0.15, vmax=0.8, aspect="auto")
        for i in range(len(amps)):
            for j in range(len(periods)):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if grid[i, j] < 0.5 else "black")
        # FELSTM-V_1's reach: cells whose fastest step 2 A sin(pi / P) <= 1 px (a staircase)
        for j, P in enumerate(periods):
            inside = [i for i, A in enumerate(amps) if 2 * A * math.sin(math.pi / P) <= 1 + 1e-9]
            top = max(inside) + 0.5 if inside else -0.5
            ax.plot([j - 0.5, j + 0.5], [top, top], color="white", lw=2.2)
            ax.plot([j - 0.5, j + 0.5], [top, top], color="#d62728", lw=1.2)
        reg = res["region"]
        if reg[0] == "shake":
            (a_lo, a_hi), (p_lo, p_hi) = reg[1], reg[2]
            ai = [i for i, A in enumerate(amps) if a_lo - 1e-9 <= A <= a_hi + 1e-9]
            pj = [j for j, P in enumerate(periods) if p_lo - 1e-9 <= P <= p_hi + 1e-9]
            if ai and pj:
                ax.add_patch(Rectangle((min(pj) - 0.5, min(ai) - 0.5), len(pj), len(ai),
                                       fill=False, ec="#ff7f0e", lw=2.2,
                                       ls="--" if reg[3] < 1 else "-"))
        elif reg[0] == "none":
            ax.add_patch(Rectangle((-0.5, -0.5), len(periods), 1, fill=False, ec="#ff7f0e",
                                   lw=2.2))
        ax.set_xticks(range(len(periods)))
        ax.set_xticklabels([f"{P:g}\n{12.5 / P:.1f}Hz" for P in periods], fontsize=7)
        ax.set_yticks(range(len(amps)))
        ax.set_yticklabels([f"{A:g}" for A in amps], fontsize=7)
        ax.set_xlabel("shake period P (frames)", fontsize=8)
        ax.set_ylabel("amplitude A (px)", fontsize=8)
        drift = "  ".join(f"{k.replace('constant:2', 'V2')} {v['acc']:.2f}"
                          for k, v in res["cells"].items() if "mode" in v)
        ax.set_title(f"{run}\n{drift}", fontsize=8)
    fig.suptitle("test accuracy per camera shake (orange: training region; red: FELSTM-V1 "
                 "reach, fastest step <= 1 px)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None):
    a = get_args(argv)
    device = torch.device(a.device) if a.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    store = KTHStore(a.root)
    sets = test_sets(store, a)
    print(f"{len(sets)} test sets x {len(next(iter(sets.values()))[0])} clips, device {device}")
    results = {}
    for run in a.runs:
        cfg, epoch, cells = score_run(run, sets, a, device)
        logged = logged_test_at_best(a.save_dir, run, epoch)
        check = ""
        if cfg.get("camera", "none") == "none" and logged is not None:
            check = (f"   check: A=0 cell {cells['none']['acc']:.4f} vs logged test at best "
                     f"epoch {logged:.4f}")
        print(f"{run:32s} best epoch {epoch:3d}  none {cells['none']['acc']:.3f}{check}")
        results[run] = {"epoch": epoch, "region": training_region(cfg), "cells": cells}
        with open(out / f"sweep_{run}.json", "w") as f:
            json.dump({"run": run, "config": cfg, **results[run],
                       "amps": a.amps, "periods": a.periods}, f, indent=2)
    plot(results, a, out / "sweep_maps.png")
    print(f"wrote {out}/sweep_*.json and {out}/sweep_maps.png")


if __name__ == "__main__":
    main()
