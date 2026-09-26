#!/usr/bin/env python
"""
Compare runs: per-lead error curves of several train_motion.py results (one line per run) with
the persistence baselines of the first run as dashed references.

    python -m motion_benchmarks.scripts.plot_results \
        experiments/motion_benchmarks/radar_synthetic_*/results.json --metric rmse --out radar.png
    # a categorical score instead (key as stored in results.json):
    python -m motion_benchmarks.scripts.plot_results runs/*/results.json --metric "csi@0.274653"
    # any extra test set (e.g. one lifetime of the radar sweep):
    python -m motion_benchmarks.scripts.plot_results runs/*/results.json --split extra_lifetime_48_frozen
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np


def label_of(r):
    a = r["args"]
    lab = a["model"]
    if a["model"] == "melstm":
        lab += f" [{a.get('velocity_source')}, K={a.get('num_vel_modes')}]"
    if a.get("residual", "none") != "none":
        lab += f" +{a['residual']} skip"
    return lab


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", nargs="+")
    ap.add_argument("--metric", default="mse")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="comparison.png")
    ap.add_argument("--no_baselines", action="store_true")
    ap.add_argument("--logy", action="store_true")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted({p for pat in a.results for p in glob.glob(pat)})
    if not files:
        raise SystemExit("no results files")
    runs = [json.load(open(p)) for p in files]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for r, p in zip(runs, files):
        res = r.get(a.split) or r.get(a.split + "_frozen")
        if res is None or a.metric not in res:
            print(f"skip {p}: no {a.split}/{a.metric}")
            continue
        y = np.array(res[a.metric], dtype=float)
        ax.plot(np.arange(1, len(y) + 1), y, marker="o", ms=3, label=label_of(r))
    if not a.no_baselines and "baselines" in runs[0]:
        split = a.split.replace("_frozen", "").replace("_tracked", "")
        for name, b in runs[0]["baselines"].items():
            res = b.get(split)
            if res is not None and a.metric in res:
                y = np.array(res[a.metric], dtype=float)
                ax.plot(np.arange(1, len(y) + 1), y, ls="--", lw=1.2,
                        label=f"persistence: {name}")
    ax.set_xlabel("lead (frames)")
    ax.set_ylabel(a.metric)
    if a.logy:
        ax.set_yscale("log")
    ax.set_title(f"{runs[0]['args']['dataset']} -- {a.split}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
