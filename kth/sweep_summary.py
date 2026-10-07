#!/usr/bin/env python
"""
Combine sweep_kth.py results over seeds: one mean map per model (with the spread per cell) and a
summary table.

    python -m kth.sweep_summary cluster_runs/kth_sweep_final_s1 cluster_runs/kth_sweep_final_s23 \
        --out cluster_runs/kth_sweep_final

Runs are grouped by name without the trailing _s<seed>. Table columns:
  static    the A = 0 cell
  inside    mean over the training region's moving cells (A > 0 inside the region)
  outside   mean over every other moving cell of the grid
  worst     the lowest cell of the grid
  drifts    constant / piecewise / V_2
each as mean +- sd over seeds.
"""
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(dirs):
    groups = defaultdict(list)
    for d in dirs:
        for f in sorted(Path(d).glob("sweep_*.json")):
            r = json.load(open(f))
            groups[re.sub(r"_s\d+$", "", r["run"])].append(r)
    # LSTM, then FELSTM, then MELSTM (then anything else), as in the paper's figures
    rank = lambda name: next((i for i, m in enumerate(("_lstm_", "_felstm_", "_melstm_"))  # noqa
                              if m in f"_{name}_"), 3)
    return dict(sorted(groups.items(), key=lambda kv: (rank(kv[0]), kv[0])))


def label(name, cfg):
    """A readable panel title from the run's config."""
    m = cfg.get("model")
    if m == "lstm":
        return "ConvLSTM"
    if m == "felstm":
        return f"FEConvLSTM (V_{cfg.get('v_range', 1)}, {(2 * cfg.get('v_range', 1) + 1) ** 2} copies)"
    if m == "melstm":
        src = cfg.get("velocity_source", "frame_pair")
        if src == "tracked" and cfg.get("x_curriculum_epochs"):
            src = "handover"
        return f"MEConvLSTM ({src}, K = {cfg.get('num_vel_modes', 4)})"
    return name


def grid(r, amps, periods):
    return np.array([[r["cells"]["none" if A == 0 else f"shake_A{A:g}_P{P:g}"]["acc"]
                      for P in periods] for A in amps])


def inside_mask(r, amps, periods):
    reg = r["region"]
    if reg[0] != "shake":
        return np.zeros((len(amps), len(periods)), dtype=bool)
    (a_lo, a_hi), (p_lo, p_hi) = reg[1], reg[2]
    return np.array([[a_lo - 1e-9 <= A <= a_hi + 1e-9 and p_lo - 1e-9 <= P <= p_hi + 1e-9
                      for P in periods] for A in amps])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dirs", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    groups = load(a.dirs)
    first = next(iter(groups.values()))[0]
    amps = first["amps"]
    periods = sorted(first["periods"], reverse=True)          # frequency increases to the right
    moving = np.array([[A > 0] * len(periods) for A in amps])

    ms = lambda x: f"{np.mean(x):.3f} +- {np.std(x, ddof=1) if len(x) > 1 else 0:.3f}"  # noqa: E731
    table, maps = {}, {}
    print(f"{'model':24s} {'seeds':>5s}  {'static':>15s}  {'inside':>15s}  {'outside':>15s}  "
          f"{'worst':>15s}  drifts const / piecewise / V2")
    for name, rs in groups.items():
        g = np.stack([grid(r, amps, periods) for r in rs])        # (seeds, A, P)
        ins = inside_mask(rs[0], amps, periods) & moving
        outside = moving & ~ins
        row = {
            "seeds": [r["run"] for r in rs],
            "static": g[:, 0, 0].tolist(),
            "inside": (g[:, ins].mean(1).tolist() if ins.any() else []),
            "outside": g[:, outside].mean(1).tolist(),
            "worst": g.reshape(len(rs), -1).min(1).tolist(),
            "drifts": {k: [r["cells"][k]["acc"] for r in rs]
                       for k in ("constant", "piecewise", "constant:2") if k in rs[0]["cells"]},
        }
        table[name] = row
        maps[name] = (g.mean(0), g.std(0, ddof=1) if len(rs) > 1 else np.zeros_like(g[0]), ins)
        drifts = " / ".join(f"{np.mean(v):.3f}" for v in row["drifts"].values())
        print(f"{name:24s} {len(rs):>5d}  {ms(row['static']):>15s}  "
              f"{ms(row['inside']) if row['inside'] else '-':>15s}  {ms(row['outside']):>15s}  "
              f"{ms(row['worst']):>15s}  {drifts}")
    with open(out / "sweep_summary.json", "w") as f:
        json.dump({"amps": amps, "periods": periods, "table": table,
                   "mean_maps": {k: v[0].tolist() for k, v in maps.items()},
                   "sd_maps": {k: v[1].tolist() for k, v in maps.items()}}, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = len(maps)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), squeeze=False)
    for ax, (name, (mean, sd, ins)) in zip(axes[0], maps.items()):
        im = ax.imshow(mean, origin="lower", cmap="viridis", vmin=0.15, vmax=0.8, aspect="auto")
        for i in range(len(amps)):
            for j in range(len(periods)):
                c = "white" if mean[i, j] < 0.5 else "black"
                ax.text(j, i + 0.12, f"{mean[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color=c)
                ax.text(j, i - 0.22, f"+-{sd[i, j]:.2f}", ha="center", va="center", fontsize=5,
                        color=c)
        for j, P in enumerate(periods):                       # FELSTM-V_1 reach, a staircase
            inside = [i for i, A in enumerate(amps) if 2 * A * math.sin(math.pi / P) <= 1 + 1e-9]
            top = max(inside) + 0.5 if inside else -0.5
            ax.plot([j - 0.5, j + 0.5], [top, top], color="white", lw=2.2)
            ax.plot([j - 0.5, j + 0.5], [top, top], color="#d62728", lw=1.2)
        rows_in = np.flatnonzero(ins.any(1) | (np.arange(len(amps)) == 0) & ins.any())
        cols_in = np.flatnonzero(ins.any(0))
        if len(cols_in):
            lo = 0 if amps[0] == 0 else rows_in.min()
            ax.add_patch(Rectangle((cols_in.min() - 0.5, lo - 0.5), len(cols_in),
                                   rows_in.max() - lo + 1, fill=False, ec="#ff7f0e", lw=2.2))
        ax.set_xticks(range(len(periods)))
        ax.set_xticklabels([f"{P:g}\n{12.5 / P:.1f}Hz" for P in periods], fontsize=7)
        ax.set_yticks(range(len(amps)))
        ax.set_yticklabels([f"{A:g}" for A in amps], fontsize=7)
        ax.set_xlabel("shake period P (frames)", fontsize=8)
        ax.set_ylabel("amplitude A (px)", fontsize=8)
        cfg = groups[name][0]["config"]
        ax.set_title(f"{label(name, cfg)}\nmean +- sd over {len(table[name]['seeds'])} seeds",
                     fontsize=9)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="test accuracy")
    fig.suptitle("KTH under camera shake: test accuracy over amplitude x period "
                 "(orange: training region; red: FELSTM-V1 reach)", fontsize=9, y=1.06)
    fig.savefig(out / "sweep_mean_maps.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / "sweep_mean_maps.pdf", bbox_inches="tight")
    print(f"wrote {out}/sweep_summary.json and {out}/sweep_mean_maps.png/.pdf")


if __name__ == "__main__":
    main()
