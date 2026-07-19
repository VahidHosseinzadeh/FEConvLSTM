"""
Combine velocity-generalization error grids from several runs into one figure:
one heatmap per model on a shared color scale, with the training velocity
range marked, plus a stdout table of in-range vs out-of-range mean MSE.

Inputs are the .npz files written by train.py (save_vel_gen_results).

Example
-------
python plot_vel_gen_comparison.py \
    fernn/movmnist/vel_gen_lstm_abc.npz \
    fernn/movmnist/vel_gen_felstm_def.npz \
    fernn/movmnist/vel_gen_melstm_ghi.npz \
    --labels "ConvLSTM" "FEConvLSTM (V=2)" "MEConvLSTM (K=2)" \
    --out vel_gen_comparison.png
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#d9d8d4"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz_files", nargs="+", help=".npz files from save_vel_gen_results, one per model")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default="vel_gen_comparison.png")
    ap.add_argument("--vmax", type=float, default=None,
                    help="Shared color-scale max (default: max over all grids)")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    runs = []
    for i, p in enumerate(args.npz_files):
        d = np.load(p, allow_pickle=False)
        label = (args.labels[i] if args.labels and i < len(args.labels)
                 else str(d["model"]))
        runs.append({"label": label, "data": d})

    vmax = args.vmax or max(float(r["data"]["err"].max()) for r in runs)
    n = len(runs)

    fig, axes = plt.subplots(1, n, figsize=(3.6 * n + 1.2, 3.8),
                             squeeze=False, constrained_layout=True)
    axes = axes[0]

    for i, (r, ax) in enumerate(zip(runs, axes)):
        d = r["data"]
        vx, vy, err = d["vx"], d["vy"], d["err"]
        train_v = int(d["data_v_range"])

        # single-hue sequential colormap: magnitude, light -> dark
        im = ax.imshow(err, origin="lower", cmap="Blues", vmin=0, vmax=vmax,
                       extent=[vx[0] - 0.5, vx[-1] + 0.5,
                               vy[0] - 0.5, vy[-1] + 0.5])
        # training velocity range
        ax.add_patch(Rectangle((-train_v - 0.5, -train_v - 0.5),
                               2 * train_v + 1, 2 * train_v + 1,
                               fill=False, edgecolor=TEXT_PRIMARY,
                               linestyle="--", linewidth=1.2))
        ax.set_title(r["label"], fontsize=10, color=TEXT_PRIMARY)
        ax.set_xlabel("$v_x$ (pixels/frame)", color=TEXT_PRIMARY, fontsize=9)
        if i == 0:
            ax.set_ylabel("$v_y$ (pixels/frame)", color=TEXT_PRIMARY, fontsize=9)
        ax.set_xticks(vx); ax.set_yticks(vy)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("MSE", color=TEXT_PRIMARY, fontsize=9)
    cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    cbar.outline.set_edgecolor(GRID_COLOR)
    fig.suptitle("Velocity generalization (dashed box = training range)",
                 fontsize=11, color=TEXT_PRIMARY)

    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved figure to {args.out}")

    # table view: mean MSE inside vs outside the training velocity box
    print(f"\n{'model':<24s} {'in-range MSE':>14s} {'out-of-range MSE':>18s}")
    for r in runs:
        d = r["data"]
        vx, vy, err = d["vx"], d["vy"], d["err"]
        train_v = int(d["data_v_range"])
        gx, gy = np.meshgrid(vx, vy)
        inside = (np.abs(gx) <= train_v) & (np.abs(gy) <= train_v)
        out_str = (f"{err[~inside].mean():>18.4f}" if (~inside).any()
                   else f"{'n/a':>18s}")
        print(f"{r['label']:<24s} {err[inside].mean():>14.4f} {out_str}")
    print()


if __name__ == "__main__":
    main()
