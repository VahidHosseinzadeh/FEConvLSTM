"""
Paper-style validation-loss-vs-training-steps comparison from the
history_*.json files written by train.py.

One curve per model with an uncertainty band:
  - one file per label  -> band = per-sequence std of the fixed val subset
    (recorded by ValCurveRecorder during training);
  - several files with the SAME label (seeds) -> band = std across runs,
    interpolated onto a common step grid.

--with-train overlays the (smoothed) per-batch training loss as a dashed
line in the same hue on the same axis (same measure, same scale).

Example
-------
python plot_loss_curves.py \
    fernn/movmnist/history_lstm_abc.json \
    fernn/movmnist/history_felstm_def.json \
    fernn/movmnist/history_melstm_ghi.json \
    --labels "ConvLSTM" "FEConvLSTM (V=2)" "MEConvLSTM (K=2)" \
    --out loss_curves.png
"""
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fixed-order categorical palette (validated reference palette, light surface).
SERIES_COLORS = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#d9d8d4"


def smooth(x, k):
    x = np.asarray(x, dtype=float)
    if k <= 1 or len(x) < k:
        return x
    c = np.convolve(x, np.ones(k) / k, mode="valid")
    return np.concatenate([x[:k - 1], c])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("history_files", nargs="+", help="history_*.json files from train.py")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="Label per file; repeat a label to group seeds into one banded curve")
    ap.add_argument("--out", default="loss_curves.png")
    ap.add_argument("--with-train", dest="with_train", action="store_true",
                    help="Overlay smoothed per-batch training loss (dashed)")
    ap.add_argument("--train_smooth", type=int, default=51,
                    help="Moving-average window (batches) for the training loss overlay")
    ap.add_argument("--yscale", choices=["linear", "log"], default="linear")
    ap.add_argument("--xmax", type=int, default=None, help="Clip the x-axis (batches)")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    # group files by label (same label = seeds of one model)
    groups = {}
    for i, path in enumerate(args.history_files):
        with open(path) as f:
            d = json.load(f)
        label = (args.labels[i] if args.labels and i < len(args.labels)
                 else str(d["config"].get("model", path)))
        if "val_curve" not in d or not d["val_curve"]["step"]:
            print(f"WARNING: {path} has no val_curve data "
                  f"(run trained with --val_curve_interval 0?) — skipped.")
            continue
        groups.setdefault(label, []).append(d)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))

    for i, (label, runs) in enumerate(groups.items()):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]

        if len(runs) == 1:
            d = runs[0]
            x = np.asarray(d["val_curve"]["step"])
            mean = np.asarray(d["val_curve"]["loss_mean"])
            band = np.asarray(d["val_curve"]["loss_std"])
        else:
            # seeds: interpolate every run onto the first run's grid
            x = np.asarray(runs[0]["val_curve"]["step"])
            stack = np.stack([
                np.interp(x, r["val_curve"]["step"], r["val_curve"]["loss_mean"])
                for r in runs
            ])
            mean, band = stack.mean(axis=0), stack.std(axis=0)

        if args.xmax:
            keep = x <= args.xmax
            x, mean, band = x[keep], mean[keep], band[keep]

        n_seeds = f" ({len(runs)} seeds)" if len(runs) > 1 else ""
        ax.plot(x, mean, color=color, linewidth=2, label=label + n_seeds, zorder=3)
        ax.fill_between(x, np.clip(mean - band, 0, None), mean + band,
                        color=color, alpha=0.18, linewidth=0, zorder=2)

        if args.with_train:
            d = runs[0]   # training loss of the first run is representative
            tx = np.asarray(d["train_curve"]["step"])
            ty = smooth(d["train_curve"]["loss"], args.train_smooth)
            if args.xmax:
                keep = tx <= args.xmax
                tx, ty = tx[keep], ty[keep]
            ax.plot(tx, ty, color=color, linewidth=1.2, linestyle="--",
                    alpha=0.8, zorder=2)

    if args.with_train:
        ax.plot([], [], color=TEXT_SECONDARY, linewidth=1.2, linestyle="--",
                label="training loss (smoothed)")

    ax.set_yscale(args.yscale)
    ax.set_xlabel("Training steps (batches)", color=TEXT_PRIMARY)
    ax.set_ylabel("Validation loss (MSE + L1)", color=TEXT_PRIMARY)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_color("none")
    ax.spines[["left", "bottom"]].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.legend(loc="upper right", frameon=False, fontsize=9,
              labelcolor=TEXT_PRIMARY)

    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved figure to {args.out}")

    # table view: val loss at a few step milestones
    milestones = None
    for label, runs in groups.items():
        x = np.asarray(runs[0]["val_curve"]["step"])
        if milestones is None:
            qs = [0.1, 0.25, 0.5, 1.0]
            milestones = [int(x[-1] * q) for q in qs]
            print(f"\n{'steps':>10s}  " + "  ".join(f"{m:>10d}" for m in milestones))
        vals = np.interp(milestones, x, np.asarray(runs[0]["val_curve"]["loss_mean"]))
        print(f"{label:>10s}  " + "  ".join(f"{v:>10.4f}" for v in vals))
    print()


if __name__ == "__main__":
    main()
