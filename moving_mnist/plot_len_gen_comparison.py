"""
Combine length-generalization results from several runs into one figure:
a qualitative frames strip (ground truth + one row per model, same fixed
benchmark sequence) next to the MSE-vs-horizon curve with uncertainty bands.

Inputs are the .npz files written by train.py (save_len_gen_results), one per
run/model. Because the benchmark set is fixed, the curves are paired: the
script also prints a per-horizon paired summary table.

Example
-------
python plot_len_gen_comparison.py \
    fernn/movmnist/len_gen_lstm_abc.npz \
    fernn/movmnist/len_gen_felstm_def.npz \
    fernn/movmnist/len_gen_melstm_ghi.npz \
    --labels "ConvLSTM" "FEConvLSTM (V=2)" "MEConvLSTM (K=2)" \
    --out len_gen_comparison.png
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Fixed-order categorical palette (validated reference palette, light surface).
SERIES_COLORS = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#d9d8d4"


def load_runs(paths, labels):
    runs = []
    for i, p in enumerate(paths):
        d = np.load(p, allow_pickle=False)
        label = labels[i] if labels and i < len(labels) else str(d["model"])
        runs.append({"path": p, "label": label, "data": d})

    # Same fixed benchmark for everyone, or the comparison isn't paired.
    ref = runs[0]["data"]["strip_gt"]
    for r in runs[1:]:
        if r["data"]["strip_gt"].shape != ref.shape or \
                not np.allclose(r["data"]["strip_gt"], ref):
            print(f"WARNING: {r['path']} was evaluated on different benchmark "
                  f"sequences than {runs[0]['path']} — curves are NOT paired. "
                  f"Re-run evaluation with the fixed gen set for all models.")
    return runs


def paired_summary(runs, horizons):
    print(f"\n{'horizon':>8s}  " + "  ".join(f"{r['label']:>20s}" for r in runs))
    for t in horizons:
        cells = []
        for r in runs:
            e = r["data"]["per_seq_err"][:, t - 1]
            cells.append(f"{e.mean():.4f} ± {e.std():.4f}")
        print(f"{t:>8d}  " + "  ".join(f"{c:>20s}" for c in cells))

    if len(runs) > 1:
        base = runs[0]
        print(f"\nPaired win rate vs {base['label']} (fraction of benchmark "
              f"sequences with lower error, over the full horizon):")
        base_err = base["data"]["per_seq_err"]
        for r in runs[1:]:
            wins = (r["data"]["per_seq_err"] < base_err).mean()
            print(f"  {r['label']:<24s} {100 * wins:.1f}%")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz_files", nargs="+", help=".npz files from save_len_gen_results, one per model")
    ap.add_argument("--labels", nargs="*", default=None, help="Legend label per file (default: stored model name)")
    ap.add_argument("--out", default="len_gen_comparison.png")
    ap.add_argument("--band", choices=["std", "iqr"], default="std",
                    help="Uncertainty band: mean±std (clipped at 0) or median+IQR")
    ap.add_argument("--yscale", choices=["linear", "log"], default="linear")
    ap.add_argument("--strip", action=argparse.BooleanOptionalAction, default=True,
                    help="Include the qualitative frames strip (--no-strip to disable)")
    ap.add_argument("--curve", action=argparse.BooleanOptionalAction, default=True,
                    help="Include the error-vs-horizon panel (--no-curve for a strip-only figure)")
    ap.add_argument("--strip_seq", type=int, default=0, help="Which stored benchmark sequence to show")
    ap.add_argument("--strip_cols", type=int, default=8, help="Number of evenly spaced frame columns in the strip")
    ap.add_argument("--strip_step", type=int, default=None,
                    help="Show every Nth predicted frame instead (overrides --strip_cols), e.g. 4 for t=11,15,19,...")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    runs = load_runs(args.npz_files, args.labels)
    d0 = runs[0]["data"]
    T_pred = runs[0]["data"]["per_seq_err"].shape[1]
    train_horizon = int(d0["train_pred_frames"])
    gen_input_frames = int(d0["gen_input_frames"])

    if not (args.strip or args.curve):
        raise SystemExit("Nothing to plot: --no-strip and --no-curve both set.")

    n_models = len(runs)
    if args.strip_step:
        cols = np.arange(0, T_pred, args.strip_step)
    else:
        cols = np.unique(np.linspace(0, T_pred - 1, args.strip_cols).round().astype(int))
    n_cols = len(cols)

    if args.strip and args.curve:
        strip_rows = 1 + n_models
        fig_h = max(4.0, 0.85 * strip_rows + 1.2)
        fig = plt.figure(figsize=(3.2 + 0.75 * n_cols + 4.4, fig_h))
        gs = GridSpec(1, 2, width_ratios=[0.75 * n_cols, 4.4],
                      wspace=0.14, figure=fig)
        strip_gs = gs[0].subgridspec(strip_rows, n_cols,
                                     wspace=0.06, hspace=0.06)
        ax_curve = fig.add_subplot(gs[1])
    elif args.strip:
        strip_rows = 1 + n_models
        fig = plt.figure(figsize=(2.4 + 0.75 * n_cols, 0.85 * strip_rows + 0.8))
        strip_gs = GridSpec(strip_rows, n_cols, wspace=0.06, hspace=0.06,
                            figure=fig)
        ax_curve = None
    else:
        fig, ax_curve = plt.subplots(figsize=(6.4, 4.2))

    # ---- frames strip ----------------------------------------------------
    if args.strip:
        s = args.strip_seq
        gt = d0["strip_gt"][s]                     # (T_pred, C, H, W)

        rows = [("Ground truth", gt)]
        rows += [(r["label"], r["data"]["strip_pred"][s]) for r in runs]

        for ri, (row_label, frames) in enumerate(rows):
            for ci, t_idx in enumerate(cols):
                ax = fig.add_subplot(strip_gs[ri, ci])
                ax.imshow(np.clip(frames[t_idx].mean(axis=0), 0, 1),
                          cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color(GRID_COLOR); spine.set_linewidth(0.5)
                if ri == 0:
                    t_abs = gen_input_frames + int(t_idx) + 1
                    in_train = (t_idx + 1) <= train_horizon
                    ax.set_title(f"t={t_abs}", fontsize=8,
                                 color=TEXT_SECONDARY if in_train else TEXT_PRIMARY)
                if ci == 0:
                    ax.text(-0.12, 0.5, row_label, transform=ax.transAxes,
                            ha="right", va="center", fontsize=9, color=TEXT_PRIMARY)

    # ---- error curve -----------------------------------------------------
    if args.curve:
        x = np.arange(1, T_pred + 1)

        # Model-free reference: copy the last context frame forever. Stored in
        # every npz; identical across runs since the benchmark is fixed.
        if "per_seq_err_copy_last" in d0.files:
            base_err = d0["per_seq_err_copy_last"]
            base_center = (np.median(base_err, axis=0) if args.band == "iqr"
                           else base_err.mean(axis=0))
            ax_curve.plot(x, base_center, color=TEXT_SECONDARY, linewidth=1.2,
                          linestyle=":", label="copy last frame", zorder=2)

        end_labels = []
        for i, r in enumerate(runs):
            err = r["data"]["per_seq_err"]             # (N, T_pred)
            color = SERIES_COLORS[i % len(SERIES_COLORS)]
            if args.band == "iqr":
                center = np.median(err, axis=0)
                lo, hi = np.percentile(err, [25, 75], axis=0)
            else:
                center = err.mean(axis=0)
                sd = err.std(axis=0)
                lo, hi = np.clip(center - sd, 0, None), center + sd
            ax_curve.plot(x, center, color=color, linewidth=2, label=r["label"],
                          zorder=3)
            ax_curve.fill_between(x, lo, hi, color=color, alpha=0.18,
                                  linewidth=0, zorder=2)
            end_labels.append((center[-1], r["label"]))

        # direct labels at the line ends, in neutral ink, pushed apart when
        # curves finish too close together to keep the labels readable
        if args.yscale == "log":
            fwd, inv = np.log10, lambda v: 10.0 ** v
        else:
            fwd, inv = (lambda v: v), (lambda v: v)
        lo_lim, hi_lim = ax_curve.get_ylim()
        min_gap = 0.05 * (fwd(hi_lim) - fwd(lo_lim))
        placed = []
        for y, label in sorted(end_labels, key=lambda e: fwd(e[0])):
            fy = fwd(y)
            if placed and fy - placed[-1] < min_gap:
                fy = placed[-1] + min_gap
            placed.append(fy)
            ax_curve.annotate(label, (x[-1], inv(fy)), xytext=(4, 0),
                              textcoords="offset points", fontsize=8,
                              color=TEXT_SECONDARY, va="center")

        ax_curve.axvline(train_horizon, color=TEXT_SECONDARY, linestyle="--",
                         linewidth=1, zorder=1)
        ax_curve.text(train_horizon, ax_curve.get_ylim()[1], " training\n horizon",
                      fontsize=8, color=TEXT_SECONDARY, va="top", ha="left")

        ax_curve.set_yscale(args.yscale)
        ax_curve.set_xlabel("Prediction step beyond context", color=TEXT_PRIMARY)
        stat_name = "median MSE (IQR)" if args.band == "iqr" else "mean MSE (±std)"
        ax_curve.set_ylabel(f"{stat_name} vs ground truth", color=TEXT_PRIMARY)
        ax_curve.set_xlim(1, T_pred + max(2, T_pred // 8))   # room for end labels
        ax_curve.grid(axis="y", color=GRID_COLOR, linewidth=0.7, zorder=0)
        ax_curve.spines[["top", "right"]].set_color("none")
        ax_curve.spines[["left", "bottom"]].set_color(GRID_COLOR)
        ax_curve.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        ax_curve.legend(loc="upper left", frameon=False, fontsize=9,
                        labelcolor=TEXT_PRIMARY)
        ax_curve.set_title("Error vs. prediction horizon", fontsize=11,
                           color=TEXT_PRIMARY)

    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight",
                facecolor="white")
    print(f"Saved figure to {args.out}")

    # accessibility/table view: per-horizon numbers + paired win rates
    horizons = sorted(set(
        [1, train_horizon, min(2 * train_horizon, T_pred), T_pred]
    ))
    paired_summary(runs, [h for h in horizons if 1 <= h <= T_pred])


if __name__ == "__main__":
    main()
