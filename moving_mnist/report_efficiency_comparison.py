"""
Efficiency comparison table + figure from history_*.json files: parameters
(should come out ~equal across models -- see script docstring in the repo
discussion), the velocity/slot multiplier each architecture pays on top of
that, the resulting recurrent-state memory footprint, and measured wall-clock
per epoch. Answers "same parameters, why is the wall-clock so different?"
with actual numbers instead of a hand-wave.

numpy + matplotlib only (no torch) -- reads only what train.py already saved,
so this runs anywhere, including headless on the cluster.

Example
-------
python report_efficiency_comparison.py \
    fernn/movmnist/history_lstm_abc.json \
    fernn/movmnist/history_felstm_def.json \
    fernn/movmnist/history_melstm_ghi.json \
    --labels "ConvLSTM" "FEConvLSTM (V=2)" "MEConvLSTM (K=2)" \
    --out efficiency.png --markdown efficiency.md
"""
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fixed-order categorical palette (validated reference palette, light surface),
# same as the other comparison scripts.
SERIES_COLORS = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#d9d8d4"


def state_multiplier(cfg):
    """
    Number of parallel velocity/slot copies the recurrent cell processes
    per step -- pure compute/memory multiplier, orthogonal to parameter
    count (the conv weights are shared/tied across all copies).

    lstm/felstm (Seq2SeqFEConvLSTM): dense grid over the v_range x v_range
        velocity lattice -> (2*v_range+1)^2. v_range=0 -> 1, i.e. lstm is
        just felstm's v_range=0 special case; same formula covers both.
    melstm (Seq2SeqMEConvLSTM): K tracked slots -> num_vel_modes directly.
    """
    if cfg["model"] == "melstm":
        return cfg["num_vel_modes"]
    return (2 * cfg["v_range"] + 1) ** 2


def state_memory_mb(cfg, mult):
    """
    Recurrent state (h and c together) footprint in MB at the config's own
    batch_size/image_size, float32. This is the number that actually
    diverges across models at equal parameter count -- the mechanism
    behind both the epoch-time and GPU-memory differences.
    """
    B = cfg["batch_size"]
    H = W = cfg["image_size"]
    hidden = cfg["hidden_size"]
    bytes_total = 2 * B * mult * hidden * H * W * 4   # h + c, float32
    return bytes_total / (1024 ** 2)


def load_runs(paths, labels):
    runs = []
    for i, p in enumerate(paths):
        with open(p) as f:
            d = json.load(f)
        label = labels[i] if labels and i < len(labels) else str(d["config"]["model"])
        runs.append({"label": label, "data": d})
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("history_files", nargs="+", help="history_*.json files from train.py")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default="efficiency.png")
    ap.add_argument("--markdown", default=None, help="Also write a markdown table to this path")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    runs = load_runs(args.history_files, args.labels)

    rows = []
    for r in runs:
        cfg = r["data"]["config"]
        mult = state_multiplier(cfg)
        mem = state_memory_mb(cfg, mult)
        params = r["data"].get("num_parameters")
        epoch_times = r["data"]["history"].get("epoch_time_sec", [])
        n_done = len(epoch_times)
        mean_t = float(np.mean(epoch_times)) if epoch_times else float("nan")
        median_t = float(np.median(epoch_times)) if epoch_times else float("nan")
        total_epochs = cfg.get("epochs")
        projected_hours = (mean_t * total_epochs / 3600) if epoch_times and total_epochs else float("nan")
        rows.append({
            "label": r["label"], "params": params, "mult": mult, "mem_mb": mem,
            "mean_epoch_s": mean_t, "median_epoch_s": median_t,
            "epochs_done": n_done, "epochs_total": total_epochs,
            "projected_hours": projected_hours,
        })

    # ---- console + markdown table -----------------------------------------
    header = (f"{'model':<20s} {'params':>10s} {'mult':>6s} {'state MB':>10s} "
              f"{'mean s/ep':>10s} {'done/total':>12s} {'proj. hours':>12s}")
    print(header)
    print("-" * len(header))
    md_lines = ["| model | params | velocity/slot multiplier | state size (MB) | "
                "mean s/epoch | epochs done/total | projected total (h) |",
                "|---|---|---|---|---|---|---|"]
    for row in rows:
        print(f"{row['label']:<20s} {row['params']:>10,} {row['mult']:>6d} {row['mem_mb']:>10.1f} "
              f"{row['mean_epoch_s']:>10.1f} {str(row['epochs_done'])+'/'+str(row['epochs_total']):>12s} "
              f"{row['projected_hours']:>12.1f}")
        md_lines.append(
            f"| {row['label']} | {row['params']:,} | {row['mult']}x | {row['mem_mb']:.1f} | "
            f"{row['mean_epoch_s']:.1f} | {row['epochs_done']}/{row['epochs_total']} | "
            f"{row['projected_hours']:.1f} |"
        )
    print()
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write("\n".join(md_lines) + "\n")
        print(f"Saved markdown table to {args.markdown}")

    # ---- figure: 3 panels, shared model order/colors ----------------------
    fig, axes = plt.subplots(1, 3, figsize=(3.2 * len(rows) + 2, 4.2))
    labels = [row["label"] for row in rows]
    colors = [SERIES_COLORS[i % len(SERIES_COLORS)] for i in range(len(rows))]
    x = np.arange(len(rows))

    panels = [
        (axes[0], [row["params"] for row in rows], "Parameters",
         "(should be ~equal — weight-tied across the velocity/slot dimension)"),
        (axes[1], [row["mult"] for row in rows], "Velocity/slot multiplier (×)",
         "compute + state-memory cost per step, NOT a parameter cost"),
        (axes[2], [row["mean_epoch_s"] for row in rows], "Mean seconds / epoch",
         "measured wall-clock so far"),
    ]
    for ax, values, title, subtitle in panels:
        bars = ax.bar(x, values, color=colors, width=0.6, zorder=3)
        for xi, v in zip(x, values):
            ax.text(xi, v, f"{v:,.0f}" if v >= 100 else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8, color=TEXT_PRIMARY)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
        ax.set_title(title, fontsize=10, color=TEXT_PRIMARY)
        ax.text(0.5, -0.32, subtitle, transform=ax.transAxes, ha="center",
                fontsize=7, color=TEXT_SECONDARY, style="italic")
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, zorder=0)
        ax.spines[["top", "right"]].set_color("none")
        ax.spines[["left", "bottom"]].set_color(GRID_COLOR)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        if max(values) / max(min(v for v in values if v > 0), 1e-9) > 20:
            ax.set_yscale("log")

    fig.suptitle("Equal parameters, unequal cost: state multiplier drives "
                 "compute/memory, not parameter count", fontsize=11, color=TEXT_PRIMARY)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
