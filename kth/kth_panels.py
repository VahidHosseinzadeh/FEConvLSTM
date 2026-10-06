"""
Pictures for the KTH runs: what each velocity copy holds over time, and what the head reads.

state_figure     one clip. One row per velocity copy (felstm: the 9 lattice velocities; melstm:
                 the K slots; lstm: its one state), each the CHANNEL MEAN of that copy at every
                 frame; then "max over V", the channel mean of the elementwise max over copies --
                 exactly the tensor the head reads (velocity_pool 'max'); then "argmax V", which
                 copy supplies that max at each pixel (in the most channels), coloured like the
                 copies; and the input frames underneath.
                 Marks, at every step t >= 1 (the step that brought the state into frame t):
                   green frame : the copy / slot moving at the TRUE camera velocity
                   red dot     : the copy / slot nearest the person's apparent velocity (the
                                 centroid proxy; within 0.5 px for slots, rounded for the lattice)
                   white tick  : (felstm) a lattice velocity among the raw frame pair's top-K phase-
                                 correlation peaks -- what a frame-pair MEConvLSTM would have picked
                 melstm tiles also print the slot's velocity.
velocity_figure  melstm only: every slot's velocity at every step against the true camera
                 velocity (black) and the person proxy (grey), x and y separately. Slots are
                 offset by 0.06 px each so coinciding slots stay visible.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

_COPY_COLORS = list(plt.get_cmap("tab10").colors)


def _fmt(v):
    v = [float(c) + 0.0 for c in v]                 # + 0.0 turns -0.0 into 0.0
    return ",".join(f"{round(c):.0f}" if abs(c - round(c)) < 1e-6 else f"{c:.1f}" for c in v)


def state_figure(copies, pooled, winner, frames, labels, cam_v, person_v, slot_v=None,
                 lattice=None, pc_picks=None, title=""):
    """
    copies (T, V, H, W), pooled (T, H, W), winner (T, H, W) int, frames (T, H, W) in [0, 1];
    labels: V row labels; cam_v (T, 2) camera steps (cam_v[t] takes frame t to t+1);
    person_v (T, 2) person proxy steps (NaN = unknown); slot_v (T-1, K, 2) melstm velocities (the
    one used into frame t is slot_v[t-1]); lattice: felstm's (vx, vy) per copy; pc_picks
    (T-1, K, 2) raw frame-pair peaks. All numpy.
    """
    T, V = copies.shape[:2]
    pool_rows = 2 if V > 1 else 0
    rows = V + pool_rows + 1
    fig_w = max(7.0, T * 0.80)
    fig, axes = plt.subplots(rows, T, figsize=(fig_w, rows * 0.84 + 1.15),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.10}, squeeze=False)
    cmap_win = ListedColormap(_COPY_COLORS[:V] if V <= 10 else plt.get_cmap("tab20").colors[:V])

    def mark(ax, kind):
        H, W = copies.shape[-2:]
        if kind == "cam":
            ax.add_patch(Rectangle((-0.5, -0.5), W, H, fill=False, ec="#2ca02c", lw=2.2))
        elif kind == "person":
            ax.plot([W - 3.5], [2.5], "o", ms=4.5, mfc="#d62728", mec="white", mew=0.6)
        elif kind == "pick":
            ax.plot([2.5], [2.5], "v", ms=4.5, mfc="white", mec="black", mew=0.5)

    for t in range(T):
        cam = cam_v[t - 1] if t >= 1 else None
        per = person_v[t - 1] if t >= 1 else None
        per_ok = per is not None and np.all(np.isfinite(per))
        picks = set()
        if pc_picks is not None and t >= 1:
            picks = {tuple(int(round(c)) for c in p) for p in pc_picks[t - 1]}
        for k in range(V):
            ax = axes[k, t]
            m = copies[t, k]
            lim = max(float(np.abs(m).max()), 1e-8)
            ax.imshow(m, cmap="coolwarm", vmin=-lim, vmax=lim, interpolation="nearest")
            if t == 0:
                continue
            if lattice is not None:
                lv = tuple(lattice[k])
                if tuple(int(c) for c in cam) == lv:
                    mark(ax, "cam")
                if per_ok and tuple(int(round(c)) for c in per) == lv:
                    mark(ax, "person")
                if lv in picks:
                    mark(ax, "pick")
            elif slot_v is not None:
                sv = slot_v[t - 1, k]
                if np.all(np.round(sv) == cam):
                    mark(ax, "cam")
                if per_ok and np.abs(sv - per).max() <= 0.5:
                    mark(ax, "person")
                ax.text(0.5, 0.02, _fmt(sv), transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=5.5, color="black",
                        bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.7))
        if pool_rows:
            p = pooled[t]
            lim = max(float(np.abs(p).max()), 1e-8)
            axes[V, t].imshow(p, cmap="coolwarm", vmin=-lim, vmax=lim, interpolation="nearest")
            axes[V + 1, t].imshow(winner[t], cmap=cmap_win, vmin=-0.5, vmax=V - 0.5,
                                  interpolation="nearest")
        ax = axes[rows - 1, t]
        ax.imshow(frames[t], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        if cam is not None and np.any(cam != 0):
            ax.text(0.5, 0.02, f"cam {_fmt(cam)}", transform=ax.transAxes, ha="center",
                    va="bottom", fontsize=5.5, color="black",
                    bbox=dict(boxstyle="square,pad=0.08", fc="#c7e9c0", ec="none", alpha=0.85))
        axes[0, t].set_title(f"t={t}", fontsize=8, pad=3)
        for r in range(rows):
            axes[r, t].set_xticks([])
            axes[r, t].set_yticks([])
            for sp in axes[r, t].spines.values():
                sp.set_linewidth(0.4)
                sp.set_color("0.7")

    row_names = list(labels) + (["max over V\n(head input)", "argmax V"] if pool_rows else [])
    row_names.append("input")
    for r, name in enumerate(row_names):
        axes[r, 0].set_ylabel(name, fontsize=8, rotation=0, ha="right", va="center", labelpad=6)
    for k in range(V):
        axes[k, 0].yaxis.label.set_color(_COPY_COLORS[k % 10] if V <= 10 else "black")

    handles = [Patch(fc="none", ec="#2ca02c", lw=2, label="true camera v"),
               plt.Line2D([], [], ls="", marker="o", mfc="#d62728", mec="white",
                          label="person (proxy)")]
    if lattice is not None and pc_picks is not None:
        handles.append(plt.Line2D([], [], ls="", marker="v", mfc="white", mec="black",
                                  label="frame-pair PC peak"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, fontsize=9.5, y=0.995)
    left = min(0.2, 0.95 / fig_w + 0.02)
    fig.subplots_adjust(top=1 - 0.55 / (rows * 0.84 + 1.15), bottom=0.35 / (rows * 0.84 + 1.15),
                        left=left, right=0.995)
    return fig


def velocity_figure(slot_v, cam_v, person_v, names, title=""):
    """
    slot_v (n, T-1, K, 2), cam_v (n, T, 2), person_v (n, T, 2); names: one label per sample.
    Step t on the x axis is the step INTO frame t.
    """
    n, Tm1, K, _ = slot_v.shape
    steps = np.arange(1, Tm1 + 1)
    fig, axes = plt.subplots(n, 2, figsize=(10, 1.55 * n + 0.8), sharex=True, squeeze=False)
    for i in range(n):
        for c, comp in enumerate("xy"):
            ax = axes[i, c]
            ax.step(steps, cam_v[i, :Tm1, c], where="mid", color="black", lw=1.6,
                    label="camera (true)", zorder=3)
            ax.plot(steps, person_v[i, :Tm1, c], "o--", color="0.55", ms=3, lw=0.8,
                    label="person (proxy)", zorder=2)
            # far-off noise peaks would squash the interesting range: clip them to the frame
            # and mark them with triangles
            truth = np.concatenate([np.abs(cam_v[i, :Tm1, c]),
                                    np.abs(np.nan_to_num(person_v[i, :Tm1, c]))])
            lim = max(4.5, float(truth.max()) + 1.0)
            for k in range(K):
                y = slot_v[i, :, k, c] + 0.06 * (k - (K - 1) / 2)
                ax.plot(steps, np.clip(y, -lim, lim), "s-", color=_COPY_COLORS[k % 10], ms=2.6,
                        lw=0.9, alpha=0.9, label=f"s{k}", zorder=4)
                out = np.abs(y) > lim
                if out.any():
                    ax.plot(steps[out], np.sign(y[out]) * lim, ls="", marker="^",
                            color=_COPY_COLORS[k % 10], ms=4, zorder=5)
            ax.set_ylim(-lim - 0.3, lim + 0.3)
            ax.axhline(0, color="0.85", lw=0.6, zorder=0)
            ax.tick_params(labelsize=7)
            if c == 0:
                ax.set_ylabel(f"{names[i]}\n$v_x$", fontsize=8)
            else:
                ax.set_ylabel("$v_y$", fontsize=8)
    for c in range(2):
        axes[-1, c].set_xlabel("step into frame t", fontsize=8)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(lab), fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(title, fontsize=9.5, y=0.999)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig
