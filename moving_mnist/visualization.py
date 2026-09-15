import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np
import math
import wandb

def log_sequence_predictions(
        input_seq, target_seq, output_seq,
        split_name,
        num_samples=2,
        frames_per_row=10,         
        upsample_scale=4,          
        dpi=160                   
    ):
    """
    Visualise GT / prediction / |diff| for a handful of samples.

    • frames_per_row   controls the wrapping, keeping height reasonable even
                       for very long sequences.
    • upsample_scale   multiplies the resolution of every frame to make small
                       MNIST digits clearly visible.
    """
    batch_size = input_seq.size(0)
    num_samples = min(num_samples, batch_size)
    indices = np.random.choice(batch_size, num_samples, replace=False)

    input_len  = input_seq.size(1)
    target_len = target_seq.size(1)
    total_len  = input_len + target_len

    # grid layout parameters ----------------------------------------------------
    ncols = min(frames_per_row, total_len)           # frames per grid‐row
    nrows = math.ceil(total_len / ncols)             # how many rows per grid
    # --------------------------------------------------------------------------

    # figure size in *inches*: width ~ ncols * upsample_scale * 0.25
    fig_w = (ncols * upsample_scale) * 0.25
    fig_h = (3 * nrows * upsample_scale) * 0.25      # 3 rows (GT / pred / diff)

    fig, axes = plt.subplots(
        3, num_samples,
        figsize=(fig_w * num_samples, fig_h),
        dpi=dpi,
        squeeze=False
    )

    for i, idx in enumerate(indices):
        s_in   = input_seq[idx].cpu()
        s_tgt  = target_seq[idx].cpu()
        s_pred = output_seq[idx].cpu()

        # full sequences --------------------------------------------------------
        full_gt   = torch.cat([s_in,  s_tgt],  dim=0)
        full_pred = torch.cat([s_in,  s_pred], dim=0)
        full_diff = torch.cat([torch.zeros_like(s_in),
                               torch.abs(s_pred - s_tgt)], dim=0)

        for row, tensor, title in zip(
                range(3),
                (full_gt, full_pred, full_diff),
                ("Ground Truth", "Prediction", "Difference |Δ|")):

            grid = make_grid(
                tensor, nrow=ncols, normalize=True, padding=1
            )

            # upscale the whole grid so each digit is bigger -------------------
            grid = F.interpolate(
                grid.unsqueeze(0),  # [1, C, H, W]
                scale_factor=upsample_scale,
                mode='nearest'
            ).squeeze(0)

            axes[row, i].imshow(grid.permute(1, 2, 0).numpy(),
                                interpolation='nearest')
            axes[row, i].set_title(f"Sample {i+1} – {title}",
                                   fontsize=10)
            axes[row, i].axis('off')

    plt.tight_layout()
    wandb.log({f"{split_name}_sequences": wandb.Image(fig)})
    plt.close(fig)
    

def log_sequence_predictions_new(
    input_seq, target_seq, output_seq,
    split_name,    
    num_samples: int = 4,          # number of sequences to visualise
    vmax_diff: float = 1.0,        # clip range for the signed difference plot
    subsample_t: int = 1,          # subsample the time dimension by this factor
    device: torch.device | None = None,
):
    """
    Visualise ground–truth, prediction, and signed error for a handful of sequences.

    """
    T = target_seq.shape[1]
    T = T // subsample_t
    num_samples = min(num_samples, target_seq.shape[0])

    # --- iterate over the first num_samples sequences ------------------------
    for idx in range(num_samples):
        gt_seq   = target_seq[idx].detach().cpu().squeeze()       # (T, H, W)
        pred_seq = output_seq[idx].detach().cpu().squeeze()   # (T, H, W)
        diff_seq = pred_seq - gt_seq                     # signed error

        # ----------- set up a long thin figure --------------------------------
        fig_height = 3          # one row per line, in inches
        fig_width  = max(6, T)
        fig, axes  = plt.subplots(
            3, T,
            figsize=(fig_width, fig_height),
            gridspec_kw={"wspace": 0.005, "hspace": 0.03},  # Reduced spacing between elements
        )

        # make axes always iterable in both dims
        if T == 1:
            axes = axes.reshape(3, 1)

        # ----------- plot -----------------------------------------------------
        for t in range(T):
            # top row – ground truth
            axes[0, t].imshow(gt_seq[t*subsample_t], cmap="gray", vmin=0, vmax=1)
            # centre row – predictions
            axes[1, t].imshow(pred_seq[t*subsample_t], cmap="gray", vmin=0, vmax=1)
            # bottom row – signed difference
            axes[2, t].imshow(
                diff_seq[t*subsample_t],
                cmap="bwr",
                vmin=-vmax_diff,
                vmax=vmax_diff,
            )

            # cosmetic clean-up
            for r in range(3):
                axes[r, t].axis("off")

        # label the rows once (left-most subplot)
        axes[0, 0].set_ylabel("GT",    rotation=0, labelpad=20, fontsize=10)
        axes[1, 0].set_ylabel("Pred",  rotation=0, labelpad=15, fontsize=10)
        axes[2, 0].set_ylabel("Error", rotation=0, labelpad=18, fontsize=10)

        # optional overall title
        fig.suptitle(f"{split_name} sample {idx}", fontsize=12)

        # ----------- log to wandb & close -------------------------------------
        wandb.log({f"{split_name}sequence_{idx}": wandb.Image(fig)})
        plt.close(fig)


def log_state_evolution(
    h_states,
    gt_frames=None,
    split_name="train",
    num_samples=3,
    subsample_t=1,
    input_frames=None,
    motion=None,
    v_list=None,
):
    """
    Visualise the per-slot channel-mean h maps over time — the exact
    h.mean(dim=2) reduction Seq2SeqMEConvLSTM.track_velocities correlates
    against, so this shows what the velocity tracker actually sees — for a
    handful of random samples, with the true frame directly below for
    comparison.

    h_states   : (B, T, K, H, W), states["h"] from model(..., return_states=True).
    gt_frames  : (B, T, C, H, W), the true frame at each step, e.g.
        torch.cat([input_seq, target_seq], dim=1).
    input_frames : if given, decoder timesteps (t >= input_frames) are
        labelled in a different colour to mark the encoder/decoder boundary;
        also the length of the input window used to select slots below.
    motion, v_list : optional, together select which of the K slots to draw
        instead of all of them. FEConvLSTM has one slot per (vx, vy)
        candidate on a dense grid (K can be dozens for a large v_range) --
        showing all of them is unreadable, so instead we show only the
        slots whose velocity is actually taken by some digit at some point
        in the input window (the first input_frames steps), matched against
        v_list = model.cell.v_list. motion is (B, T, N, 2) GT per-digit
        velocity (frame t -> t+1). Leave both None (default, e.g. for
        MEConvLSTM's few learned slots) to show every slot as before.
    """
    B, T, K, H, W = h_states.shape
    num_samples = min(num_samples, B)
    indices = np.random.choice(B, num_samples, replace=False)

    T_shown = max(1, T // subsample_t)
    has_gt_frames = gt_frames is not None
    select_slots = motion is not None and v_list is not None
    v_index = {tuple(v): k for k, v in enumerate(v_list)} if select_slots else None

    for idx in indices:
        h_sample = h_states[idx].detach().cpu()   # (T, K, H, W)
        gt_sample = gt_frames[idx].detach().cpu() if has_gt_frames else None   # (T, C, H, W)

        if select_slots:
            window = motion[idx, :input_frames].reshape(-1, 2).tolist()
            observed = {tuple(int(x) for x in v) for v in window}
            slot_idx = sorted(v_index[v] for v in observed if v in v_index)
            if not slot_idx:
                slot_idx = list(range(K))  # no grid match -- fall back to showing all
            h_sample = h_sample[:, slot_idx]
            slot_labels = [f"v={v_list[k]}" for k in slot_idx]
        else:
            slot_labels = [f"h slot{k}" for k in range(K)]

        rows_k = h_sample.shape[1]
        rows = rows_k + int(has_gt_frames)
        gt_row = rows_k

        fig, axes = plt.subplots(
            rows, T_shown,
            figsize=(max(6, T_shown * 1.3), max(2, rows * 1.3) + 0.6),
            gridspec_kw={"wspace": 0.05, "hspace": 0.35},
            squeeze=False,
        )

        for tt in range(T_shown):
            t = tt * subsample_t
            is_decoder = input_frames is not None and t >= input_frames
            title_color = "crimson" if is_decoder else "black"

            for k in range(rows_k):
                v = h_sample[t, k]
                vmax = v.abs().max().clamp(min=1e-8).item()
                axes[k, tt].imshow(v, cmap="coolwarm", vmin=-vmax, vmax=vmax)
                axes[k, tt].axis("off")

            if has_gt_frames:
                axes[gt_row, tt].imshow(
                    gt_sample[t].mean(dim=0), cmap="gray", vmin=0, vmax=1
                )
                axes[gt_row, tt].axis("off")

            axes[0, tt].set_title(f"t={t}", fontsize=8, color=title_color)

        for k in range(rows_k):
            axes[k, 0].text(-0.4, 0.5, slot_labels[k], rotation=90,
                             va="center", ha="center", fontsize=8,
                             transform=axes[k, 0].transAxes)
        if has_gt_frames:
            axes[gt_row, 0].text(-0.4, 0.5, "frame\n(GT)", rotation=90,
                                  va="center", ha="center", fontsize=7,
                                  transform=axes[gt_row, 0].transAxes)

        title = f"{split_name} h evolution — sample {idx}"
        if input_frames is not None:
            title += " (red titles = decoder)"
        fig.suptitle(title, fontsize=10)

        wandb.log({f"{split_name}_states_sample{idx}": wandb.Image(fig)})
        plt.close(fig)


def log_velocity_report(summary, split_name="train", epoch=None):
    """
    Log a VelocityMetrics.summary() dict to wandb: a per-timestep table
    (accuracy/mean-L2/correct/total, one row per t) plus overall/encoder/
    decoder/last-step scalars. The scalars are logged under a stable key
    each call, so wandb charts them as a trend across calls (epochs) rather
    than needing a hand-built "accuracy per epoch" table.

    summary : dict from VelocityMetrics.summary(), or None (no-op if so —
        summary() returns None when nothing has been recorded yet).
    epoch   : if given, included in the logged dict so points align with
        the "epoch" x-axis already used elsewhere (e.g. train.py's
        train_loss/val_loss logging).
    """
    if summary is None:
        return

    # acc_pct is the per-timestep (stepwise) assignment -- the estimator at
    # time t on its own. acc_pct_seq is the sequence-locked assignment kept
    # alongside it; their difference (binding_loss) is accuracy lost purely
    # to the slot->digit binding disagreeing across time. See VelocityMetrics.
    table = wandb.Table(columns=["t", "acc_pct", "acc_pct_seq", "binding_loss",
                                 "mean_l2", "mean_l2_seq", "correct", "total"])
    for t in range(summary["T"]):
        table.add_data(
            t + 1,
            summary["per_t_acc_stepwise"][t],
            summary["per_t_acc"][t],
            summary["per_t_binding_loss"][t],
            summary["per_t_l2_stepwise"][t],
            summary["per_t_l2"][t],
            summary["per_t_correct_stepwise"][t],
            summary["per_t_total"][t],
        )

    log_dict = {
        f"{split_name}_vel_table"       : table,
        # primary: per-timestep assignment
        f"{split_name}_vel_overall_acc" : summary["overall_acc_stepwise"],
        f"{split_name}_vel_overall_l2"  : summary["overall_l2_stepwise"],
        f"{split_name}_vel_encoder_acc" : summary["encoder_acc_stepwise"],
        f"{split_name}_vel_decoder_acc" : summary["decoder_acc_stepwise"],
        f"{split_name}_vel_last_step_acc": summary["last_step_acc_stepwise"],
        f"{split_name}_vel_last_step_l2" : summary["last_step_l2_stepwise"],
        # the bootstrap step (t=1) is parameter-free: on a fixed eval set this
        # must be flat across epochs. If it is not, the metric is at fault.
        f"{split_name}_vel_bootstrap_acc": summary["per_t_acc_stepwise"][0],
        # sequence-locked, kept for continuity with earlier runs
        f"{split_name}_vel_overall_acc_seq" : summary["overall_acc"],
        f"{split_name}_vel_encoder_acc_seq" : summary["encoder_acc"],
        f"{split_name}_vel_decoder_acc_seq" : summary["decoder_acc"],
    }
    if epoch is not None:
        log_dict["epoch"] = epoch

    wandb.log(log_dict)

def log_motion_classification_states(
    h_states,
    frames,
    mask_track=None,
    velocities=None,
    v_list=None,
    gt_motion=None,
    split_name="val",
    epoch=None,
    step=None,
    num_samples=2,
    subsample_t=1,
    max_slots=6,
    save_dir=None,
    dpi=160,
    show_shape_readout=False,
    mask_color="#E69F00",
    mask_lw=1.2,
    mask_halo=1.6,
    mask_halo_color="black",
):
    """
    Publication-quality view of what each velocity copy accumulated.

    This is the picture the whole experiment rests on. The bottom two rows are
    ground truth: the frames the model saw -- which are noise, the digit is
    genuinely not in them -- and the figure's true aperture. Above them sits one
    row per velocity copy, so the question the figure asks reads top-to-bottom:
    does the copy transported at the FIGURE's velocity develop the digit's shape
    while the others stay textureless?

    h_states   : (B, T, V, H, W) per-timestep CHANNEL-MEAN of each velocity copy.
    frames     : (B, T, C, H, W) the input the model saw.
    mask_track : (B, T, N, H, W) ground-truth figure aperture, or None. Drawn as
        a contour ON the input frame rather than as its own row: a row of pure
        noise beside a row of pure mask wastes vertical space and reads oddly,
        while the outline puts the answer key exactly where the reader needs it
        -- over the frame that appears to contain nothing.

        mask_halo draws a darker, thicker stroke UNDER the coloured line. That is
        what makes it survive: against high-frequency noise a plain hairline is
        lighter than the texture in some places and darker in others, so it keeps
        disappearing. A halo gives the line local contrast wherever it runs.
        Compared side by side (see the notebook), a 0.7pt line with no halo is
        barely findable; 1.2pt over a 1.6pt black halo reads at every timestep.
        Okabe-Ito amber by default -- colourblind-safe and the strongest hue
        against neutral grey; "#56B4E9" (sky) and "white" also work.
    velocities : (B, T-1, K, 2) tracked slot velocities (melstm), or None.
    v_list     : the fixed lattice, model.cell.v_list (felstm/lstm), or None.
    gt_motion  : (B, T, N+1, 2) true (vx, vy), figures then background last.
    save_dir   : also write each panel as PNG and PDF here, for the paper.
    show_shape_readout : add a row under the FIGURE copy holding the local
        variance of its channel mean -- the picture form of
        val_state_shape_iou, showing where that copy accumulated coherent
        structure. Off for paper panels: it is a diagnostic about the metric,
        not about the model, and the metric turned out to measure transport
        coherence rather than anything predictive of accuracy.

    Slot selection. felstm's lattice can be dozens of copies, which is
    unreadable, so the ones worth seeing are picked: the copy matching the
    figure's velocity, the copy matching the background's, and a few others as a
    control. Selection uses the velocity at the LAST encoder step -- the one h_T
    was most recently transported at, and the same step slot_hit_fig uses.

    Sample keys are positional and stable across epochs on purpose: wandb then
    gives one slider per sample showing the SAME sequence developing as training
    proceeds. Random sample indices would scatter the keys and make that
    impossible.

    `step` MUST be passed by any caller whose other wandb.log calls use an
    explicit step. Every sample is logged in ONE call at that step: a wandb.log
    without `step` commits and advances wandb's internal counter, so logging per
    sample would push the counter past the epoch and the NEXT epoch's metrics
    would be silently dropped.
    """
    payload = {}
    B, T, V, H, W = h_states.shape
    n = min(num_samples, B)
    steps = list(range(0, T, max(1, subsample_t)))
    if steps[-1] != T - 1:
        steps.append(T - 1)

    # One untransported state is not a "lattice", so the lattice vocabulary --
    # per-copy velocity labels, and the note about velocities it cannot
    # represent -- is meaningless for lstm and was only ever noise there.
    has_lattice = v_list is not None and len(v_list) > 1

    if save_dir is not None:
        from pathlib import Path as _Path
        save_dir = _Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        h = h_states[i].detach().cpu()
        fr = frames[i].detach().cpu()
        mk = mask_track[i].detach().cpu() if mask_track is not None else None

        v_fig = v_bg = None
        if gt_motion is not None:
            g = gt_motion[i].detach().cpu()
            v_fig = tuple(int(x) for x in g[-2, 0])
            v_bg = tuple(int(x) for x in g[-2, -1])

        missing = []
        if has_lattice:                                       # felstm
            index = {tuple(v): k for k, v in enumerate(v_list)}
            chosen, labels = [], []
            for tag, vel in (("FIGURE", v_fig), ("background", v_bg)):
                if vel in index and index[vel] not in chosen:
                    chosen.append(index[vel])
                    labels.append(f"$v$ = {vel}\n({tag})")
                elif vel is not None:
                    missing.append(f"{tag} {vel}")
            for k in range(len(v_list)):
                if len(chosen) >= max_slots:
                    break
                if k not in chosen:
                    chosen.append(k)
                    labels.append(f"$v$ = {tuple(v_list[k])}")
        elif velocities is not None:                          # melstm
            vl = velocities[i, -1].detach().cpu().round().long()
            chosen, labels = [], []
            for k in range(min(V, max_slots)):
                vk = tuple(int(x) for x in vl[k])
                # No velocity on the label: a melstm slot RE-ESTIMATES its velocity
                # every step, so any single value is a snapshot of the last one and
                # misrepresents the sequence. Which motion the slot followed is the
                # informative part, and that is what the tag says.
                tag = (" (FIGURE)" if vk == v_fig else
                       (" (background)" if vk == v_bg else ""))
                chosen.append(k)
                labels.append(f"slot {k}{tag}")
        else:                                                 # lstm
            chosen, labels = [0], ["hidden state\n(no transport)"]

        def local_var(x, k=3):
            """Local variance in a kxk window; circular, because the canvas is a torus."""
            x = x[None, None]
            pad = k // 2
            mu = torch.nn.functional.avg_pool2d(
                torch.nn.functional.pad(x, (pad,) * 4, mode="circular"), k, stride=1)
            mu2 = torch.nn.functional.avg_pool2d(
                torch.nn.functional.pad(x * x, (pad,) * 4, mode="circular"), k, stride=1)
            return (mu2 - mu * mu).clamp(min=0)[0, 0]

        fig_row = next((r for r, lab in enumerate(labels) if "FIGURE" in lab), None)
        add_readout = show_shape_readout and fig_row is not None

        rows = len(chosen) + int(add_readout) + 1
        readout_row = len(chosen)
        frame_row = readout_row + int(add_readout)

        fig_w = max(7, len(steps) * 0.92)
        fig, axes = plt.subplots(
            rows, len(steps),
            figsize=(fig_w, rows * 1.02 + 0.85),
            gridspec_kw={"wspace": 0.035, "hspace": 0.16},
            squeeze=False,
        )

        # Reserve the left margin from the LONGEST label actually drawn. A fixed
        # fraction clips "slot 0 (background)" in the wandb copy -- the saved PNG
        # and PDF escape it only because bbox_inches="tight" crops afterwards.
        all_labels = [ln for lab in labels for ln in lab.split("\n")]
        all_labels += ["input frame", "(figure outlined)"]
        longest = max(len(x.replace("$", "")) for x in all_labels)
        left_margin = min(0.30, (longest * 0.078 + 0.30) / fig_w)

        for col, t in enumerate(steps):
            for r, k in enumerate(chosen):
                m = h[t, k]
                lim = m.abs().max().clamp(min=1e-8).item()
                axes[r, col].imshow(m, cmap="coolwarm", vmin=-lim, vmax=lim,
                                    interpolation="nearest")
            if add_readout:
                axes[readout_row, col].imshow(local_var(h[t, chosen[fig_row]]),
                                              cmap="magma", interpolation="nearest")
            ax = axes[frame_row, col]
            ax.imshow(fr[t].mean(0), cmap="gray", interpolation="nearest")
            if mk is not None:
                # Contour rather than a pixel mask: anti-aliased, hairline, and it
                # sits over the texture instead of blocking it. Drawn on a
                # circularly PADDED copy and then clipped back, so a figure that
                # wraps around the torus keeps a continuous outline instead of
                # picking up a spurious straight segment along the frame edge.
                m2 = mk[t].amax(0).numpy()
                pad = 3
                mp = np.pad(m2, pad, mode="wrap")
                Hm, Wm = m2.shape
                cs = ax.contour(mp, levels=[0.5], colors=[mask_color],
                                linewidths=mask_lw, antialiased=True,
                                extent=(-pad - 0.5, Wm + pad - 0.5,
                                        Hm + pad - 0.5, -pad - 0.5))
                if mask_halo:
                    fx = [patheffects.withStroke(linewidth=mask_lw + mask_halo,
                                                 foreground=mask_halo_color,
                                                 alpha=0.9)]
                    # matplotlib >= 3.8 makes ContourSet a Collection itself;
                    # older versions expose .collections.
                    for coll in (cs.collections if hasattr(cs, "collections") else [cs]):
                        coll.set_path_effects(fx)
                ax.set_xlim(-0.5, Wm - 0.5)
                ax.set_ylim(Hm - 0.5, -0.5)
            axes[0, col].set_title(f"$t$ = {t}", fontsize=11, pad=5)
            for r in range(rows):
                axes[r, col].set_xticks([]); axes[r, col].set_yticks([])
                for sp in axes[r, col].spines.values():
                    sp.set_linewidth(0.4); sp.set_color("0.75")

        def ylabel(row, text, weight="normal"):
            axes[row, 0].set_ylabel(text, fontsize=10, rotation=0, ha="right",
                                    va="center", labelpad=10, fontweight=weight)

        for r, lab in enumerate(labels):
            ylabel(r, lab, "bold" if "FIGURE" in lab else "normal")
        if add_readout:
            ylabel(readout_row, "local variance\n(figure copy)")
        ylabel(frame_row, "input frame" + ("\n(figure outlined)" if mk is not None else ""))

        # No ground-truth velocity in the title: the motion is piecewise-constant,
        # so quoting a single v_fig / v_bg for the whole sequence is simply wrong.
        # Where a velocity IS constant -- felstm's lattice copies -- it is shown on
        # the row instead.
        what = "hidden state over time" if not has_lattice and velocities is None \
            else "hidden state per velocity copy"
        title = f"{split_name} — {what}, sample {i}"
        if epoch is not None:
            title += f"   (epoch {epoch})"
        if missing:
            title += f"\nnot representable on this model's velocity lattice: {', '.join(missing)}"
        fig.suptitle(title, fontsize=12, y=0.995)
        fig.subplots_adjust(top=0.90 - 0.02 * (title.count(chr(10))),
                            bottom=0.015, left=left_margin, right=0.995)

        if save_dir is not None:
            # No epoch in the name: only the final epoch is written, so a bare
            # name is what a paper wants to cite.
            stem = f"{split_name}_states_sample{i}"
            fig.savefig(save_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
            fig.savefig(save_dir / f"{stem}.pdf", bbox_inches="tight")

        payload[f"{split_name}_states_sample{i}"] = wandb.Image(fig)
        plt.close(fig)

    if payload:
        wandb.log(payload, step=step) if step is not None else wandb.log(payload)
