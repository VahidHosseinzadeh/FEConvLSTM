import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
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