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
    c_states=None,
    split_name="train",
    num_samples=3,
    subsample_t=1,
    input_frames=None,
):
    """
    Visualise the per-slot channel-mean h (and optionally c) maps over time,
    for a handful of random samples — the exact h.mean(dim=2)/c.mean(dim=2)
    reduction Seq2SeqMEConvLSTM._track_velocities correlates against, so
    this shows what the velocity tracker actually sees at every step.

    h_states, c_states : (B, T, K, H, W), from model(..., return_states=True)
        (the dict's "h"/"c" entries). c_states is optional.
    input_frames : if given, decoder timesteps (t >= input_frames) are
        labelled in a different colour to mark the encoder/decoder boundary.
    """
    B, T, K, H, W = h_states.shape
    num_samples = min(num_samples, B)
    indices = np.random.choice(B, num_samples, replace=False)

    T_shown = max(1, T // subsample_t)
    rows = K * (2 if c_states is not None else 1)

    for idx in indices:
        h_sample = h_states[idx].detach().cpu()   # (T, K, H, W)
        c_sample = c_states[idx].detach().cpu() if c_states is not None else None

        fig, axes = plt.subplots(
            rows, T_shown,
            figsize=(max(6, T_shown * 1.1), max(2, rows * 1.3)),
            gridspec_kw={"wspace": 0.05, "hspace": 0.2},
            squeeze=False,
        )

        for tt in range(T_shown):
            t = tt * subsample_t
            is_decoder = input_frames is not None and t >= input_frames
            title_color = "crimson" if is_decoder else "black"

            for k in range(K):
                v = h_sample[t, k]
                vmax = v.abs().max().clamp(min=1e-8).item()
                axes[k, tt].imshow(v, cmap="coolwarm", vmin=-vmax, vmax=vmax)
                axes[k, tt].axis("off")

            if c_sample is not None:
                for k in range(K):
                    v = c_sample[t, k]
                    vmax = v.abs().max().clamp(min=1e-8).item()
                    axes[K + k, tt].imshow(v, cmap="coolwarm", vmin=-vmax, vmax=vmax)
                    axes[K + k, tt].axis("off")

            axes[0, tt].set_title(f"t={t}", fontsize=8, color=title_color)

        for k in range(K):
            axes[k, 0].text(-0.4, 0.5, f"h slot{k}", rotation=90,
                             va="center", ha="center", fontsize=8,
                             transform=axes[k, 0].transAxes)
        if c_sample is not None:
            for k in range(K):
                axes[K + k, 0].text(-0.4, 0.5, f"c slot{k}", rotation=90,
                                     va="center", ha="center", fontsize=8,
                                     transform=axes[K + k, 0].transAxes)

        fig.suptitle(f"{split_name} h/c evolution — sample {idx}"
                     + (" (red titles = decoder)" if input_frames is not None else ""),
                     fontsize=10)

        wandb.log({f"{split_name}_states_sample{idx}": wandb.Image(fig)})
        plt.close(fig)