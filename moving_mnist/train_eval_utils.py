import torch
import wandb
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from moving_mnist_dataset import FixedVelocityMovingMNIST
from visualization import (
    log_sequence_predictions,
    log_sequence_predictions_new,
    log_state_evolution,
    log_velocity_report,
)
from velocity_metrics import VelocityMetrics
from velocity_model_based_MEConvLSTM_model import Seq2SeqMEConvLSTM


def _unpack_batch(batch, device):
    """
    Datasets yield (seq, label) or, with return_motion=True, (seq, label,
    motion). Velocity reporting below activates iff motion is present.
    """
    seq = batch[0].to(device)
    gt_motion = batch[2].to(device) if len(batch) > 2 else None
    return seq, gt_motion


def _run_model(model, input_seq, pred_len, target_seq,
               want_velocity, want_states, track_decoder_velocity=None):
    """
    Single call site for every model type.

    MELSTM always receives target_seq when available — NOT teacher forcing:
    its training protocol tracks decoder velocities against the true next
    frame (v = track(h, target_seq[:, t])). Without it the model silently
    runs in inference mode (last encoder velocity frozen for the whole
    rollout — garbage early in training) and learning stalls.

    track_decoder_velocity=None (default) picks the protocol by phase:
    training -> tracked (required, see above); eval -> frozen (honest
    deployable inference). Pass True explicitly for the oracle eval
    (GT-tracked decoder velocities), e.g. via --eval_velocity_mode.

    Returns (output_seq, pred_motion_or_None, states_or_None).
    """
    if isinstance(model, Seq2SeqMEConvLSTM):
        if track_decoder_velocity is None:
            track_decoder_velocity = model.training
        result = model(
            input_seq,
            pred_len=pred_len,
            target_seq=target_seq,
            track_decoder_velocity=track_decoder_velocity,
            return_velocity=want_velocity,
            return_states=want_states,
        )
        if not (want_velocity or want_states):
            return result, None, None
        result = list(result if isinstance(result, tuple) else [result])
        output_seq = result.pop(0)
        pred_motion = result.pop(0) if want_velocity else None
        states = result.pop(0) if want_states else None
        return output_seq, pred_motion, states

    return model(input_seq, pred_len=pred_len), None, None


class ValCurveRecorder:
    """
    Fine-grained loss-vs-training-batches curves (paper-style validation
    curve with a band), recorded during training:

      - training loss of every optimizer step (free, already computed);
      - every `interval` steps, mean/std of the per-sequence validation
        loss on a small FIXED set of sequences.

    The val set is materialized into a tensor once at construction: the
    on-the-fly Moving MNIST datasets resample content on every access, so
    holding indices fixed is not enough to keep the measured set fixed.

    The per-sequence loss replicates train.py's MSEPlusL1Loss with default
    weights (MSE + L1). Models are evaluated in honest inference mode
    (no target_seq), matching eval_epoch.
    """

    def __init__(self, val_dataset, n_sequences, interval, input_frames,
                 device, batch_size=64):
        n = min(n_sequences, len(val_dataset))
        self.data = torch.stack([val_dataset[i][0] for i in range(n)])
        self.interval = interval
        self.input_frames = input_frames
        self.device = device
        self.batch_size = batch_size
        self.step = 0
        self.train_steps, self.train_losses = [], []
        self.val_steps, self.val_means, self.val_stds = [], [], []

    @torch.no_grad()
    def _val_loss(self, model):
        was_training = model.training
        model.eval()
        per_seq = []
        for s in range(0, self.data.size(0), self.batch_size):
            seq = self.data[s:s + self.batch_size].to(self.device)
            inp = seq[:, :self.input_frames]
            tgt = seq[:, self.input_frames:]
            pred, _, _ = _run_model(model, inp, tgt.size(1), None, False, False)
            d = pred - tgt
            per_seq.append((d.pow(2).mean(dim=(1, 2, 3, 4))
                            + d.abs().mean(dim=(1, 2, 3, 4))).cpu())
        if was_training:
            model.train()
        per_seq = torch.cat(per_seq)
        return per_seq.mean().item(), per_seq.std().item()

    def on_batch(self, model, train_loss):
        self.step += 1
        self.train_steps.append(self.step)
        self.train_losses.append(float(train_loss))
        if self.interval > 0 and self.step % self.interval == 0:
            m, s = self._val_loss(model)
            self.val_steps.append(self.step)
            self.val_means.append(m)
            self.val_stds.append(s)
            wandb.log({"val_curve_loss": m, "val_curve_std": s,
                       "global_batch": self.step})

    def as_dict(self):
        return {
            "train_curve": {"step": self.train_steps,
                            "loss": self.train_losses},
            "val_curve": {"step": self.val_steps,
                          "loss_mean": self.val_means,
                          "loss_std": self.val_stds},
        }


def train_epoch(model, dataloader, optimizer, criterion, device, input_frames, grad_clip=None,
                curve_recorder=None):
    model.train()
    running_loss = 0.0
    velocity_metrics = VelocityMetrics()
    has_velocity_data = False

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for i, batch in enumerate(pbar):
        seq, gt_motion = _unpack_batch(batch, device)   # (B, seq_len, C, H, W)
        input_seq = seq[:, :input_frames]
        target_seq = seq[:, input_frames:]
        pred_len = target_seq.size(1)

        is_melstm = isinstance(model, Seq2SeqMEConvLSTM)
        want_velocity = is_melstm and gt_motion is not None
        want_states = want_velocity and i == 0

        optimizer.zero_grad()
        output_seq, pred_motion, states = _run_model(
            model, input_seq, pred_len, target_seq, want_velocity, want_states
        )
        loss = criterion(output_seq, target_seq)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        if want_velocity:
            velocity_metrics.update(pred_motion, gt_motion)
            has_velocity_data = True

        batch_loss = loss.item()
        running_loss += batch_loss * seq.size(0)
        pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

        if curve_recorder is not None:
            curve_recorder.on_batch(model, batch_loss)

        if i == 0:
            log_sequence_predictions(input_seq, target_seq, output_seq, split_name="train")
            if states is not None:
                log_state_evolution(
                    states["h"],
                    gt_frames=torch.cat([input_seq, target_seq], dim=1),
                    split_name="train",
                    input_frames=input_frames,
                )

    if has_velocity_data:
        velocity_metrics.report("Training Velocity")
        log_velocity_report(velocity_metrics.summary(), split_name="train")

    return running_loss / len(dataloader.dataset)


def eval_epoch(model, dataloader, criterion, device, input_frames, epoch, split_name,
               decoder_velocity_mode="frozen"):
    """
    decoder_velocity_mode : "frozen" (default) — honest deployable inference,
        the last encoder velocity rolls the whole horizon; velocity metrics
        then cover encoder steps only. "tracked" — oracle protocol: MELSTM's
        decoder velocities are tracked against the true next frames (upper
        bound; the gap to "frozen" isolates velocity-estimation error from
        rendering error).
    """
    model.eval()
    track = decoder_velocity_mode == "tracked"
    running_loss = 0.0
    velocity_metrics = VelocityMetrics()
    has_velocity_data = False

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for i, batch in enumerate(pbar):
            seq, gt_motion = _unpack_batch(batch, device)
            input_seq = seq[:, :input_frames]
            target_seq = seq[:, input_frames:]
            pred_len = target_seq.size(1)

            is_melstm = isinstance(model, Seq2SeqMEConvLSTM)
            want_velocity = is_melstm and gt_motion is not None
            want_states = want_velocity and i == 0

            output_seq, pred_motion, states = _run_model(
                model, input_seq, pred_len, target_seq, want_velocity, want_states,
                track_decoder_velocity=True if track else None,
            )
            loss = criterion(output_seq, target_seq)
            batch_loss = loss.item()
            running_loss += batch_loss * seq.size(0)
            pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

            if want_velocity:
                velocity_metrics.update(pred_motion, gt_motion)
                has_velocity_data = True

            if i == 0:
                log_sequence_predictions(input_seq, target_seq, output_seq, split_name=split_name)
                if states is not None:
                    log_state_evolution(
                        states["h"],
                        gt_frames=torch.cat([input_seq, target_seq], dim=1),
                        split_name=split_name,
                        input_frames=input_frames,
                    )

    if has_velocity_data:
        velocity_metrics.report(f"{split_name} Velocity")
        log_velocity_report(velocity_metrics.summary(), split_name=split_name, epoch=epoch)

    return running_loss / len(dataloader.dataset)


def eval_len_generalization(model, dataloader, device, input_frames, subsample_t=1,
                            n_strip_sequences=3):
    """
    Returns:
        mean_err  – numpy array [T]  (MSE at each future step, averaged over test set)
        std_err   – numpy array [T]  (sample‑wise std at each step)
        details   – dict for offline comparison plots (plot_len_gen_comparison.py):
            per_seq_err : numpy [N, T]  full per-sequence per-step MSE matrix —
                enables paired cross-model statistics on a fixed benchmark set
            per_seq_err_copy_last : numpy [N, T]  same matrix for the
                copy-last-context-frame baseline (model-free reference line)
            strip_input / strip_gt / strip_pred : numpy frames of the first
                n_strip_sequences benchmark sequences, for qualitative strips
            boundary_age : numpy [N, n_digits], only when the dataset returns
                motion — frames between each digit's last velocity change and
                the context/prediction boundary (evidence available to lock
                onto the frozen velocity). Enables binning decoder error by
                adaptation time offline.
    """
    model.eval()
    velocity_metrics = VelocityMetrics()
    has_velocity_data = False
    first_pass = True
    per_seq_chunks = []
    baseline_chunks = []
    age_chunks = []
    strip = {}

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating Length Generalization", leave=False)
        for batch in pbar:
            seq, gt_motion = _unpack_batch(batch, device)
            inp, tgt = seq[:, :input_frames], seq[:, input_frames:]
            T = tgt.size(1)

            # Copy-last-frame baseline: repeat the final context frame.
            baseline_chunks.append(
                ((inp[:, -1:] - tgt) ** 2).mean(dim=(2, 3, 4)).detach().cpu()
            )

            if gt_motion is not None:
                # Frames between each digit's last velocity change and the
                # boundary. motions[t] is the step frame t -> t+1; with
                # freeze_after=input_frames the frozen velocity is
                # motions[input_frames-2], active since its last change.
                f = input_frames
                m = gt_motion[:, :f - 1]                     # steps 0 .. f-2
                changed = (m[:, 1:] != m[:, :-1]).any(dim=-1)  # (B, f-2, N)
                idx = torch.arange(1, f - 1, device=changed.device).view(1, -1, 1)
                t_last = torch.where(changed, idx,
                                     torch.zeros_like(idx)).amax(dim=1)  # (B, N)
                age_chunks.append(((f - 1) - t_last).detach().cpu())

            is_melstm = isinstance(model, Seq2SeqMEConvLSTM)
            want_velocity = is_melstm and gt_motion is not None
            want_states = want_velocity and first_pass

            # target_seq=None: length generalization is pure rollout.
            pred, pred_motion, states = _run_model(
                model, inp, T, None, want_velocity, want_states
            )

            if want_velocity:
                velocity_metrics.update(pred_motion, gt_motion)
                has_velocity_data = True

            # MSE per example per timestep  →  [B, T]
            per_ex_t = ((pred - tgt) ** 2).mean(dim=(2, 3, 4))
            per_seq_chunks.append(per_ex_t.detach().cpu())

            if first_pass:
                first_pass = False
                n = n_strip_sequences
                strip = {
                    "strip_input": inp[:n].detach().cpu().numpy(),
                    "strip_gt":    tgt[:n].detach().cpu().numpy(),
                    "strip_pred":  pred[:n].detach().cpu().numpy(),
                }

                log_sequence_predictions_new(
                    inp, tgt, pred, split_name="len_gen",
                    num_samples=10, device=device, subsample_t=subsample_t,
                )
                if states is not None:
                    log_state_evolution(
                        states["h"],
                        gt_frames=torch.cat([inp, tgt], dim=1),
                        split_name="len_gen",
                        subsample_t=subsample_t,
                        input_frames=input_frames,
                    )

            pbar.set_postfix({"loss": per_ex_t.mean().item()})

    if has_velocity_data:
        velocity_metrics.report("Length Generalization Velocity")
        log_velocity_report(velocity_metrics.summary(), split_name="len_gen")

    per_seq_err = torch.cat(per_seq_chunks, dim=0).numpy()   # [N, T]
    mean = per_seq_err.mean(axis=0)
    std = per_seq_err.std(axis=0)

    details = {
        "per_seq_err": per_seq_err,
        "per_seq_err_copy_last": torch.cat(baseline_chunks, dim=0).numpy(),
        **strip,
    }
    if age_chunks:
        details["boundary_age"] = torch.cat(age_chunks, dim=0).numpy()
    return mean, std, details


def eval_velocity_generalization(model, device, args):
    """
    Returns
    -------
    vx_vals : np.ndarray [K]   (sorted unique velocities on x-axis)
    vy_vals : np.ndarray [K]   (same on y-axis)
    err_mat : np.ndarray [K,K] (mean MSE at (vy, vx))
    """
    vx_vals = np.arange(args.gen_vel_min, args.gen_vel_max + 1, args.gen_vel_step)
    vy_vals = np.arange(args.gen_vel_min, args.gen_vel_max + 1, args.gen_vel_step)
    err_mat = np.zeros((len(vy_vals), len(vx_vals)))

    crit = torch.nn.MSELoss(reduction='none')
    model.eval()

    vel_pbar = tqdm(total=len(vy_vals)*len(vx_vals), desc="Evaluating Velocity Generalization", leave=False)
    for iy, vy in enumerate(vy_vals):
        for ix, vx in enumerate(vx_vals):
            vel_pbar.set_postfix({"vx": vx, "vy": vy})
            dataset = FixedVelocityMovingMNIST(
                vx=vx, vy=vy,
                root=args.root,
                train=False,
                seq_len=args.seq_len,
                image_size=args.image_size,
                num_digits=2,
                random=False)

            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

            mse_sum, n_seen = 0.0, 0
            with torch.no_grad():
                batch_pbar = tqdm(loader, desc=f"vx={vx}, vy={vy}", leave=False)
                for seq, _ in batch_pbar:
                    seq = seq.to(device)
                    inp, tgt = seq[:, :args.input_frames], seq[:, args.input_frames:]
                    pred, _, _ = _run_model(model, inp, tgt.size(1), None, False, False)
                    mse = crit(pred, tgt).mean(dim=(2, 3, 4))  # [B, T]
                    batch_mse = mse.mean().item()
                    mse_sum += batch_mse * mse.size(0)
                    n_seen += mse.size(0)
                    batch_pbar.set_postfix({"mse": f"{batch_mse:.4f}"})
                    # Log sequence predictions for the first batch of this velocity pair
                    if batch_pbar.n == 0:
                        log_sequence_predictions_new(inp, tgt, pred, split_name=f"vel_gen_vx{vx}_vy{vy}", num_samples=10, device=device)
            err_mat[iy, ix] = mse_sum / n_seen
            vel_pbar.update(1)
    vel_pbar.close()

    return vx_vals, vy_vals, err_mat
