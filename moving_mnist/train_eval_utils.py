import torch
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
               want_velocity, want_states):
    """
    Single call site for every model type.

    MELSTM always receives target_seq when available — NOT teacher forcing:
    its training protocol tracks decoder velocities against the true next
    frame (v = track(h, target_seq[:, t])). Without it the model silently
    runs in inference mode (last encoder velocity frozen for the whole
    rollout — garbage early in training) and learning stalls.
    In eval() mode we pass track_decoder_velocity=False instead: honest
    deployable inference (frozen velocity), while target_seq stays unused.

    Returns (output_seq, pred_motion_or_None, states_or_None).
    """
    if isinstance(model, Seq2SeqMEConvLSTM):
        result = model(
            input_seq,
            pred_len=pred_len,
            target_seq=target_seq,
            track_decoder_velocity=model.training,
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


def train_epoch(model, dataloader, optimizer, criterion, device, input_frames, grad_clip=None):
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


def eval_epoch(model, dataloader, criterion, device, input_frames, epoch, split_name):
    model.eval()
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
                model, input_seq, pred_len, target_seq, want_velocity, want_states
            )
            loss = criterion(output_seq, target_seq)
            batch_loss = loss.item()
            running_loss += batch_loss * seq.size(0)
            pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

            if want_velocity:
                # eval runs with track_decoder_velocity=False, so these are
                # encoder velocities only (decoder rolls out frozen).
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


def eval_len_generalization(model, dataloader, device, input_frames, subsample_t=1):
    """
    Returns:
        mean_err  – numpy array [T]  (MSE at each future step, averaged over test set)
        std_err   – numpy array [T]  (sample‑wise std at each step)
    """
    model.eval()
    velocity_metrics = VelocityMetrics()
    has_velocity_data = False
    first_pass = True

    with torch.no_grad():
        n_sequences = 0
        pbar = tqdm(dataloader, desc="Evaluating Length Generalization", leave=False)
        for batch in pbar:
            seq, gt_motion = _unpack_batch(batch, device)
            inp, tgt = seq[:, :input_frames], seq[:, input_frames:]
            T = tgt.size(1)

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
            if first_pass:
                sum_err = per_ex_t.sum(dim=0)           # [T]
                sum_err2 = (per_ex_t ** 2).sum(dim=0)   # [T]
                first_pass = False

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
            else:
                sum_err += per_ex_t.sum(dim=0)
                sum_err2 += (per_ex_t ** 2).sum(dim=0)

            n_sequences += per_ex_t.size(0)
            pbar.set_postfix({"loss": per_ex_t.mean().item()})

    if has_velocity_data:
        velocity_metrics.report("Length Generalization Velocity")
        log_velocity_report(velocity_metrics.summary(), split_name="len_gen")

    mean = sum_err / n_sequences
    var = sum_err2 / n_sequences - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=0.0))
    return mean.cpu().numpy(), std.cpu().numpy()


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
