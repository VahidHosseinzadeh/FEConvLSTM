import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from moving_mnist_dataset import FixedVelocityMovingMNIST
from visualization import log_sequence_predictions, log_sequence_predictions_new, log_state_evolution
from velocity_metrics import VelocityMetrics, motions_to_true_vel




def train_epoch_velocity_check(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    input_frames,
    teacher_forcing_ratio,
    grad_clip=None,
):

    model.train()
    velocity_metrics = VelocityMetrics()
    velocity_metrics.reset()

    running_loss = 0.0

    pbar = tqdm(dataloader, desc="Training", leave=False)

    for i, (seq, _, gt_motion) in enumerate(pbar):

        seq = seq.to(device)
        gt_motion = gt_motion.to(device)

        input_seq = seq[:, :input_frames]
        target_seq = seq[:, input_frames:]
        pred_len = target_seq.size(1)

        optimizer.zero_grad()

        if i == 0:
            output_seq, pred_motion, states = model(
                input_seq,
                pred_len=pred_len,
                teacher_forcing_ratio=teacher_forcing_ratio,
                target_seq=target_seq,
                return_velocity=True,
                return_states=True,
            )
        else:
            output_seq, pred_motion = model(
                input_seq,
                pred_len=pred_len,
                teacher_forcing_ratio=teacher_forcing_ratio,
                target_seq=target_seq,
                return_velocity=True,
            )

        velocity_metrics.update(pred_motion, gt_motion)

        loss = criterion(output_seq, target_seq)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_loss = loss.item()
        running_loss += batch_loss * seq.size(0)

        pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

        if i == 0:
            log_sequence_predictions(
                input_seq,
                target_seq,
                output_seq,
                split_name="train",
            )
            log_state_evolution(
                states["h"],
                frames=states["frames"],
                gt_frames=torch.cat([input_seq, target_seq], dim=1),
                pred_vel=pred_motion,
                gt_vel=motions_to_true_vel(gt_motion, pred_motion.size(1)),
                split_name="train",
                input_frames=input_frames,
            )

    velocity_metrics.report("Training Velocity")

    return running_loss / len(dataloader.dataset)



def eval_epoch_velocity_check(
    model,
    dataloader,
    criterion,
    device,
    input_frames,
    epoch,
    split_name
):

    model.eval()
    velocity_metrics = VelocityMetrics()
    velocity_metrics.reset()

    running_loss = 0.0

    with torch.no_grad():

        pbar = tqdm(dataloader, desc="Evaluating", leave=False)

        for i, (seq, _, gt_motion) in enumerate(pbar):

            seq = seq.to(device)
            gt_motion = gt_motion.to(device)

            input_seq = seq[:, :input_frames]
            target_seq = seq[:, input_frames:]
            pred_len = target_seq.size(1)

            if i == 0:
                output_seq, pred_motion, states = model(
                    input_seq,
                    pred_len=pred_len,
                    teacher_forcing_ratio=0.0,
                    target_seq=target_seq,
                    return_velocity=True,
                    return_states=True,
                )
            else:
                output_seq, pred_motion = model(
                    input_seq,
                    pred_len=pred_len,
                    teacher_forcing_ratio=0.0,
                    target_seq=target_seq,
                    return_velocity=True,
                )

            velocity_metrics.update(pred_motion, gt_motion)

            loss = criterion(output_seq, target_seq)

            batch_loss = loss.item()
            running_loss += batch_loss * seq.size(0)

            pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

            if i == 0:
                log_sequence_predictions(
                    input_seq,
                    target_seq,
                    output_seq,
                    split_name=split_name,
                )
                log_state_evolution(
                    states["h"],
                    frames=states["frames"],
                    gt_frames=torch.cat([input_seq, target_seq], dim=1),
                    pred_vel=pred_motion,
                    gt_vel=motions_to_true_vel(gt_motion, pred_motion.size(1)),
                    split_name=split_name,
                    input_frames=input_frames,
                )

    velocity_metrics.report(f"{split_name} Velocity")

    return running_loss / len(dataloader.dataset)





def eval_len_generalization_velocity_check(
    model,
    dataloader,
    device,
    input_frames,
    subsample_t=1,
):

    model.eval()
    velocity_metrics = VelocityMetrics()
    velocity_metrics.reset()

    first_pass = True

    with torch.no_grad():

        n_sequences = 0

        pbar = tqdm(
            dataloader,
            desc="Evaluating Length Generalization",
            leave=False,
        )

        for seq, _, gt_motion in pbar:

            seq = seq.to(device)
            gt_motion = gt_motion.to(device)

            inp = seq[:, :input_frames]
            tgt = seq[:, input_frames:]

            T = tgt.size(1)

            if first_pass:
                pred, pred_motion, states = model(
                    inp,
                    pred_len=T,
                    teacher_forcing_ratio=0.0,
                    return_velocity=True,
                    return_states=True,
                )
            else:
                pred, pred_motion = model(
                    inp,
                    pred_len=T,
                    teacher_forcing_ratio=0.0,
                    return_velocity=True,
                )

            velocity_metrics.update(pred_motion, gt_motion)

            per_ex_t = ((pred - tgt) ** 2).mean(dim=(2, 3, 4))

            if first_pass:

                sum_err = per_ex_t.sum(dim=0)
                sum_err2 = (per_ex_t ** 2).sum(dim=0)

                first_pass = False

                log_sequence_predictions_new(
                    inp,
                    tgt,
                    pred,
                    split_name="len_gen",
                    num_samples=10,
                    device=device,
                    subsample_t=subsample_t,
                )
                log_state_evolution(
                    states["h"],
                    frames=states["frames"],
                    gt_frames=torch.cat([inp, tgt], dim=1),
                    pred_vel=pred_motion,
                    gt_vel=motions_to_true_vel(gt_motion, pred_motion.size(1)),
                    split_name="len_gen",
                    subsample_t=subsample_t,
                    input_frames=input_frames,
                )

            else:

                sum_err += per_ex_t.sum(dim=0)
                sum_err2 += (per_ex_t ** 2).sum(dim=0)

            n_sequences += per_ex_t.size(0)

            pbar.set_postfix({"loss": per_ex_t.mean().item()})

    velocity_metrics.report("Length Generalization Velocity")

    mean = sum_err / n_sequences
    var = sum_err2 / n_sequences - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=0.0))

    return mean.cpu().numpy(), std.cpu().numpy()