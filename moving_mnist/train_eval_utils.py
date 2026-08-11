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
from velocity_model_based_MEConvRNN_model import Seq2SeqMEConvRNN
from channel_based_FEConvLSTM_model import Seq2SeqFEConvLSTM

# The two motion-estimating models. Seq2SeqMEConvRNN is a drop-in twin of
# Seq2SeqMEConvLSTM -- same constructor signature, same forward signature and
# return protocol, only the recurrence differs (h only, no cell state) -- so
# every "is this a slot/velocity model?" branch below applies to both.
ME_MODELS = (Seq2SeqMEConvLSTM, Seq2SeqMEConvRNN)


def build_model(cfg):
    """Construct the model a run config describes.

    Accepts either the argparse Namespace used during training or the config
    dict stored in history_<model>_<runid>.json, so offline evaluation
    (motion_generalization.py) rebuilds exactly the architecture that was
    trained instead of re-specifying the hyperparameters by hand.
    """
    get = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))

    name = get("model")
    hidden = get("hidden_size")
    kernel = get("kernel_size", 3)
    dec_layers = get("decoder_conv_layers", 1)
    # None/absent means "decoder width follows the cell width"
    dec_channels = get("decoder_hidden_size") or hidden

    if name in ("felstm", "lstm"):
        v_range = get("v_range", 0) if name == "felstm" else 0
        if name == "lstm":
            assert get("v_range", 0) == 0, "v_range must be 0 for lstm"
        return Seq2SeqFEConvLSTM(
            input_channels=1, hidden_channels=hidden, kernel_size=kernel,
            v_range=v_range, pool_type="max",
            decoder_conv_layers=dec_layers, decoder_channels=dec_channels)

    if name in ("melstm", "mernn"):
        # mernn is the vanilla-RNN ablation of melstm: identical slot/velocity
        # machinery, gateless recurrent cell. Same hyperparameters apply.
        cls = Seq2SeqMEConvLSTM if name == "melstm" else Seq2SeqMEConvRNN
        return cls(
            input_channels=1, hidden_channels=hidden, kernel_size=kernel,
            n_slots=get("num_vel_modes", 2), slot_reduce="max",
            decoder_layers=dec_layers, decoder_channels=dec_channels,
            # Baked in so offline evaluation (motion_generalization.py), which
            # calls forward() without the per-call override, reproduces the
            # velocity protocol the run was trained under. Annealed runs store
            # the STARTING value, so pass the final one explicitly there.
            bootstrap_until=get("bootstrap_until", 1))

    raise ValueError(f"unknown model {name!r}")


def _unpack_batch(batch, device):
    """
    Datasets yield (seq, label) or, with return_motion=True, (seq, label,
    motion). Velocity reporting below activates iff motion is present.
    """
    seq = batch[0].to(device)
    gt_motion = batch[2].to(device) if len(batch) > 2 else None
    return seq, gt_motion


def _run_model(model, input_seq, pred_len, target_seq,
               want_velocity, want_states, track_decoder_velocity=None,
               bootstrap_until=None, want_track_loss=False,
               track_loss_temperature=1.0):
    """
    Single call site for every model type.

    MELSTM/MERNN always receive target_seq when available — NOT teacher forcing:
    its training protocol tracks decoder velocities against the true next
    frame (v = track(h, target_seq[:, t])). Without it the model silently
    runs in inference mode (last encoder velocity frozen for the whole
    rollout — garbage early in training) and learning stalls.

    track_decoder_velocity=None (default) picks the protocol by phase:
    training -> tracked (required, see above); eval -> frozen (honest
    deployable inference). Pass True explicitly for the oracle eval
    (GT-tracked decoder velocities), e.g. via --eval_velocity_mode.

    bootstrap_until / want_track_loss are ME_MODELS-only (see
    Seq2SeqMEConvLSTM.forward): the first controls how many encoder steps read
    velocity off the raw frame pair instead of self-tracking h, the second
    asks for the distillation cross-entropy between the two.

    Returns (output_seq, pred_motion_or_None, states_or_None, track_loss_or_None).
    """
    if isinstance(model, ME_MODELS):
        if track_decoder_velocity is None:
            track_decoder_velocity = model.training
        result = model(
            input_seq,
            pred_len=pred_len,
            target_seq=target_seq,
            track_decoder_velocity=track_decoder_velocity,
            bootstrap_until=bootstrap_until,
            track_loss_temperature=track_loss_temperature,
            return_velocity=want_velocity,
            return_states=want_states,
            return_track_loss=want_track_loss,
        )
        if not (want_velocity or want_states or want_track_loss):
            return result, None, None, None
        result = list(result if isinstance(result, tuple) else [result])
        output_seq = result.pop(0)
        pred_motion = result.pop(0) if want_velocity else None
        states = result.pop(0) if want_states else None
        track_loss = result.pop(0) if want_track_loss else None
        return output_seq, pred_motion, states, track_loss

    if isinstance(model, Seq2SeqFEConvLSTM):
        # FEConvLSTM has no explicit velocity readout (want_velocity is
        # always False for it, see _velocity_flags) -- it only exposes
        # h_states, one channel-mean map per (vx, vy) candidate slot.
        if not want_states:
            return model(input_seq, pred_len=pred_len), None, None, None
        output_seq, states = model(input_seq, pred_len=pred_len, return_states=True)
        return output_seq, None, states, None

    return model(input_seq, pred_len=pred_len), None, None, None


def _velocity_flags(model, gt_motion, show_h_state=False):
    """
    want_velocity : True only for the ME models (ConvLSTM / ConvRNN), whenever
        gt_motion is available (--check_velocity_predictor). Returns an
        explicit tracked (vx, vy) per slot to score against gt_motion.
    want_states   : gates the log_state_evolution h_states plot, and is
        deliberately NOT a single shared flag across model types:
          ME models   -- tied to want_velocity/--check_velocity_predictor.
              This plot exists to debug the velocity tracker, so it travels
              with the flag that turns the tracker's own debug output on.
          FEConvLSTM  -- independent --show_h_state flag. FELSTM has no
              velocity predictor to debug -- it shows the raw per-(vx,vy)
              candidate slots instead, which isn't conceptually part of
              "the velocity predictor".
    Callers still AND want_states with their own "first batch only" gate.
    """
    is_me = isinstance(model, ME_MODELS)
    is_felstm = isinstance(model, Seq2SeqFEConvLSTM)
    want_velocity = is_me and gt_motion is not None
    want_states = want_velocity or (is_felstm and show_h_state and gt_motion is not None)
    return want_velocity, want_states


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
        # Velocity protocol this curve is measured under. train_epoch sets it
        # each epoch so the curve tracks the SAME protocol training is using;
        # left at None it would silently pin to the constructor value and keep
        # reporting the un-annealed model, which is not what this curve is for.
        self.bootstrap_until = None

    @torch.no_grad()
    def _val_loss(self, model):
        was_training = model.training
        model.eval()
        per_seq = []
        for s in range(0, self.data.size(0), self.batch_size):
            seq = self.data[s:s + self.batch_size].to(self.device)
            inp = seq[:, :self.input_frames]
            tgt = seq[:, self.input_frames:]
            pred, _, _, _ = _run_model(model, inp, tgt.size(1), None, False, False,
                                       bootstrap_until=self.bootstrap_until)
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

    def state_dict(self):
        """Full state for crash-resume, including the materialized val data
        (regenerating it on resume would silently switch the measured set)."""
        return {
            "data": self.data,
            "step": self.step,
            "train_steps": self.train_steps,
            "train_losses": self.train_losses,
            "val_steps": self.val_steps,
            "val_means": self.val_means,
            "val_stds": self.val_stds,
        }

    def load_state_dict(self, state):
        self.data = state["data"]
        self.step = state["step"]
        self.train_steps = list(state["train_steps"])
        self.train_losses = list(state["train_losses"])
        self.val_steps = list(state["val_steps"])
        self.val_means = list(state["val_means"])
        self.val_stds = list(state["val_stds"])


def train_epoch(model, dataloader, optimizer, criterion, device, input_frames, grad_clip=None,
                curve_recorder=None, show_h_state=False,
                bootstrap_until=None, track_loss_weight=0.0,
                track_loss_temperature=1.0):
    """
    bootstrap_until / track_loss_weight : MELSTM/MERNN velocity-distillation
        controls. The tracking cross-entropy is computed whenever it can be
        (it is the "is h ready to take over" readout) and only ADDED to the
        loss when track_loss_weight > 0 -- so weight 0 gives a pure monitor
        with no gradient, which is the honest default for old configs.
    """
    model.train()
    running_loss = 0.0
    running_track = 0.0
    n_track = 0
    velocity_metrics = VelocityMetrics()
    has_velocity_data = False
    is_me = isinstance(model, ME_MODELS)
    if curve_recorder is not None:
        curve_recorder.bootstrap_until = bootstrap_until

    pbar = tqdm(dataloader, desc="Training", leave=False, disable=True)
    for i, batch in enumerate(pbar):
        seq, gt_motion = _unpack_batch(batch, device)   # (B, seq_len, C, H, W)
        input_seq = seq[:, :input_frames]
        target_seq = seq[:, input_frames:]
        pred_len = target_seq.size(1)

        want_velocity, want_states = _velocity_flags(model, gt_motion, show_h_state)
        want_states = want_states and i == 0

        optimizer.zero_grad()
        output_seq, pred_motion, states, track_loss = _run_model(
            model, input_seq, pred_len, target_seq, want_velocity, want_states,
            bootstrap_until=bootstrap_until, want_track_loss=is_me,
            track_loss_temperature=track_loss_temperature,
        )
        loss = criterion(output_seq, target_seq)
        # Report the RECONSTRUCTION loss only, captured before the auxiliary
        # term is folded in -- otherwise train_loss stops being comparable to
        # val_loss the moment the distillation is switched on.
        recon_loss = loss.item()
        if track_loss is not None:
            running_track += track_loss.item()
            n_track += 1
            if track_loss_weight > 0:
                loss = loss + track_loss_weight * track_loss
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        if want_velocity:
            velocity_metrics.update(pred_motion, gt_motion)
            has_velocity_data = True

        batch_loss = recon_loss
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
                    motion=gt_motion,
                    v_list=model.cell.v_list if isinstance(model, Seq2SeqFEConvLSTM) else None,
                )

    if has_velocity_data:
        velocity_metrics.report("Training Velocity")
        log_velocity_report(velocity_metrics.summary(), split_name="train")

    if n_track:
        mean_ce = running_track / n_track
        # Falling = h is becoming a usable tracking template; this is the
        # signal for when bootstrap_until can safely be annealed down.
        print(f"  Tracking CE (h vs frame-pair teacher): {mean_ce:.4f}"
              f"   [bootstrap_until={bootstrap_until}, "
              f"weight={track_loss_weight}]")
        if wandb.run is not None:
            wandb.log({"train_track_ce": mean_ce,
                       "bootstrap_until": (bootstrap_until
                                           if bootstrap_until is not None
                                           else model.bootstrap_until)})

    return running_loss / len(dataloader.dataset)


def eval_epoch(model, dataloader, criterion, device, input_frames, epoch, split_name,
               decoder_velocity_mode="frozen", show_h_state=False,
               bootstrap_until=None):
    """
    decoder_velocity_mode : "frozen" (default) — honest deployable inference,
        the last encoder velocity rolls the whole horizon; velocity metrics
        then cover encoder steps only. "tracked" — oracle protocol: the ME
        models' decoder velocities are tracked against the true next frames (upper
        bound; the gap to "frozen" isolates velocity-estimation error from
        rendering error).
    """
    model.eval()
    track = decoder_velocity_mode == "tracked"
    running_loss = 0.0
    velocity_metrics = VelocityMetrics()
    has_velocity_data = False

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False, disable=True)
        for i, batch in enumerate(pbar):
            seq, gt_motion = _unpack_batch(batch, device)
            input_seq = seq[:, :input_frames]
            target_seq = seq[:, input_frames:]
            pred_len = target_seq.size(1)

            want_velocity, want_states = _velocity_flags(model, gt_motion, show_h_state)
            want_states = want_states and i == 0

            output_seq, pred_motion, states, _ = _run_model(
                model, input_seq, pred_len, target_seq, want_velocity, want_states,
                track_decoder_velocity=True if track else None,
                bootstrap_until=bootstrap_until,
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
                        motion=gt_motion,
                        v_list=model.cell.v_list if isinstance(model, Seq2SeqFEConvLSTM) else None,
                    )

    if has_velocity_data:
        velocity_metrics.report(f"{split_name} Velocity")
        log_velocity_report(velocity_metrics.summary(), split_name=split_name, epoch=epoch)

    return running_loss / len(dataloader.dataset)


def eval_len_generalization(model, dataloader, device, input_frames, subsample_t=1,
                            n_strip_sequences=3, show_h_state=False):
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
        pbar = tqdm(dataloader, desc="Evaluating Length Generalization", leave=False, disable=True)
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

            want_velocity, want_states = _velocity_flags(model, gt_motion, show_h_state)
            want_states = want_states and first_pass

            # target_seq=None: length generalization is pure rollout.
            pred, pred_motion, states, _ = _run_model(
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
                        motion=gt_motion,
                        v_list=model.cell.v_list if isinstance(model, Seq2SeqFEConvLSTM) else None,
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

    vel_pbar = tqdm(total=len(vy_vals)*len(vx_vals), desc="Evaluating Velocity Generalization", leave=False, disable=True)
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
                batch_pbar = tqdm(loader, desc=f"vx={vx}, vy={vy}", leave=False, disable=True)
                for seq, _ in batch_pbar:
                    seq = seq.to(device)
                    inp, tgt = seq[:, :args.input_frames], seq[:, args.input_frames:]
                    pred, _, _, _ = _run_model(model, inp, tgt.size(1), None, False, False)
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
