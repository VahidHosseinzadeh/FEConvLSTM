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
from fourier_loss import FourierShapePhaseLoss
from velocity_model_based_MEConvLSTM_model import Seq2SeqMEConvLSTM
from channel_based_FEConvLSTM_model import Seq2SeqFEConvLSTM


# Parameter-name prefixes owned by the velocity process model. Everything else
# -- cell.* (the recurrent encoder) and decoder.* -- is the renderer. The
# correlators (phase_corr_*) hold no parameters at all: they read a velocity
# off an argmax, so they neither have weights to route gradient to nor pass
# gradient through. Verified, not assumed.
_VELOCITY_PARAM_PREFIXES = ("vel_dyn",)


def split_parameters(model):
    """(velocity-model params, encoder/decoder params) by name prefix.

    Used by --loss_routing split to send each half of the Fourier-decomposed
    pixel loss only to the parameters it is a statement about.
    """
    vel, enc = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (vel if name.startswith(_VELOCITY_PARAM_PREFIXES) else enc).append(p)
    return vel, enc


def _accumulate_grads(params, grads):
    """.grad += g, tolerating the None that zero_grad(set_to_none=True) leaves
    and the None autograd returns for a parameter outside the graph."""
    for p, g in zip(params, grads):
        if g is None:
            continue
        p.grad = g if p.grad is None else p.grad + g


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

    if name == "melstm":
        return Seq2SeqMEConvLSTM(
            input_channels=1, hidden_channels=hidden, kernel_size=kernel,
            n_slots=get("num_vel_modes", 2), slot_reduce="max",
            decoder_layers=dec_layers, decoder_channels=dec_channels,
            # Absent from older config dicts -> the defaults reproduce exactly
            # the architecture those runs were trained with.
            use_velocity_dynamics=get("use_velocity_dynamics", False),
            vel_dyn_state_dim=get("vel_dyn_state_dim", 32),
            vel_dyn_use_h=get("vel_dyn_use_h", False),
            vel_dyn_gain=get("vel_dyn_gain", "fixed"),
            vel_dyn_openloop_k=get("vel_dyn_openloop_k", 0),
            vel_dyn_arch=get("vel_dyn_arch", "gru"),
            vel_dyn_layers=get("vel_dyn_layers", 1),
            vel_dyn_decoder_supervision=get("vel_dyn_decoder_supervision", "none"),
            vel_dyn_v_max=get("vel_dyn_v_max", None),
            track_corr_alpha=get("track_corr_alpha", None))

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
               predict_decoder_velocity=False, want_dyn_loss=False,
               decoder_sampling_p=0.0):
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

    predict_decoder_velocity rolls the decoder velocity forward with the
    velocity dynamics head instead of freezing it — the third decoder mode,
    deployable like "frozen" but able to follow a still-changing velocity.
    Ignored unless the model was built with --use_velocity_dynamics, and
    superseded by the tracked (oracle) protocol when that is active.

    want_dyn_loss asks the model for the one-step-ahead velocity dynamics
    loss so the caller can add it to the training objective.

    Returns (output_seq, pred_motion_or_None, states_or_None, dyn_loss_or_None).
    """
    if isinstance(model, Seq2SeqMEConvLSTM):
        if track_decoder_velocity is None:
            track_decoder_velocity = model.training
        result = model(
            input_seq,
            pred_len=pred_len,
            target_seq=target_seq,
            track_decoder_velocity=track_decoder_velocity,
            predict_decoder_velocity=predict_decoder_velocity,
            decoder_sampling_p=decoder_sampling_p,
            return_velocity=want_velocity,
            return_dyn_loss=want_dyn_loss,
            return_states=want_states,
        )
        if not (want_velocity or want_states or want_dyn_loss):
            return result, None, None, None
        # Model's return order: outputs, [velocities], [dyn_loss], [states]
        result = list(result if isinstance(result, tuple) else [result])
        output_seq = result.pop(0)
        pred_motion = result.pop(0) if want_velocity else None
        dyn_loss = result.pop(0) if want_dyn_loss else None
        states = result.pop(0) if want_states else None
        return output_seq, pred_motion, states, dyn_loss

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
    want_velocity : True only for Seq2SeqMEConvLSTM, whenever gt_motion is
        available (--check_velocity_predictor). Returns an explicit tracked
        (vx, vy) per slot to score against gt_motion.
    want_states   : gates the log_state_evolution h_states plot, and is
        deliberately NOT a single shared flag across model types:
          MEConvLSTM  -- tied to want_velocity/--check_velocity_predictor.
              This plot exists to debug the velocity tracker, so it travels
              with the flag that turns the tracker's own debug output on.
          FEConvLSTM  -- independent --show_h_state flag. FELSTM has no
              velocity predictor to debug -- it shows the raw per-(vx,vy)
              candidate slots instead, which isn't conceptually part of
              "the velocity predictor".
    Callers still AND want_states with their own "first batch only" gate.
    """
    is_melstm = isinstance(model, Seq2SeqMEConvLSTM)
    is_felstm = isinstance(model, Seq2SeqFEConvLSTM)
    want_velocity = is_melstm and gt_motion is not None
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

    @torch.no_grad()
    def _val_loss(self, model):
        was_training = model.training
        model.eval()
        per_seq = []
        for s in range(0, self.data.size(0), self.batch_size):
            seq = self.data[s:s + self.batch_size].to(self.device)
            inp = seq[:, :self.input_frames]
            tgt = seq[:, self.input_frames:]
            pred, _, _, _ = _run_model(model, inp, tgt.size(1), None, False, False)
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
                curve_recorder=None, show_h_state=False, vel_dyn_loss_weight=0.0,
                decoder_sampling_p=0.0, loss_routing='shared'):
    """
    vel_dyn_loss_weight : weight on the velocity dynamics head's one-step-ahead
        loss (--vel_dyn_loss_weight). 0.0 (default) leaves both the objective
        and the model call identical to before this feature existed; with a
        head present but zero weight the head simply never trains.

    loss_routing : 'shared' (default) backpropagates the whole pixel loss into
        every parameter, as always. 'split' sends the Fourier shape term (plus
        L1) only to the encoder/decoder and the location term only to the
        velocity head -- same objective value, two restricted backward passes
        instead of one. Requires a criterion exposing routed_terms(), i.e.
        --fourier_loss.
    """
    model.train()
    running_loss = 0.0
    running_dyn_loss = 0.0
    running_shape = 0.0
    running_location = 0.0
    velocity_metrics = VelocityMetrics()
    has_velocity_data = False

    want_dyn_loss = (vel_dyn_loss_weight > 0.0
                     and getattr(model, "vel_dyn", None) is not None)

    routed = loss_routing == 'split'
    if routed:
        if not hasattr(criterion, "routed_terms"):
            raise ValueError("--loss_routing split needs the Fourier-decomposed "
                             "criterion; pass --fourier_loss.")
        vel_params, enc_params = split_parameters(model)
        if not vel_params:
            raise ValueError("--loss_routing split found no velocity-model "
                             "parameters to route the location term to; the "
                             "head is off (pass --use_velocity_dynamics).")

    pbar = tqdm(dataloader, desc="Training", leave=False, disable=True)
    for i, batch in enumerate(pbar):
        seq, gt_motion = _unpack_batch(batch, device)   # (B, seq_len, C, H, W)
        input_seq = seq[:, :input_frames]
        target_seq = seq[:, input_frames:]
        pred_len = target_seq.size(1)

        want_velocity, want_states = _velocity_flags(model, gt_motion, show_h_state)
        want_states = want_states and i == 0

        optimizer.zero_grad()
        output_seq, pred_motion, states, dyn_loss = _run_model(
            model, input_seq, pred_len, target_seq, want_velocity, want_states,
            want_dyn_loss=want_dyn_loss, decoder_sampling_p=decoder_sampling_p,
        )
        # Image loss and total objective are kept separate on purpose: the
        # reported/curve-recorded train_loss must stay the pixel loss, or
        # turning the dynamics head on would shift the training curve for
        # reasons that have nothing to do with prediction quality and make it
        # incomparable with every previous run.
        if routed:
            # Same objective, restricted gradients. Two autograd.grad calls
            # rather than one .backward(): the halves have to be differentiated
            # separately, because once they are summed there is no way to tell
            # which parameter's gradient came from which term. Costs one extra
            # backward traversal per batch.
            enc_term, vel_term, _ = criterion.routed_terms(output_seq, target_seq)
            loss = enc_term + vel_term          # == criterion(...), reported as before
            _accumulate_grads(enc_params, torch.autograd.grad(
                enc_term, enc_params, retain_graph=True, allow_unused=True))
            _accumulate_grads(vel_params, torch.autograd.grad(
                vel_term, vel_params,
                retain_graph=dyn_loss is not None, allow_unused=True))
            # The head's own supervision is deliberately NOT routed: it is a
            # velocity-space target, and letting it shape h through the encoder
            # is the point of --vel_dyn_use_h.
            if dyn_loss is not None:
                running_dyn_loss += dyn_loss.item() * seq.size(0)
                (vel_dyn_loss_weight * dyn_loss).backward()
        else:
            loss = criterion(output_seq, target_seq)
            total_loss = loss
            if dyn_loss is not None:
                running_dyn_loss += dyn_loss.item() * seq.size(0)
                total_loss = loss + vel_dyn_loss_weight * dyn_loss
            total_loss.backward()

        with torch.no_grad():
            sh, lo = FourierShapePhaseLoss.decompose(output_seq.detach(), target_seq)
        running_shape += sh.item() * seq.size(0)
        running_location += lo.item() * seq.size(0)

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
                    motion=gt_motion,
                    v_list=model.cell.v_list if isinstance(model, Seq2SeqFEConvLSTM) else None,
                )

    if has_velocity_data:
        velocity_metrics.report("Training Velocity")
        log_velocity_report(velocity_metrics.summary(), split_name="train",
                            input_frames=input_frames)

    n = len(dataloader.dataset)
    # Shape/location are logged unweighted and in pixel-MSE units, whatever the
    # criterion is, so they sum to the plain MSE and stay comparable across
    # runs with different --shape_weight (or with --fourier_loss off entirely).
    wandb.log({"train_shape_loss": running_shape / n,
               "train_location_loss": running_location / n})
    # (pixel loss, dynamics loss). The second is None when the head is absent
    # or unweighted, so the caller can tell "not measured" from "measured 0".
    return running_loss / n, (running_dyn_loss / n if want_dyn_loss else None)


def eval_epoch(model, dataloader, criterion, device, input_frames, epoch, split_name,
               decoder_velocity_mode="frozen", show_h_state=False):
    """
    decoder_velocity_mode : "frozen" (default) — honest deployable inference,
        the last encoder velocity rolls the whole horizon; velocity metrics
        then cover encoder steps only. "tracked" — oracle protocol: MELSTM's
        decoder velocities are tracked against the true next frames (upper
        bound; the gap to "frozen" isolates velocity-estimation error from
        rendering error). "predicted" — the velocity dynamics head rolls the
        velocity forward with no measurement: deployable like "frozen", but
        able to keep following a velocity that is still changing after the
        context ends. Requires a model built with --use_velocity_dynamics;
        without one it silently degrades to "frozen", which is exactly what
        the head predicts at initialization anyway.
    """
    model.eval()
    track = decoder_velocity_mode == "tracked"
    predict = decoder_velocity_mode == "predicted"
    running_loss = 0.0
    running_trivial = 0.0
    running_shape = 0.0
    running_location = 0.0
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
                predict_decoder_velocity=predict,
            )
            loss = criterion(output_seq, target_seq)
            batch_loss = loss.item()
            running_loss += batch_loss * seq.size(0)
            # The all-zeros predictor, scored by the SAME criterion. On sparse
            # bright-on-black frames this is a surprisingly strong baseline --
            # a digit displaced by half its own width already costs as much as
            # a blank frame -- so a model can sit ABOVE it for an entire run
            # without that being visible in the loss curve alone. Logging it
            # makes "are we beating nothing at all?" impossible to miss.
            running_trivial += criterion(torch.zeros_like(target_seq),
                                         target_seq).item() * seq.size(0)
            # Logged for every split and every criterion, so the two halves of
            # the error are trackable side by side: shape is what the renderer
            # owns, location is what a velocity owns.
            sh, lo = FourierShapePhaseLoss.decompose(output_seq, target_seq)
            running_shape += sh.item() * seq.size(0)
            running_location += lo.item() * seq.size(0)
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
        log_velocity_report(velocity_metrics.summary(), split_name=split_name, epoch=epoch,
                            input_frames=input_frames)

    n = len(dataloader.dataset)
    trivial = running_trivial / n
    loss = running_loss / n
    wandb.log({f"{split_name}_trivial_baseline": trivial,
               f"{split_name}_vs_trivial": trivial - loss,
               f"{split_name}_shape_loss": running_shape / n,
               f"{split_name}_location_loss": running_location / n})
    if loss > trivial:
        print(f"  WARNING: {split_name} loss {loss:.4f} is WORSE than predicting all zeros "
              f"({trivial:.4f}). The model is being penalised for drawing anything at all -- "
              f"see fourier_loss.py.")
    return loss


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
        log_velocity_report(velocity_metrics.summary(), split_name="len_gen",
                            input_frames=input_frames)

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
