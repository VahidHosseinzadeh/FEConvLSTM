"""
The KTH classifiers: the Motion-Only MNIST classifier, unchanged in architecture.

MotionVideoClassifier (motion_benchmarks/models/cf_classifier.py) is MotionDigitClassifier
(moving_mnist/motion_classification_model.py) with opt-in sub-pixel / suppressed phase correlation.
Built with the same arguments it has the same modules in the same initialisation order, so a
model seed gives the same initial weights whatever the velocity source.

    lstm   : one state, no transport (FEConvLSTM with v_range 0).
    felstm : (2R+1)^2 copies at fixed lattice velocities; R = 1 is the full 9-velocity bank V_1.
    melstm : K slots transported at estimated velocities. Sources: 'frame_pair' (the top-K phase-
             correlation peaks of the raw pair (x_{t-1}, x_t), each slot taking the candidate
             nearest its previous velocity), 'bootstrap' (the dominant peak, then the residual's
             peak: the minority motion pulled out from under the dominant one), 'tracked' (each
             slot correlates its own hidden state with the next frame), and the stochastic
             handover from frame-pair to tracked (--x_curriculum_epochs, training only).

Two KTH options on top:

slot_assign  how frame-pair candidates are handed to the slots at each step t >= 2:
             'nearest' (default, the classifier's rule): each slot takes the candidate nearest its
                 previous velocity. Exactly equivariant to a CONSTANT camera on the torus, but not
                 to a time-varying one: a camera change shifts every candidate by the same step
                 while the slots' previous velocities stay put, so the matching geometry changes
                 and slots swap contents. Measured on val (pc_analysis.py, K = 4): the background
                 keeps its slot on 97% of steps with a constant camera, 80% piecewise, 50% shake.
             'shift': nearest previous velocity after the best COMMON shift -- the assignment
                 minimises sum_k |cand_pi(k) - (v_prev_k + delta)|_1 over the K^2 shifts delta =
                 cand_j - v_prev_k. A camera change is exactly such a common shift, so this is
                 equivariant to any camera trajectory, like tracked slots are by construction.
             'anchored': slot 0 takes the top peak at every step, the rest nearest previous.
                 Keeps the dominant motion in one slot -- but on KTH's plain backgrounds the top
                 peak is often the person, so slot 0 then alternates.
             Measured on val (K = 4, search radius 5): the share of steps on which the
             background keeps its slot, static or constant camera / piecewise / shake:
                 nearest 0.967 / 0.796 / 0.499,  shift 0.627 / 0.621 / 0.627 (exactly
                 equivariant, but it reshuffles slots even with a static camera),
                 anchored 0.885 / 0.865 / 0.819.
             Which slots exist (the candidates, hence bg_hit and person_hit) is the same under
             all three; only the slots' identities differ.
integer warp MotionVideoClassifier turns on MEConvLSTMCell's padded (sub-pixel-exact) warp for
             every model; it is only needed when phase correlation is sub-pixel, so it is turned
             back off otherwise (whole-pixel shifts are exact either way, the padded path is
             ~28% slower).
"""
import torch
import torch.nn as nn

from motion_benchmarks import _repo  # noqa: F401  (puts moving_mnist/ on sys.path)
from motion_benchmarks.models.cf_classifier import MotionVideoClassifier  # noqa: E402
from motion_classification_model import MotionDigitClassifier  # noqa: E402

from .kth_dataset import KTH_ACTIONS


class KTHClassifier(MotionVideoClassifier):
    """
    static_slot   melstm: slot 0 is pinned to velocity (0, 0) -- an untransported state, i.e. a
                  ConvLSTM inside the MEConvLSTM. Slots 1..K-1 behave exactly like the slots of
                  the (K-1)-slot model: they take the top K-1 bootstrap / frame-pair candidates
                  (assigned by slot_assign among themselves) and, with h-tracking, track their own
                  states. So K = 5 with a static slot is the 4-slot model plus a resting slot
                  (tested, for frame-pair, tracking and the handover). No extra parameters.
    readout_steps average the head's logits over the states of the last N encoder steps instead
                  of reading h_T alone (1 = h_T, the original). Implemented with a hook on the
                  recurrent cell, so it works for every backbone and velocity source, the
                  handover included (MotionVideoClassifier's own readout path does not run the
                  handover). No extra parameters.
    velocity_readout
                  melstm: 'centered' also hands the head the slot velocities of the whole
                  sequence. A slot that moves with the person sees the person standing still,
                  so its state cannot carry the person's speed -- the velocity can.
                    * centered: at every step each slot's velocity minus the mean over the
                      (moving) slots. The camera shifts every slot alike, so this is camera-
                      invariant: what is left is motion relative to the scene.
                    * encoded per slot by a small MLP (2 -> 16 -> 16), max over slots (slot order
                      does not matter), then mean and max over the T-1 steps: 32 numbers.
                    * appended to the head's pooled conv features before its MLP.
                  The velocities carry no gradient (phase-correlation argmaxes); the encoder and
                  the widened first MLP layer train. ~4.5k extra parameters at the defaults.
    """

    VEL_DIM = 16

    def __init__(self, *args, slot_assign="nearest", static_slot=False, velocity_readout="none",
                 **kwargs):
        super().__init__(*args, **kwargs)
        if slot_assign not in ("nearest", "shift", "anchored"):
            raise ValueError(f"slot_assign {slot_assign!r}: expected nearest, shift or anchored")
        if static_slot and (self.model != "melstm" or self.n_velocities < 2):
            raise ValueError("static_slot needs melstm with >= 2 slots")
        if velocity_readout not in ("none", "centered"):
            raise ValueError(f"velocity_readout {velocity_readout!r}: expected none or centered")
        if velocity_readout != "none" and (self.model != "melstm"
                                           or self.n_velocities - int(static_slot) < 2):
            raise ValueError("velocity_readout needs melstm with >= 2 moving slots")
        self.slot_assign = slot_assign
        self.static_slot = bool(static_slot)
        self.velocity_readout = velocity_readout
        if velocity_readout != "none":
            # Built after everything else, so the rest of the initialisation is unchanged.
            D = self.VEL_DIM
            self.vel_encoder = nn.Sequential(nn.Linear(2, D), nn.LeakyReLU(0.01),
                                             nn.Linear(D, D))
            first = self.head.mlp[0]
            self.head.mlp[0] = nn.Linear(first.in_features + 2 * D, first.out_features)

    # ------------------------------------------------- readout over time / velocity readout
    def velocity_features(self, velocities):
        """(B, T-1, K, 2) slot velocities -> (B, 32), invariant to a common shift and slot order."""
        v = velocities[:, :, 1:] if self.static_slot else velocities
        c = v - v.mean(dim=2, keepdim=True)
        e = self.vel_encoder(c).amax(dim=2)                              # (B, T-1, D)
        return torch.cat([e.mean(dim=1), e.amax(dim=1)], dim=1)

    def _head(self, features, vel_feat=None):
        """ConvClassifierHead.forward, with the velocity features joining the pooled vector."""
        if vel_feat is None:
            return self.head(features)
        f = self.head.conv(features)
        pooled = torch.cat([f.mean(dim=(-2, -1)), f.amax(dim=(-2, -1)), vel_feat], dim=1)
        return self.head.mlp(pooled)

    def forward(self, seq, return_aux=False):
        if self.readout_steps <= 1 and self.velocity_readout == "none":
            return MotionDigitClassifier.forward(self, seq, return_aux=return_aux)
        states, handle = [], None
        if self.readout_steps > 1:
            handle = self.backbone.cell.register_forward_hook(
                lambda _module, _inputs, out: states.append(out[0]))
        try:
            h, velocities = self.encode(seq)[:2]
        finally:
            if handle is not None:
                handle.remove()
        vel_feat = (self.velocity_features(velocities)
                    if self.velocity_readout != "none" else None)
        logits, weights = [], None
        for h_t in (states[-self.readout_steps:] if self.readout_steps > 1 else [h]):
            features, weights = self.pool(h_t)
            logits.append(self._head(features, vel_feat))
        logits = torch.stack(logits, dim=0).mean(dim=0)
        if not return_aux:
            return logits
        return logits, {"velocities": velocities, "pool_weights": weights}

    def head_parameters(self):
        extra = list(self.vel_encoder.parameters()) if self.velocity_readout != "none" else []
        return super().head_parameters() + extra

    def submodule_report(self):
        rows, trained = super().submodule_report()
        if self.velocity_readout != "none":
            n = sum(p.numel() for p in self.vel_encoder.parameters())
            rows.insert(-1, ("vel_encoder", "slot velocities -> head (centered)", n))
            trained += n
        return rows, trained

    # ------------------------------------------------- static slot with h-tracking / handover
    def encode(self, seq, return_states=False):
        if self.model == "melstm" and self.static_slot and self.velocity_source == "tracked":
            p = self.x_track_p if self.training else 0.0
            h, vels, states = self._encode_static_tracked(seq, p, return_states)
            v = torch.stack(vels, dim=1)
            st = torch.stack(states, dim=1) if states else None
            return (h, v, st) if return_states else (h, v)
        return super().encode(seq, return_states=return_states)

    def _encode_static_tracked(self, seq, p, return_states=False):
        """
        MotionDigitClassifier's tracked encoder (p = 0, the backbone's own protocol) and its
        stochastic handover (0 < p <= 1, _encode_melstm_mixed), with slot 0 at rest: the bootstrap
        and the frame-pair candidates go to slots 1..K-1, and only those slots track.
        """
        cell = self.backbone.cell
        B, T, C, H, W = seq.shape
        K = self.n_velocities
        h, c = cell.init_hidden(B, K, H, W, seq.device, seq.dtype)
        rest = torch.zeros(B, 1, 2, device=seq.device, dtype=seq.dtype)
        v = torch.zeros(B, K, 2, device=seq.device, dtype=seq.dtype)
        vels, states = [], ([] if return_states else None)
        for t in range(T):
            if t == 1:
                boot = self.backbone.bootstrap_velocities(seq[:, 0], seq[:, 1]).to(seq.dtype)
                v = torch.cat([rest, boot[:, :K - 1]], dim=1)
            elif t >= 2:
                if p > 0:
                    with torch.no_grad():
                        cand = self._frame_pair_pc(seq[:, t - 1], seq[:, t])[0]
                    v_x = self._assign(cand[:, :K - 1].to(seq.dtype), v[:, 1:])
                if p >= 1:
                    v_new = v_x
                else:
                    v_h = self.backbone.track_velocities(h[:, 1:], seq[:, t]).to(seq.dtype)
                    if p <= 0:
                        v_new = v_h
                    else:
                        use_x = torch.rand(B, device=seq.device) < p
                        v_new = torch.where(use_x.view(B, 1, 1), v_x, v_h)
                v = torch.cat([rest, v_new], dim=1)
            h, c = cell(seq[:, t], h, c, v)
            if t > 0:
                vels.append(v.detach())
            if return_states:
                states.append(h.mean(dim=2).detach())
        return h, vels, states

    def _candidate_velocities(self, x0, x1):
        cand = super()._candidate_velocities(x0, x1)
        if self.static_slot:
            cand = torch.cat([torch.zeros_like(cand[:, :1]), cand[:, :-1]], dim=1)
        return cand

    def _assign(self, cand, v_prev):
        match = MotionDigitClassifier._match_to_slots
        if self.slot_assign == "anchored" and cand.shape[1] > 1:
            return torch.cat([cand[:, :1], match(cand[:, 1:], v_prev[:, 1:])], dim=1)
        if self.slot_assign == "shift":
            return match_up_to_shift(cand, v_prev)
        return match(cand, v_prev)

    def _match_to_slots(self, cand, v_prev):
        if self.static_slot:
            return torch.cat([cand[:, :1], self._assign(cand[:, 1:], v_prev[:, 1:])], dim=1)
        return self._assign(cand, v_prev)

    def copy_labels(self):
        """Row labels for the velocity axis: lattice velocities, slot names, or the one state."""
        if self.model == "felstm":
            return [f"({vx},{vy})" for vx, vy in self.backbone.cell.v_list]
        if self.model == "melstm":
            return [("s0 (0,0)" if self.static_slot and k == 0 else f"s{k}")
                    for k in range(self.n_velocities)]
        return ["h"]

    def pool_label(self):
        label = {"max": "max over V", "attention": "attention\npool", "mean": "mean over V",
                 "concat": "concat V\n(ch. mean)"}[self.pool.mode]
        if self.readout_steps > 1:
            return label + f"\n(head input,\nlast {self.readout_steps} steps)"
        return label + "\n(head input)"


def match_up_to_shift(cand, v_prev):
    """
    Nearest-previous matching after the best common shift. For every delta in the K^2 offsets
    cand_j - v_prev_k, match greedily to v_prev + delta (MotionDigitClassifier's rule) and keep
    the assignment with the smallest total L1 cost; the first minimum wins. Every quantity is a
    difference of velocities and the candidate and slot orders are themselves equivariant, so
    adding any c to every candidate and any c' to every previous velocity leaves the chosen
    permutation unchanged.
    """
    B, K, _ = cand.shape
    deltas = (cand[:, :, None, :] - v_prev[:, None, :, :]).reshape(B, K * K, 2)
    best = cand.clone()
    best_cost = torch.full((B,), float("inf"), dtype=cand.dtype, device=cand.device)
    for i in range(K * K):
        target = v_prev + deltas[:, i:i + 1, :]
        out = MotionDigitClassifier._match_to_slots(cand, target)
        cost = (out - target).abs().sum(dim=(-2, -1))
        better = cost < best_cost
        best = torch.where(better[:, None, None], out, best)
        best_cost = torch.where(better, cost, best_cost)
    return best


def phase_corr_kwargs(cfg):
    get = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    pc = {}
    if get("pc_subpixel", 0):
        pc["subpixel"] = True
    if get("pc_suppress_radius", 0):
        pc["suppress_radius"] = int(get("pc_suppress_radius"))
    r = get("pc_search_radius", None)
    if r is not None and int(r) > 0:
        pc["search_radius"] = int(r)
    if get("pc_alpha", 1.0) != 1.0:
        pc["alpha"] = float(get("pc_alpha"))
    return pc


def build_kth_classifier(cfg):
    """From the argparse Namespace or the config dict saved with a checkpoint."""
    get = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    pc = phase_corr_kwargs(cfg)
    net = KTHClassifier(
        model=get("model", "melstm"),
        hidden_channels=get("hidden_size", 64),
        kernel_size=get("kernel_size", 3),
        v_range=get("v_range", 1),
        n_slots=get("num_vel_modes", 4),
        n_classes=len(KTH_ACTIONS),
        velocity_pool=get("velocity_pool", "max"),
        velocity_source=get("velocity_source", "frame_pair"),
        head_channels=get("head_channels", 64),
        head_blocks=get("head_blocks", 3),
        head_mlp_hidden=get("head_mlp_hidden", 128),
        head_dropout=get("head_dropout", 0.0),
        head_norm=get("head_norm", "batch"),
        head_mlp_act=get("head_mlp_act", "leaky"),
        phase_corr_kwargs=pc,
        forget_bias=get("forget_bias", None),
        slot_assign=get("slot_assign", "nearest"),
        static_slot=bool(get("static_slot", 0)),
        readout_steps=int(get("readout_steps", 1)),
        velocity_readout=get("velocity_readout", "none"),
    )
    if net.model == "melstm":
        net.backbone.cell.integer_shift = not pc.get("subpixel", False)
    return net


@torch.no_grad()
def record_states(model, seq):
    """
    One eval-mode encoder pass with a hook on the recurrent cell, which every backbone calls
    once per frame with the full state h (B, V, C, H, W). Per step, keeps:

      copies : channel mean of every velocity copy / slot             (B, T, V, H, W)
      pooled : channel mean of the model's own velocity pool of h -- at the last step exactly
               what the head reads (max, attention, ...)               (B, T, H, W)
      winner : per pixel, the copy that supplies the max in the most channels (B, T, H, W)
      share  : at the last step, the fraction of all (channel, pixel) maxima each copy supplies,
               per sample                                              (B, V)
      weights: the attention pool's weight on each copy per step (B, T, V), or None
    plus the slot velocities (B, T-1, K, 2) for melstm (None otherwise) and the logits.
    """
    rec = []

    def hook(_module, _inputs, out):
        h = out[0]
        B, V, C, H, W = h.shape
        feat, w = model.pool(h)
        hmax, arg = h.max(dim=1)                                        # (B, C, H, W)
        counts = torch.zeros(B, V, H, W, device=h.device).scatter_add_(
            1, arg, torch.ones_like(hmax))
        rec.append((h.mean(dim=2), feat.mean(dim=1), counts.argmax(dim=1),
                    counts.sum(dim=(-2, -1)) / (C * H * W), w))

    was_training = model.training
    model.eval()
    handle = model.backbone.cell.register_forward_hook(hook)
    try:
        velocities = model.encode(seq)[1]
    finally:
        handle.remove()
    logits = model(seq)
    model.train(was_training)
    return {
        "copies": torch.stack([r[0] for r in rec], dim=1).cpu(),
        "pooled": torch.stack([r[1] for r in rec], dim=1).cpu(),
        "winner": torch.stack([r[2] for r in rec], dim=1).cpu(),
        "share": rec[-1][3].cpu(),
        "weights": (None if rec[-1][4] is None
                    else torch.stack([r[4] for r in rec], dim=1).cpu()),
        "velocities": None if velocities is None else velocities.cpu(),
        "logits": logits.cpu(),
    }
