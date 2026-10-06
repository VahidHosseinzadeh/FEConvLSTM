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

from motion_benchmarks import _repo  # noqa: F401  (puts moving_mnist/ on sys.path)
from motion_benchmarks.models.cf_classifier import MotionVideoClassifier  # noqa: E402
from motion_classification_model import MotionDigitClassifier  # noqa: E402

from .kth_dataset import KTH_ACTIONS


class KTHClassifier(MotionVideoClassifier):
    """
    static_slot  melstm with a frame-pair or bootstrap source: slot 0 is pinned to velocity
                 (0, 0) -- an untransported state, i.e. a ConvLSTM inside the MEConvLSTM -- and
                 slots 1..K-1 take the top K-1 candidates, assigned by slot_assign among
                 themselves. With K-1 moving slots this is exactly the K-1-slot model plus one
                 static slot. Same parameters (all slots share the cell).
    """

    def __init__(self, *args, slot_assign="nearest", static_slot=False, **kwargs):
        super().__init__(*args, **kwargs)
        if slot_assign not in ("nearest", "shift", "anchored"):
            raise ValueError(f"slot_assign {slot_assign!r}: expected nearest, shift or anchored")
        if static_slot and (self.model != "melstm" or self.velocity_source == "tracked"
                            or self.n_velocities < 2):
            raise ValueError("static_slot needs melstm, >= 2 slots and a frame-pair or "
                             "bootstrap velocity source")
        self.slot_assign = slot_assign
        self.static_slot = bool(static_slot)

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
        return {"max": "max over V", "attention": "attention\npool", "mean": "mean over V",
                "concat": "concat V\n(ch. mean)"}[self.pool.mode] + "\n(head input)"


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
