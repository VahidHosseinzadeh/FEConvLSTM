"""
MotionVideoClassifier -- MotionDigitClassifier (moving_mnist/motion_classification_model.py) with
the three things real, articulating silhouettes need, all opt-in:

phase_corr_kwargs   sub-pixel / partially whitened / windowed / suppressed phase correlation for
                    every PC module the classifier and its backbone own (real motion is not an
                    integer number of pixels). {} keeps the original integer modules.
forget_bias(_long)  the memory time constant. A long memory averages the swinging limbs away and
                    keeps only the torso; ~2-4 frames keeps the instantaneous pose. Two
                    timescales (half the channels short, half long) keeps both
                    (claude/common_fate_real_video.md sections 7 and 9).
readout_steps       classify from the TRAJECTORY of transported states, not only h_T: logits are
                    averaged over the last `readout_steps` encoder steps. The head is
                    translation-invariant (circular convs + global pooling), so states at
                    different positions can be averaged at the logit level. The articulation IS
                    the label; the final state alone has already integrated it away.

With the defaults the model is MotionDigitClassifier exactly (same modules, same init order).
"""
import torch

from .. import _repo  # noqa: F401
from motion_classification_model import MotionDigitClassifier  # noqa: E402
from velocity_predictor_model import PhaseCorrelation  # noqa: E402

from .gates import set_forget_bias  # noqa: E402


class MotionVideoClassifier(MotionDigitClassifier):

    def __init__(self, *args, phase_corr_kwargs=None, forget_bias=None, forget_bias_long=None,
                 long_fraction=0.5, readout_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        pc = dict(phase_corr_kwargs or {})
        if pc:
            K = max(1, self.n_velocities)
            self._frame_pair_pc = PhaseCorrelation(n_modes=K, **pc)
            self._pc1 = PhaseCorrelation(n_modes=1, **pc)
            if self.model == "melstm":
                self.backbone.phase_corr_bootstrap = PhaseCorrelation(
                    n_modes=self.backbone.n_slots, **pc)
                self.backbone.phase_corr_track = PhaseCorrelation(n_modes=1, **pc)
        if hasattr(self.backbone.cell, "integer_shift"):
            # sub-pixel velocities need the exact warp (see MEConvLSTMCell.integer_shift)
            self.backbone.cell.integer_shift = False
        set_forget_bias(self.backbone.cell.conv, self.hidden_channels, short=forget_bias,
                        long=forget_bias_long, long_fraction=long_fraction)
        self.readout_steps = int(readout_steps)

    def _encode_collect(self, seq):
        """Encoder that keeps the full state of the last `readout_steps` steps."""
        B, T, C, H, W = seq.shape
        keep_from = max(0, T - self.readout_steps)
        hs, vels = [], []
        if self.model == "melstm":
            cell = self.backbone.cell
            K = self.n_velocities
            h, c = cell.init_hidden(B, K, H, W, seq.device, seq.dtype)
            v = torch.zeros(B, K, 2, device=seq.device, dtype=seq.dtype)
            for t in range(T):
                if t > 0:
                    if self.velocity_source in ("frame_pair", "bootstrap"):
                        with torch.no_grad():
                            cand = self._candidate_velocities(seq[:, t - 1], seq[:, t])
                        cand = cand.to(seq.dtype)
                        v = cand if t == 1 else self._match_to_slots(cand, v)
                    elif t == 1:
                        v = self.backbone.bootstrap_velocities(seq[:, 0], seq[:, 1]).to(seq.dtype)
                    else:
                        v = self.backbone.track_velocities(h, seq[:, t]).to(seq.dtype)
                h, c = cell(seq[:, t], h, c, v)
                if t > 0:
                    vels.append(v.detach())
                if t >= keep_from:
                    hs.append(h)
            vel = torch.stack(vels, dim=1) if vels else None
        else:
            cell = self.backbone.cell
            h, c = cell.init_hidden(B, H, W, seq.device)
            for t in range(T):
                h, c = cell(seq[:, t], (h, c))
                if t >= keep_from:
                    hs.append(h)
            vel = None
        return hs, vel

    def forward(self, seq, return_aux=False):
        if self.readout_steps <= 1:
            return super().forward(seq, return_aux=return_aux)
        hs, vel = self._encode_collect(seq)
        logits, weights = [], []
        for h in hs:
            f, w = self.pool(h)
            logits.append(self.head(f))
            weights.append(w)
        out = torch.stack(logits, dim=0).mean(dim=0)
        if not return_aux:
            return out
        w = weights[-1]
        return out, {"velocities": vel, "pool_weights": w}
