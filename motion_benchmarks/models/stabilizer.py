"""
Oracle stabiliser: undo the TRUE frame motion, run a static model in the co-moving frame, redo it.

For a single global frame motion this is the best a static-symmetry model (a ConvLSTM, or Fromme
et al.'s Z^2 x| D4-equivariant ConvLSTM) can do when it is TOLD the frame. MEConvLSTM should match
it without being told -- that is the honest, strong form of the claim (claude/fluids_rbc_plan.md
section 6), so report MEConvLSTM against this, not only against the plain ConvLSTM.

Regular action only (fields translated, values untouched). Uses motion slot 0 as the frame motion.
"""
import torch
import torch.nn as nn

from ..common.shifts import shift_torch


class OracleStabilized(nn.Module):

    def __init__(self, base_model, shift="fourier"):
        super().__init__()
        self.base = base_model
        self.shift = shift

    @staticmethod
    def displacements(motion):
        """motion (B, T, N, 2) -> D (B, T, 2) with D[:, 0] = 0, D[:, s] = sum motion[:, :s, 0]."""
        m = motion[:, :, 0].float()
        z = torch.zeros_like(m[:, :1])
        return torch.cat([z, torch.cumsum(m, dim=1)[:, :-1]], dim=1)

    def forward(self, input_seq, pred_len, motion, **base_kwargs):
        B, T_in = input_seq.shape[:2]
        if motion is None or motion.shape[1] < T_in + pred_len:
            raise ValueError("OracleStabilized needs ground-truth motion covering context + rollout")
        D = self.displacements(motion).to(input_seq.dtype)
        stab = torch.stack([shift_torch(input_seq[:, s], -D[:, s], self.shift)
                            for s in range(T_in)], dim=1)
        out = self.base(stab, pred_len=pred_len, **base_kwargs)
        if isinstance(out, tuple):
            out = out[0]
        return torch.stack([shift_torch(out[:, t], D[:, T_in + t], self.shift)
                            for t in range(pred_len)], dim=1)
