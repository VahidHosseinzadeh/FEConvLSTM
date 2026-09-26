"""
FEConvLSTMPlus -- Seq2SeqFEConvLSTM (v_range = 0 is the ConvLSTM baseline) with opt-in extras.

Same construction order as the parent, so default arguments give bit-identical weights and
outputs. Extras:

residual  'none' (parent) | 'eulerian' (x_last + decoder) | 'lagrangian' (x_last shifted by a
          phase-correlation estimate from the context, frozen over the rollout, + decoder).
          'lagrangian' on a plain ConvLSTM is the key ABLATION for the transport claim: it gives
          the baseline the same transported skip as MEConvLSTM, so any remaining gap is due to
          transporting the MEMORY, not the output.
decoder_input 'previous' (parent: the unshifted previous frame meets every shifted velocity
          copy -- misaligned by v, so the FE rollout is not exactly flow-equivariant) | 'warped'
          (each velocity copy receives the previous frame shifted by its own velocity: aligned,
          exactly equivariant) | 'zeros'.
forget_bias / forget_bias_long / long_fraction   see models/gates.py.
"""
import torch

from .. import _repo  # noqa: F401
from channel_based_FEConvLSTM_model import Seq2SeqFEConvLSTM  # noqa: E402

from ..common.baselines import estimate_last_velocity  # noqa: E402
from ..common.shifts import shift_torch  # noqa: E402
from .gates import set_forget_bias  # noqa: E402


class FEConvLSTMPlus(Seq2SeqFEConvLSTM):

    def __init__(self, input_channels, hidden_channels, output_channels=None, kernel_size=3,
                 v_range=0, pool_type="max", decoder_conv_layers=1, decoder_channels=None,
                 residual="none", residual_shift="fourier", pc_alpha=0.5, pc_subpixel=True,
                 pc_channels=None, pc_pairs=1, pc_window=False, pc_radius=None,
                 decoder_input="previous",
                 forget_bias=None, forget_bias_long=None, long_fraction=0.5):
        super().__init__(input_channels, hidden_channels, output_channels=output_channels,
                         kernel_size=kernel_size, v_range=v_range, pool_type=pool_type,
                         decoder_conv_layers=decoder_conv_layers,
                         decoder_channels=decoder_channels)
        if residual not in ("none", "eulerian", "lagrangian"):
            raise ValueError("residual must be none|eulerian|lagrangian")
        self.residual = residual
        self.residual_shift = residual_shift
        self.pc_alpha = pc_alpha
        self.pc_subpixel = pc_subpixel
        self.pc_channels = pc_channels
        self.pc_pairs = pc_pairs
        self.pc_window = bool(pc_window)
        self.pc_radius = pc_radius
        if decoder_input not in ("previous", "warped", "zeros"):
            raise ValueError("decoder_input must be previous|warped|zeros")
        self.decoder_input = decoder_input
        set_forget_bias(self.cell.conv, hidden_channels, short=forget_bias,
                        long=forget_bias_long, long_fraction=long_fraction)

    def _decoder_step(self, x, h, c):
        if self.decoder_input == "previous":
            return self.cell(x, (h, c))
        if self.decoder_input == "zeros":
            return self.cell(torch.zeros_like(x), (h, c))
        cell = self.cell
        B, C, H, W = x.shape
        nv, Ch = cell.num_v, cell.hidden_channels
        xs = cell.shift_tensor(x.unsqueeze(1).expand(B, nv, C, H, W).contiguous())
        hs = cell.shift_tensor(h).reshape(B * nv, Ch, H, W)
        cs = cell.shift_tensor(c).reshape(B * nv, Ch, H, W)
        i, f, o, g = torch.chunk(cell.conv(torch.cat([xs.reshape(B * nv, C, H, W), hs], 1)), 4, 1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c_next = f * cs + i * g
        h_next = o * torch.tanh(c_next)
        return h_next.reshape(B, nv, Ch, H, W), c_next.reshape(B, nv, Ch, H, W)

    def forward(self, input_seq, pred_len, return_states=False):
        if self.residual == "none" and self.decoder_input == "previous":
            return super().forward(input_seq, pred_len, return_states=return_states)

        h, c, h_states = self.encode(input_seq, return_states=return_states)
        x_last = input_seq[:, -1]
        if self.residual == "lagrangian":
            v = estimate_last_velocity(input_seq, n_pairs=self.pc_pairs, alpha=self.pc_alpha,
                                       subpixel=self.pc_subpixel, channels=self.pc_channels,
                                       radius=self.pc_radius, window=self.pc_window)
            v = v.to(x_last.dtype)
        prev = x_last
        outputs = []
        for t in range(pred_len):
            h, c = self._decoder_step(prev.detach(), h, c)
            if return_states:
                h_states.append(h.mean(dim=2).detach())
            out = self.decoder_conv(self.pool_velocity(h))
            if self.residual == "eulerian":
                pred = x_last + out
            elif self.residual == "lagrangian":
                pred = shift_torch(x_last, v * (t + 1), self.residual_shift) + out
            else:
                pred = out
            outputs.append(pred)
            prev = pred
        outputs = torch.stack(outputs, dim=1)
        if return_states:
            return outputs, {"h": torch.stack(h_states, dim=1)}
        return outputs
