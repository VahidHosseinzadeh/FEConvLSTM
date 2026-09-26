"""Persistence forecasts as parameter-free nn.Modules, so they run through the same evaluation."""
import torch
import torch.nn as nn

from ..common.baselines import eulerian_persistence, lagrangian_persistence


class Persistence(nn.Module):
    """
    mode     'eulerian' | 'lagrangian'
    velocity 'estimated' (phase correlation on the context) | 'oracle' (true future motion)
    """

    def __init__(self, mode="lagrangian", velocity="estimated", alpha=0.5, subpixel=True,
                 n_pairs=1, channels=None, window=False, shift="fourier", radius=None):
        super().__init__()
        self.mode = mode
        self.velocity = velocity
        self.kw = dict(alpha=alpha, subpixel=subpixel, n_pairs=n_pairs, channels=channels,
                       window=window, radius=radius)
        self.shift = shift
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)   # .to(device) works

    @torch.no_grad()
    def forward(self, input_seq, pred_len, motion=None):
        if self.mode == "eulerian":
            return eulerian_persistence(input_seq, pred_len)
        if self.velocity == "oracle":
            if motion is None:
                raise ValueError("oracle Lagrangian persistence needs the true motion")
            T_in = input_seq.shape[1]
            v = motion[:, T_in - 1:T_in - 1 + pred_len, 0].to(input_seq.dtype)
            return lagrangian_persistence(input_seq, pred_len, velocity=v, shift=self.shift)
        return lagrangian_persistence(input_seq, pred_len, shift=self.shift, **self.kw)
