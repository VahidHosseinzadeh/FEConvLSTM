"""
Persistence forecasts -- the operational baselines of nowcasting, and the two limits between which
the learned models live:

  Eulerian persistence    copy the last frame in place. What an in-place recurrence tends
                          toward; the ceiling-free baseline a ConvLSTM must beat.
  Lagrangian persistence  move the last frame along the (estimated or true) motion, no change of
                          intensity. Transport without learning; the baseline MEConvLSTM must beat.

headroom = MSE(Eulerian) / MSE(Lagrangian) is the first-order ceiling on how much transport can buy
on a dataset (claude/third_experiment_options.md, synthetic radar table).
"""
import torch

from .phase_correlation import phase_correlate, hann2d
from .shifts import shift_torch


@torch.no_grad()
def eulerian_persistence(inp, pred_len):
    """inp (B, T, C, H, W) -> (B, pred_len, C, H, W) copies of the last frame."""
    return inp[:, -1:].expand(-1, pred_len, -1, -1, -1).clone()


@torch.no_grad()
def estimate_last_velocity(inp, n_pairs=1, alpha=0.5, subpixel=True, radius=None,
                           channels=None, window=False):
    """
    Velocity from the last n_pairs consecutive frame pairs of inp (B, T, C, H, W); the
    component-wise median over pairs when n_pairs > 1 (robust to one bad pair). Returns (B, 2).
    """
    T = inp.shape[1]
    n_pairs = max(1, min(n_pairs, T - 1))
    win = hann2d(*inp.shape[-2:], device=inp.device) if window else None
    vs = []
    for j in range(n_pairs):
        a, b = inp[:, T - 2 - j], inp[:, T - 1 - j]
        v, _, _ = phase_correlate(a, b, k=1, alpha=alpha, radius=radius, subpixel=subpixel,
                                  channels=channels, window=win)
        vs.append(v[:, 0])
    return torch.stack(vs, 0).median(dim=0).values if n_pairs > 1 else vs[0]


@torch.no_grad()
def lagrangian_persistence(inp, pred_len, velocity=None, shift="fourier", **est_kw):
    """
    inp (B, T, C, H, W). `velocity`:
        None            estimate from the context (est_kw -> estimate_last_velocity)
        (B, 2)          one velocity for the whole rollout
        (B, pred_len, 2) per-step displacements (e.g. the true future motion: an oracle)
    Frame t of the forecast is the last input frame shifted by the cumulative displacement --
    ONE interpolation from the original, never repeated, so no numerical diffusion builds up.
    """
    x = inp[:, -1]
    if velocity is None:
        velocity = estimate_last_velocity(inp, **est_kw)
    velocity = velocity.to(x.dtype)
    if velocity.dim() == 2:
        velocity = velocity[:, None, :].expand(-1, pred_len, -1)
    D = torch.cumsum(velocity, dim=1)
    return torch.stack([shift_torch(x, D[:, t], shift) for t in range(pred_len)], dim=1)
