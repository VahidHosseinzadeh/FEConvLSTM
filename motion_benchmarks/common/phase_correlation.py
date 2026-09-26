"""
Phase correlation, generalised: partial whitening, sub-pixel peaks, search windows, several
peaks with suppression, residual (multi-motion) peaks, confidence, and a gated tracker step.

One torch implementation; the numpy helpers at the bottom are thin wrappers around it, so the
data-side scripts and the models can never disagree about a sign or a convention.

Conventions
-----------
* Displacements are (vx, vy) = (columns, rows), the repo convention (see common/shifts.py).
* pc_surface(a, b) peaks at the displacement d that takes a to b:  b ~= shift(a, d), i.e.
  b[y, x] ~= a[y - dy, x - dx].  The legacy PhaseCorrelation module returns the same quantity
  (it correlates the other way round and negates), so the two agree.

What the knobs are for (all measured on this project's data, see claude/*.md in the project)
-------------------------------------------------------------------------------------------
alpha      whitening exponent in R / |R|^alpha. alpha = 1 is classical phase correlation and is
           right for BROADBAND data (common-fate noise textures). Narrowband physics fields
           (convection rolls, rain) need alpha ~= 0.5: full whitening gives equal weight to the
           many high-k modes that carry almost no signal and was 5-10x less accurate on steady
           Swift-Hohenberg rolls; plain cross-correlation (alpha = 0) fails once the field
           evolves quickly (0.40 px vs 0.10 px at a 3-frame rain lifetime).
subpixel   3-point parabolic refinement of the integer peak: 0.11 px median error vs 0.36 px for
           the integer peak and 0.26 px for a soft-argmax on sub-pixel common-fate motion.
radius     restrict the peak to |d - center|_inf <= radius. Velocities are piecewise constant,
           so a large jump from the previous estimate is implausible -- a local window around
           v_prev is one of the three tracker fixes that made articulating subjects trackable.
suppress   non-maximum suppression radius between successive peaks. Without it a sub-pixel
           motion's peak spreads over two pixels and the "second motion" is just the other half
           of the first one.
min_conf   gate: when the peak's z-score falls below this, coast on the previous velocity
           instead of accepting the measurement.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from .shifts import fourier_shift_torch


# ============================================================================ surfaces
def hann2d(H, W, device=None, dtype=torch.float32):
    """Separable Hann window, for NON-periodic data (real radar/satellite crops): tapers the
    edges so the implied periodic wrap does not create a spurious zero-displacement peak."""
    wy = torch.hann_window(H, periodic=False, device=device, dtype=dtype)
    wx = torch.hann_window(W, periodic=False, device=device, dtype=dtype)
    return wy[:, None] * wx[None, :]


def pc_surface(a, b, alpha=1.0, eps=1e-8, demean=True, window=None):
    """
    a, b : (..., H, W) real tensors.
    Returns the correlation surface (..., H, W); its peak sits at index [dy mod H, dx mod W]
    where d = (dx, dy) is the displacement taking a to b.
    """
    if demean:
        a = a - a.mean(dim=(-2, -1), keepdim=True)
        b = b - b.mean(dim=(-2, -1), keepdim=True)
    if window is not None:
        a = a * window
        b = b * window
    Fa = torch.fft.rfft2(a)
    Fb = torch.fft.rfft2(b)
    R = Fb * Fa.conj()
    if alpha != 0:
        R = R / (R.abs() + eps) ** alpha
    return torch.fft.irfft2(R, s=a.shape[-2:])


def _signed(n, device):
    """Index -> signed displacement, same wrap rule as the legacy module (i > n/2 -> i - n)."""
    i = torch.arange(n, device=device)
    return torch.where(i > n / 2, i - n, i)


def _wrap(d, n):
    """Wrap an (integer-valued) displacement difference into (-n/2, n/2]."""
    return torch.remainder(d + (n - 1) // 2, n) - (n - 1) // 2


def parabolic_offset(sm, s0, sp):
    """Vertex of the parabola through (-1, sm), (0, s0), (+1, sp); 0 where it is not a maximum."""
    den = sm - 2 * s0 + sp
    off = 0.5 * (sm - sp) / torch.where(den.abs() < 1e-12, torch.ones_like(den), den)
    off = torch.where(den < 0, off, torch.zeros_like(off))
    return off.clamp(-0.5, 0.5)


def surface_zscore(s):
    """(B, H, W) -> per-sample (mean, std) of the surface, for peak confidence."""
    flat = s.reshape(s.shape[0], -1)
    return flat.mean(dim=1), flat.std(dim=1).clamp_min(1e-12)


def pc_peaks(s, k=1, radius=None, center=None, suppress=1, subpixel=True, allowed=None):
    """
    Extract up to k peaks from surfaces s (B, H, W).

    radius  : None, or search only |d - center|_inf <= radius.
    center  : None (window around 0) or (B, 2) (vx, vy) -- e.g. the previous velocity.
    suppress: Chebyshev radius zeroed around each accepted peak before looking for the next.
    allowed : optional (B, H, W) bool mask of admissible peak positions (e.g. excluding peaks
              already taken). Confidence and sub-pixel refinement always use the full surface.

    Returns
    -------
    v     : (B, k, 2) displacements (vx, vy), sub-pixel if requested
    peak  : (B, k)    raw surface value at each integer peak
    conf  : (B, k)    z-score of each peak against the whole surface
    """
    B, H, W = s.shape
    dev = s.device
    mu, sd = surface_zscore(s)
    gy = _signed(H, dev)                     # (H,)
    gx = _signed(W, dev)                     # (W,)

    if center is not None:
        cx = center[:, 0].round().long()
        cy = center[:, 1].round().long()
    else:
        cx = torch.zeros(B, dtype=torch.long, device=dev)
        cy = torch.zeros(B, dtype=torch.long, device=dev)

    # displacement of every surface pixel measured relative to the (rounded) centre
    rel_y = _wrap(gy[None, :] - cy[:, None], H)     # (B, H)
    rel_x = _wrap(gx[None, :] - cx[:, None], W)     # (B, W)

    work = s.clone()
    if allowed is not None:
        work = work.masked_fill(~allowed, float("-inf"))
    if radius is not None:
        r = int(radius)
        ok = (rel_y.abs() <= r)[:, :, None] & (rel_x.abs() <= r)[:, None, :]
        work = work.masked_fill(~ok, float("-inf"))

    rows = torch.arange(B, device=dev)
    vs, peaks, confs = [], [], []
    for j in range(k):
        flat = work.reshape(B, -1)
        val, idx = flat.max(dim=1)
        iy = idx // W
        ix = idx % W
        dy = (cy + rel_y[rows, iy]).to(s.dtype)
        dx = (cx + rel_x[rows, ix]).to(s.dtype)
        if subpixel:
            s0 = s[rows, iy, ix]
            oy = parabolic_offset(s[rows, (iy - 1) % H, ix], s0, s[rows, (iy + 1) % H, ix])
            ox = parabolic_offset(s[rows, iy, (ix - 1) % W], s0, s[rows, iy, (ix + 1) % W])
            dy = dy + oy
            dx = dx + ox
        # a sample whose window is exhausted gets a zero-confidence copy of its last peak
        dead = ~torch.isfinite(val)
        if dead.any() and j > 0:
            dx = torch.where(dead, vs[-1][:, 0], dx)
            dy = torch.where(dead, vs[-1][:, 1], dy)
        vs.append(torch.stack([dx, dy], dim=-1))
        pk = torch.where(dead, torch.zeros_like(val), s[rows, iy, ix])
        peaks.append(pk)
        confs.append(torch.where(dead, torch.zeros_like(val), (pk - mu) / sd))
        if j < k - 1:
            ry = _wrap(torch.arange(H, device=dev)[None, :] - iy[:, None], H).abs() <= suppress
            rx = _wrap(torch.arange(W, device=dev)[None, :] - ix[:, None], W).abs() <= suppress
            work = work.masked_fill(ry[:, :, None] & rx[:, None, :], float("-inf"))
    return torch.stack(vs, dim=1), torch.stack(peaks, dim=1), torch.stack(confs, dim=1)


def _collapse_channels(x, channels=None):
    """(B, C, H, W) -> (B, H, W): select channels (optional) and average, like the legacy module."""
    if x.dim() == 3:
        return x
    if channels is not None:
        x = x[:, channels]
    return x.mean(dim=1)


@torch.no_grad()
def phase_correlate(a, b, k=1, alpha=1.0, radius=None, center=None, subpixel=True,
                    suppress=1, window=None, channels=None, demean=True):
    """
    Batched motion estimate between frames a and b, (B, C, H, W) or (B, H, W).
    Returns (v (B, k, 2), peak (B, k), conf (B, k)).
    """
    a = _collapse_channels(a, channels).float()
    b = _collapse_channels(b, channels).float()
    s = pc_surface(a, b, alpha=alpha, window=window, demean=demean)
    return pc_peaks(s, k=k, radius=radius, center=center, suppress=suppress, subpixel=subpixel)


def _gaussian_blur_fft(x, sigma):
    if sigma <= 0:
        return x
    H, W = x.shape[-2:]
    ky = torch.fft.fftfreq(H, device=x.device)[:, None]
    kx = torch.fft.rfftfreq(W, device=x.device)[None, :]
    g = torch.exp(-2 * (math.pi * sigma) ** 2 * (ky ** 2 + kx ** 2))
    return torch.fft.irfft2(torch.fft.rfft2(x) * g, s=(H, W))


@torch.no_grad()
def residual_peaks(a, b, k=2, alpha=1.0, radius=None, subpixel=True, blur=1.5, suppress=1,
                   channels=None):
    """
    Several motions from one frame pair by explaining them away one at a time.

    Round 1 is ordinary phase correlation: the dominant peak is the motion carrying the most
    energy (the background, in common-fate data). Every later round correlates only what the
    motions found so far fail to explain: the residual |a - shift(b, -v_i)| (minimum over the
    found v_i), blurred into a soft weight map, multiplies both frames, and the peaks already
    taken are suppressed. This is the generalisation of the residual bootstrap that recovered the
    minority (figure) velocity in Common-Fate MNIST.

    Returns (v (B, k, 2), conf (B, k)).
    """
    a = _collapse_channels(a, channels).float()
    b = _collapse_channels(b, channels).float()
    a = a - a.mean(dim=(-2, -1), keepdim=True)
    b = b - b.mean(dim=(-2, -1), keepdim=True)
    B, H, W = a.shape
    found, confs = [], []
    wa = torch.ones_like(a)
    wb = torch.ones_like(b)
    explained = None
    for j in range(k):
        s = pc_surface(a * wa, b * wb, alpha=alpha)
        allowed = None
        if found:
            # exclude every peak already taken so the next round cannot re-find it
            allowed = torch.ones_like(s, dtype=torch.bool)
            for v in found:
                iy = torch.remainder(v[:, 1].round().long(), H)
                ix = torch.remainder(v[:, 0].round().long(), W)
                ry = _wrap(torch.arange(H, device=a.device)[None, :] - iy[:, None], H).abs() <= suppress
                rx = _wrap(torch.arange(W, device=a.device)[None, :] - ix[:, None], W).abs() <= suppress
                allowed &= ~(ry[:, :, None] & rx[:, None, :])
        v, _, conf = pc_peaks(s, k=1, radius=radius, suppress=suppress, subpixel=subpixel,
                              allowed=allowed)
        v = v[:, 0]
        found.append(v)
        confs.append(conf[:, 0])
        if j == k - 1:
            break
        # what does the union of found motions fail to explain?
        err = (a - fourier_shift_torch(b, -v)).abs()
        explained = err if explained is None else torch.minimum(explained, err)
        r = _gaussian_blur_fft(explained, blur).clamp_min(0)
        r = r / r.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        wa = r
        wb = fourier_shift_torch(r, v)
    return torch.stack(found, dim=1), torch.stack(confs, dim=1)


@torch.no_grad()
def gated_track(template, frame, v_prev, alpha=1.0, radius=None, min_conf=None,
                subpixel=True, erode_quantile=None, channels=None, window=None):
    """
    One tracker step: where did `template` go in `frame`?

    template : (B, H, W) or (B, C, H, W)   -- e.g. a slot's channel-mean hidden state
    frame    : (B, C, H, W)
    v_prev   : (B, 2) previous velocity (window centre and coasting value)

    erode_quantile : correlate only the template's highest-|energy| pixels (e.g. 0.8 keeps the
                     top 20%), so a deforming boundary contributes less.
    Returns (v (B, 2), conf (B,), accepted (B,) bool).
    """
    t = _collapse_channels(template).float()
    f = _collapse_channels(frame, channels).float()
    if erode_quantile is not None:
        e = (t - t.mean(dim=(-2, -1), keepdim=True)).abs()
        thr = torch.quantile(e.reshape(e.shape[0], -1), erode_quantile, dim=1)
        t = t * (e >= thr[:, None, None]).to(t.dtype)
    s = pc_surface(t, f, alpha=alpha, window=window)
    center = v_prev if radius is not None else None
    v, _, conf = pc_peaks(s, k=1, radius=radius, center=center, subpixel=subpixel)
    v, conf = v[:, 0], conf[:, 0]
    accepted = torch.ones_like(conf, dtype=torch.bool)
    if min_conf is not None:
        accepted = conf >= min_conf
        v = torch.where(accepted[:, None], v, v_prev.to(v.dtype))
    return v, conf, accepted


def match_to_slots(cand, v_prev):
    """
    Permute candidate velocities (B, K, 2) so candidate k is the one nearest slot k's previous
    velocity, greedily without replacement. Gives slots a persistent identity when peaks come out
    ordered by score (the ordering is unstable frame to frame). Same rule as
    MotionDigitClassifier._match_to_slots.
    """
    B, K, _ = cand.shape
    cost = (cand[:, :, None, :] - v_prev[:, None, :, :]).abs().sum(-1)   # (B, cand, slot)
    out = torch.zeros_like(cand)
    taken = torch.zeros(B, K, dtype=torch.bool, device=cand.device)
    rows = torch.arange(B, device=cand.device)
    for kk in range(K):
        c = cost[:, :, kk].masked_fill(taken, float("inf"))
        j = c.argmin(dim=1)
        out[:, kk] = cand[rows, j]
        taken[rows, j] = True
    return out


# ============================================================================ numpy wrappers
def estimate_shift_np(a, b, alpha=1.0, radius=None, subpixel=True, window=False):
    """
    Single pair, numpy in / numpy out. Returns (d (2,) = (vx, vy), confidence z-score).
    `window=True` applies a Hann taper (use for non-periodic crops).
    """
    ta = torch.as_tensor(np.asarray(a, dtype=np.float32))[None]
    tb = torch.as_tensor(np.asarray(b, dtype=np.float32))[None]
    win = hann2d(*ta.shape[-2:]) if window else None
    v, _, conf = phase_correlate(ta, tb, k=1, alpha=alpha, radius=radius, subpixel=subpixel,
                                 window=win)
    return v[0, 0].numpy().astype(np.float64), float(conf[0, 0])


def estimate_peaks_np(a, b, k=2, alpha=1.0, radius=None, subpixel=True, suppress=1):
    """Top-k peaks of one pair. Returns (d (k, 2), conf (k,))."""
    ta = torch.as_tensor(np.asarray(a, dtype=np.float32))[None]
    tb = torch.as_tensor(np.asarray(b, dtype=np.float32))[None]
    v, _, conf = phase_correlate(ta, tb, k=k, alpha=alpha, radius=radius, subpixel=subpixel,
                                 suppress=suppress)
    return v[0].numpy().astype(np.float64), conf[0].numpy()


def estimate_sequence_np(frames, alpha=1.0, radius=None, subpixel=True, window=False):
    """
    Consecutive-pair estimates for a (T, H, W) or (T, C, H, W) sequence.
    Returns motion (T, 2) in the repo convention: motion[t] = displacement t -> t+1, and the last
    entry (no successor) repeats the previous one.
    """
    x = torch.as_tensor(np.asarray(frames, dtype=np.float32))
    if x.dim() == 4:
        x = x.mean(dim=1)
    win = hann2d(*x.shape[-2:]) if window else None
    v, _, _ = phase_correlate(x[:-1], x[1:], k=1, alpha=alpha, radius=radius,
                              subpixel=subpixel, window=win)
    v = v[:, 0].numpy().astype(np.float64)
    return np.concatenate([v, v[-1:]], axis=0)
