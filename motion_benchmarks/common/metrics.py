"""
Forecast metrics, accumulated over batches and reported per lead time.

  PerLeadError      MSE / MAE / RMSE per lead (optionally per channel group), pooled over the set
  Categorical       hits / misses / false alarms at intensity thresholds -> CSI, POD, FAR, bias
  FSS               fractions skill score at thresholds x neighbourhood sizes (Roberts & Lean
                    2008), pooled over the set -- what nowcasters read after CSI
  radial_spectrum   isotropic power spectrum, for kinetic-energy spectra of fluid rollouts
  nusselt_volume    volume-averaged Nusselt number from T and w
  velocity_epe      endpoint error of estimated velocities, NaN-aware

Everything takes torch tensors shaped (B, T_pred, C, H, W) for fields.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F


def _to_np(x):
    return x.detach().cpu().double().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


class PerLeadError:
    """Pooled per-lead MSE/MAE. `groups` maps a name to a list of channel indices."""

    def __init__(self, groups=None):
        self.groups = dict(groups or {})
        self.sq = None
        self.ab = None
        self.n = 0
        self.gsq = {g: None for g in self.groups}
        self.gn = 0

    @torch.no_grad()
    def update(self, pred, target):
        d = (pred.float() - target.float())
        sq = (d ** 2).mean(dim=(2, 3, 4)).sum(dim=0)      # (T,)
        ab = d.abs().mean(dim=(2, 3, 4)).sum(dim=0)
        self.sq = sq if self.sq is None else self.sq + sq
        self.ab = ab if self.ab is None else self.ab + ab
        self.n += pred.shape[0]
        for g, idx in self.groups.items():
            gs = (d[:, :, idx] ** 2).mean(dim=(2, 3, 4)).sum(dim=0)
            self.gsq[g] = gs if self.gsq[g] is None else self.gsq[g] + gs
        self.gn += pred.shape[0]

    def result(self):
        if self.n == 0:
            return {}
        mse = _to_np(self.sq / self.n)
        out = {"mse": mse.tolist(), "mae": _to_np(self.ab / self.n).tolist(),
               "rmse": np.sqrt(mse).tolist(), "mse_mean": float(mse.mean())}
        for g in self.groups:
            m = _to_np(self.gsq[g] / self.gn)
            out[f"mse_{g}"] = m.tolist()
            out[f"rmse_{g}"] = np.sqrt(m).tolist()
        return out


class Categorical:
    """Contingency counts per threshold and lead, pooled over pixels, channels and samples."""

    def __init__(self, thresholds):
        self.thresholds = [float(t) for t in thresholds]
        self.h = self.m = self.f = None

    @torch.no_grad()
    def update(self, pred, target):
        hs, ms, fs = [], [], []
        for thr in self.thresholds:
            p = pred >= thr
            o = target >= thr
            hs.append((p & o).sum(dim=(0, 2, 3, 4)).double())
            ms.append((~p & o).sum(dim=(0, 2, 3, 4)).double())
            fs.append((p & ~o).sum(dim=(0, 2, 3, 4)).double())
        h, m, f = torch.stack(hs), torch.stack(ms), torch.stack(fs)      # (n_thr, T)
        self.h = h if self.h is None else self.h + h
        self.m = m if self.m is None else self.m + m
        self.f = f if self.f is None else self.f + f

    def result(self):
        if self.h is None:
            return {}
        h, m, f = _to_np(self.h), _to_np(self.m), _to_np(self.f)
        with np.errstate(divide="ignore", invalid="ignore"):
            csi = h / (h + m + f)
            pod = h / (h + m)
            far = f / (h + f)
            bias = (h + f) / (h + m)
        out = {}
        for i, thr in enumerate(self.thresholds):
            key = f"{thr:g}"
            out[f"csi@{key}"] = csi[i].tolist()
            out[f"pod@{key}"] = pod[i].tolist()
            out[f"far@{key}"] = far[i].tolist()
            out[f"bias@{key}"] = bias[i].tolist()
        return out


class FSS:
    """
    Fractions skill score, pooled: FSS = 1 - sum (P - O)^2 / (sum P^2 + sum O^2), where P and O
    are the fractions of pixels >= threshold in an n x n neighbourhood (zero padding, as in
    pysteps' implementation). Scales must be odd.
    """

    def __init__(self, thresholds, scales=(1, 5, 9, 17)):
        self.thresholds = [float(t) for t in thresholds]
        self.scales = [int(s) if int(s) % 2 == 1 else int(s) + 1 for s in scales]
        self.num = {}
        self.den = {}

    @torch.no_grad()
    def update(self, pred, target):
        B, T, C, H, W = pred.shape
        for thr in self.thresholds:
            p = (pred >= thr).float().reshape(B * T * C, 1, H, W)
            o = (target >= thr).float().reshape(B * T * C, 1, H, W)
            for n in self.scales:
                if n > 1:
                    P = F.avg_pool2d(p, n, stride=1, padding=n // 2, count_include_pad=True)
                    O = F.avg_pool2d(o, n, stride=1, padding=n // 2, count_include_pad=True)
                else:
                    P, O = p, o
                P = P.reshape(B, T, C, H, W)
                O = O.reshape(B, T, C, H, W)
                num = ((P - O) ** 2).sum(dim=(0, 2, 3, 4)).double()
                den = (P ** 2 + O ** 2).sum(dim=(0, 2, 3, 4)).double()
                key = (thr, n)
                self.num[key] = num if key not in self.num else self.num[key] + num
                self.den[key] = den if key not in self.den else self.den[key] + den

    def result(self):
        out = {}
        for (thr, n), num in self.num.items():
            den = self.den[(thr, n)]
            with np.errstate(divide="ignore", invalid="ignore"):
                v = 1.0 - _to_np(num) / _to_np(den)
            out[f"fss@{thr:g}_n{n}"] = v.tolist()
        return out


# ---------------------------------------------------------------------------- spectra & physics
def radial_spectrum(x):
    """
    Isotropic power spectrum of (..., H, W) fields, binned on integer |k| (cycles per domain).
    Returns (..., K) with K = floor(min(H, W) / 2) + 1; bin 0 is the mean.
    """
    H, W = x.shape[-2:]
    X = torch.fft.fft2(x.float()) / (H * W)
    P = X.real ** 2 + X.imag ** 2
    ky = torch.fft.fftfreq(H, device=x.device) * H
    kx = torch.fft.fftfreq(W, device=x.device) * W
    K = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2).round().long()
    nb = min(H, W) // 2 + 1
    K = K.clamp_max(nb)            # everything beyond the last full shell goes to a discard bin
    flat = P.reshape(-1, H * W)
    out = torch.zeros(flat.shape[0], nb + 1, device=x.device, dtype=flat.dtype)
    out.index_add_(1, K.reshape(-1), flat)
    return out[:, :nb].reshape(*x.shape[:-2], nb)


def ke_spectrum(u, v):
    """Horizontal kinetic-energy spectrum 0.5 (|u_k|^2 + |v_k|^2), shells of integer |k|."""
    return 0.5 * (radial_spectrum(u) + radial_spectrum(v))


def log_spectral_distance(spec_pred, spec_true, k_min=1, eps=1e-20):
    """RMS difference of log10 spectra over shells k >= k_min (mean over leading dims)."""
    a = torch.log10(spec_pred[..., k_min:] + eps)
    b = torch.log10(spec_true[..., k_min:] + eps)
    return torch.sqrt(((a - b) ** 2).mean(dim=-1))


def nusselt_volume(T, w, kappa, delta_T=1.0, Lz=1.0):
    """
    Volume-averaged Nusselt number, Nu = 1 + <w T> Lz / (kappa delta_T), for T and w of shape
    (..., nz, H, W) in physical (un-normalised) units. For the Dedalus/Oceananigans scripts in
    physics/ (free-fall units, delta_T = 1): kappa = (Ra Pr)^(-1/2) and Lz = the layer height.
    """
    return 1.0 + (w * T).mean(dim=(-3, -2, -1)) * Lz / (kappa * delta_T)


def velocity_epe(v_pred, v_true):
    """Mean endpoint error over all finite entries of (..., 2) tensors; NaN if none."""
    v_pred = v_pred.float()
    v_true = v_true.float()
    e = torch.linalg.norm(v_pred - v_true, dim=-1)
    ok = torch.isfinite(e)
    return float(e[ok].mean()) if ok.any() else float("nan")
