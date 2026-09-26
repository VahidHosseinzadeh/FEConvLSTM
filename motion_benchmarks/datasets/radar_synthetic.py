"""
STEPS-style synthetic radar: the "synthetic but real" rain experiment.

Every knob is physical, and the ground-truth wind is known, so velocity accuracy is measurable:

  field      Gaussian random field with a power-law spectrum |k|^-beta (beta ~ 2.6, fitted to
             radar by pySTEPS), thresholded into intermittent rain (wet_fraction ~ 0.35, i.e.
             65% dry) with a log-normal-like intensity above the threshold.
  evolution  every Fourier mode is an AR(1) process with a SCALE-DEPENDENT Lagrangian lifetime,
             tau(k) = lifetime * (k_ref / k)^0.8 -- small structures die faster, the cascade
             structure of STEPS. `lifetime` (frames) is the e-folding time at `lifetime_scale_px`.
  advection  a uniform wind, time-varying (continuous-valued schedules in common/schedules.py),
             applied as an exact Fourier shift. Optional freeze_after for the rollout protocol.
  layers     n_layers > 1 superposes independent rain layers with independent winds and
             lifetimes (wind shear) -- a natural K-slot, local-motion case.

Measured headroom (Eulerian / Lagrangian-persistence MSE, estimated wind, 96x96, beta 2.6):

    lifetime   lead 1   lead 3   lead 6   lead 12
        3       1.12     1.08     1.05     1.09
        6       1.31     1.13     1.16     1.07
       12       2.13     1.68     1.36     1.12
       24       2.93     2.49     1.82     1.61
       48       5.29     4.03     2.76     2.05

Transport pays in proportion to the rain's lifetime. Reproduce with scripts/headroom.py.

Output transform: 'log' (default) x = log1p(R) / 4, which keeps x in ~[0, 1.1]; 'raw' x = R / 10.
meta['thresholds'] are the CSI/FSS thresholds in the SAME transformed units, from rain-rate
thresholds R in {0.5, 2, 8} (arbitrary but fixed units).
"""
import math

import numpy as np
from scipy.special import ndtri

from ..common.fields import complex_white, power_law_amplitude, wavenumber_grid
from ..common.schedules import apply_freeze, make_schedule
from ..common.shifts import cumulative_displacement
from .base import GeneratedSequenceDataset

RAIN_THRESHOLDS = (0.5, 2.0, 8.0)


def transform_rain(R, transform="log"):
    if transform == "log":
        return np.log1p(R) / 4.0
    if transform == "raw":
        return R / 10.0
    raise ValueError(f"unknown transform {transform!r}")


def inverse_transform_rain(x, transform="log"):
    if transform == "log":
        return np.expm1(np.maximum(x, 0) * 4.0)
    return np.maximum(x, 0) * 10.0


class SyntheticRadarDataset(GeneratedSequenceDataset):

    def __init__(self, length=10000, seq_len=24, image_size=96, beta=2.6, wet_fraction=0.35,
                 lifetime=(6.0, 48.0), lifetime_scale_px=24.0, scale_exponent=0.8,
                 max_speed=2.5, min_speed=0.0, schedule="piecewise", hold=(3, 6),
                 smooth_prob=0.0, ou_tau=6.0, freeze_after=None, n_layers=1,
                 layer_combine="max", intensity_gain=1.2, transform="log", seed=0, random=True):
        super().__init__(length, seed, random)
        self.seq_len = int(seq_len)
        self.S = int(image_size)
        self.beta = float(beta)
        self.wet_fraction = float(wet_fraction)
        self.lifetime = lifetime
        self.scale_exponent = float(scale_exponent)
        self.max_speed = float(max_speed)
        self.min_speed = float(min_speed)
        self.schedule = schedule
        self.hold = tuple(hold)
        self.smooth_prob = float(smooth_prob)
        self.ou_tau = float(ou_tau)
        self.freeze_after = freeze_after
        self.n_layers = int(n_layers)
        self.layer_combine = layer_combine
        self.gain = float(intensity_gain)
        self.transform = transform

        S = self.S
        self._amp = power_law_amplitude(S, S, self.beta)
        K = wavenumber_grid(S, S)
        K[0, 0] = 1.0
        self._K = K
        self._k_ref = S / float(lifetime_scale_px)
        self._thr = float(ndtri(1.0 - self.wet_fraction))
        ky = np.fft.fftfreq(S)[:, None]
        kx = np.fft.fftfreq(S)[None, :]
        self._ky, self._kx = ky, kx

        self.meta = dict(
            name="radar_synthetic", in_channels=1, out_channels=1, channel_names=["rain"],
            has_motion=True, n_motions=self.n_layers, periodic=True,
            thresholds=[float(transform_rain(np.array(r), transform)) for r in RAIN_THRESHOLDS],
            rain_thresholds=list(RAIN_THRESHOLDS), fss_scales=[1, 5, 9, 17],
            pc_alpha=0.5, transform=transform,
            pc_search_radius=int(math.ceil(max(self.max_speed, 1.0))) + 1,
        )

    # --------------------------------------------------------------------------------------
    def _draw_lifetime(self, rng):
        lt = self.lifetime
        if isinstance(lt, (tuple, list)):
            lo, hi = float(lt[0]), float(lt[1])
            return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        return float(lt)

    def _layer(self, rng, T):
        """One rain layer: (T, S, S) standardised Gaussian field + its (T, 2) motion."""
        S = self.S
        lifetime = self._draw_lifetime(rng)
        tau = lifetime * (self._k_ref / self._K) ** self.scale_exponent
        rho = np.exp(-1.0 / tau)
        innov = np.sqrt(1.0 - rho ** 2) * self._amp
        motion = make_schedule(self.schedule, T, rng, self.max_speed, min_speed=self.min_speed,
                               hold=self.hold, smooth_prob=self.smooth_prob, tau=self.ou_tau)
        motion = apply_freeze(motion, self.freeze_after)
        D = cumulative_displacement(motion)              # (T, 2) (vx, vy)
        Z = self._amp * complex_white(S, S, rng)
        out = np.empty((T, S, S), np.float64)
        for t in range(T):
            if t > 0:
                Z = rho * Z + innov * complex_white(S, S, rng)
            phase = np.exp(-2j * np.pi * (self._ky * D[t, 1] + self._kx * D[t, 0]))
            z = np.real(np.fft.ifft2(Z * phase))
            out[t] = (z - z.mean()) / (z.std() + 1e-8)
        return out, motion, lifetime

    def _to_rain(self, z):
        return np.where(z > self._thr, np.expm1(self.gain * (z - self._thr)), 0.0)

    def generate(self, rng, index):
        T = self.seq_len
        rains, motions, lifetimes = [], [], []
        for _ in range(self.n_layers):
            z, m, lt = self._layer(rng, T)
            rains.append(self._to_rain(z))
            motions.append(m)
            lifetimes.append(lt)
        R = np.max(rains, axis=0) if self.layer_combine == "max" else np.sum(rains, axis=0)
        x = transform_rain(R, self.transform).astype(np.float32)
        motion = np.stack(motions, axis=1)                   # (T, n_layers, 2)
        return self.pack(x, motion, label=0)
