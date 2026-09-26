"""
Two-photon calcium imaging with injected brain motion -- the cross-domain wildcard.

The imaged brain moves (breathing, heartbeat, locomotion): a real, time-varying, near-rigid
translation. Standard pipelines (Suite2p, NoRMCorre) register frames with phase correlation,
rigidly and piecewise-rigidly -- the same design as the transporter and its K slots. The signal,
neurons flashing, is an intrinsic change TRANSVERSE to the motion: Figure 1's setup, in real data.

Recipe: take MOTION-CORRECTED movies, re-apply motion -- either real registration traces
(Suite2p ops['xoff'] / ops['yoff'], NoRMCorre rigid shifts; transplanted between movies) or a
physiological model -- and crop an interior window so shifted content never wraps. Ground-truth
motion is exact, so velocity accuracy is measurable, and the "stabilised" oracle is available.

Unlike the other datasets the velocity changes EVERY frame (respiration is a sinusoid), so this
is the smooth-motion end of the benchmark.

`synthetic_calcium_movie` makes a stand-in movie (cells with calcium transients, neuropil, shot
noise) so the pipeline runs without data; scripts/prepare_calcium.py packs real movies.
"""
import os

import numpy as np

from .base import GeneratedSequenceDataset


def synthetic_calcium_movie(T=1500, H=160, W=160, n_cells=60, fps=30.0, seed=0,
                            cell_radius=(4.0, 7.0), rate_hz=(0.05, 0.6), tau_s=(0.4, 1.0),
                            dff_amp=(0.6, 2.5), noise=0.08):
    """(T, H, W) float32 fluorescence: somata with GCaMP-like transients over neuropil."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    F0 = np.zeros((H, W), np.float32)
    fps = float(fps)
    t = np.arange(T) / fps
    movie = np.zeros((T, H, W), np.float32)
    # neuropil: smooth, slowly fluctuating background
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.fftfreq(W)[None, :]
    lp = np.exp(-2 * (np.pi * 12) ** 2 * (ky ** 2 + kx ** 2))
    base = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((H, W))) * lp))
    base = 0.3 + 0.1 * (base - base.mean()) / (base.std() + 1e-8)
    npl = 1.0 + 0.05 * np.convolve(rng.standard_normal(T), np.ones(int(fps)) / fps, mode="same")
    movie += base[None] * npl[:, None, None].astype(np.float32)
    for _ in range(n_cells):
        cy, cx = rng.uniform(8, H - 8), rng.uniform(8, W - 8)
        r = rng.uniform(*cell_radius)
        ecc = rng.uniform(0.7, 1.0)
        th = rng.uniform(0, np.pi)
        u = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
        v = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
        fp = np.exp(-0.5 * ((u / r) ** 2 + (v / (r * ecc)) ** 2) * 2.5).astype(np.float32)
        brightness = rng.uniform(0.4, 1.0)
        F0 += brightness * fp
        spikes = rng.random(T) < rng.uniform(*rate_hz) / fps
        tau = rng.uniform(*tau_s) * fps
        kern = np.exp(-np.arange(int(6 * tau)) / tau)
        dff = np.convolve(spikes.astype(np.float32) * rng.uniform(*dff_amp), kern)[:T]
        movie += (brightness * (1.0 + dff))[:, None, None].astype(np.float32) * fp[None]
    movie = movie + noise * np.sqrt(np.maximum(movie, 0)) * rng.standard_normal(movie.shape).astype(np.float32)
    return movie.astype(np.float32)


def physiological_motion(T, fps, rng, resp_hz=(1.5, 3.5), resp_amp=(0.4, 1.5),
                         heart_hz=(7.0, 10.0), heart_amp=(0.05, 0.25), drift_amp=0.8,
                         drift_tau_s=20.0, burst_rate_hz=0.03, burst_amp=(1.5, 5.0),
                         burst_dur_s=(0.5, 2.0)):
    """
    Rigid brain displacement D (T, 2) in px, (x, y), with D[0] = 0.

    respiration  sinusoid along a random axis (mostly one axis in real data)
    heartbeat    small, faster sinusoid
    drift        Ornstein-Uhlenbeck, slow
    bursts       locomotion / licking: smooth excursions that return, rare
    """
    t = np.arange(T) / float(fps)
    D = np.zeros((T, 2))
    ax = rng.uniform(0, np.pi)
    d_resp = np.array([np.cos(ax), np.sin(ax)])
    D += np.outer(rng.uniform(*resp_amp) * np.sin(2 * np.pi * rng.uniform(*resp_hz) * t
                                                   + rng.uniform(0, 2 * np.pi)), d_resp)
    ax2 = rng.uniform(0, np.pi)
    D += np.outer(rng.uniform(*heart_amp) * np.sin(2 * np.pi * rng.uniform(*heart_hz) * t),
                  [np.cos(ax2), np.sin(ax2)])
    a = np.exp(-1.0 / (drift_tau_s * fps))
    x = np.zeros(2)
    for i in range(T):
        x = a * x + np.sqrt(1 - a * a) * rng.normal(0, drift_amp, 2)
        D[i] += x
    n_b = rng.poisson(burst_rate_hz * T / fps)
    for _ in range(n_b):
        c = rng.uniform(0, t[-1] if T > 1 else 1.0)
        w = rng.uniform(*burst_dur_s) / 2
        amp = rng.uniform(*burst_amp)
        th = rng.uniform(0, 2 * np.pi)
        prof = np.exp(-0.5 * ((t - c) / w) ** 2)
        D += np.outer(amp * prof, [np.cos(th), np.sin(th)])
    return D - D[0]


class CalciumMotionDataset(GeneratedSequenceDataset):
    """
    movies       list of (T, H, W) motion-corrected movies (arrays or h5py datasets)
    fps          acquisition rate of the movies
    time_stride  keep every n-th frame (e.g. 3: 30 Hz -> 10 Hz, per-step motion ~3x larger)
    traces       None -> physiological_motion(); or a list of (T_i, 2) real rigid offsets (px,
                 (x, y)) that are transplanted: a random segment is re-applied to a random movie
    motion_scale multiply the injected displacement (a velocity-generalisation knob)
    normalize    'zscore' (per movie, robust) | 'none'
    """

    def __init__(self, movies, length, seq_len, image_size=64, fps=30.0, time_stride=3,
                 traces=None, motion_scale=1.0, normalize="zscore", interp_order=3, seed=0,
                 random=True, motion_kw=None):
        super().__init__(length, seed, random)
        self.movies = movies
        self.seq_len = int(seq_len)
        self.S = int(image_size)
        self.fps = float(fps)
        self.stride = int(time_stride)
        self.traces = traces
        self.motion_scale = float(motion_scale)
        self.order = int(interp_order)
        self.motion_kw = dict(motion_kw or {})
        self._norm = []
        for m in movies:
            if normalize == "zscore":
                sample = np.asarray(m[::max(1, len(m) // 50)], dtype=np.float32)
                med = float(np.median(sample))
                mad = float(np.median(np.abs(sample - med))) * 1.4826 + 1e-6
                self._norm.append((med, mad))
            else:
                self._norm.append((0.0, 1.0))
        self.meta = dict(name="calcium", in_channels=1, out_channels=1,
                         channel_names=["fluorescence"], has_motion=True, n_motions=1,
                         periodic=False, pc_window=True, pc_alpha=0.25, pc_search_radius=6,
                         fps=self.fps / self.stride)

    def _trace(self, rng, n):
        span = (n - 1) * self.stride + 1
        if self.traces:
            tr = self.traces[int(rng.integers(len(self.traces)))]
            if len(tr) >= span:
                s = int(rng.integers(len(tr) - span + 1))
                seg = np.asarray(tr[s:s + span], dtype=np.float64)
                return (seg - seg[0])[::self.stride]
        D = physiological_motion(span, self.fps, rng, **self.motion_kw)
        return D[::self.stride]

    def generate(self, rng, index):
        from scipy.ndimage import map_coordinates
        S, T = self.S, self.seq_len
        i = int(rng.integers(len(self.movies)))
        mv = self.movies[i]
        span = (T - 1) * self.stride + 1
        n_frames, H, W = mv.shape
        if span > n_frames:
            raise ValueError(f"movie {i} has {n_frames} frames, need {span}")
        D = self._trace(rng, T) * self.motion_scale                  # (T, 2) (x, y)
        m = int(np.ceil(np.abs(D).max())) + 3
        if S + 2 * m > min(H, W):
            raise ValueError(f"crop {S} + margin {m} does not fit the {H}x{W} movie")
        y0 = int(rng.integers(m, H - S - m + 1))
        x0 = int(rng.integers(m, W - S - m + 1))
        t0 = int(rng.integers(n_frames - span + 1))
        block = np.asarray(mv[t0:t0 + span:self.stride, y0 - m:y0 + S + m, x0 - m:x0 + S + m],
                           dtype=np.float64)
        yy, xx = np.mgrid[0:S, 0:S].astype(np.float64)
        frames = np.empty((T, S, S), np.float32)
        for t in range(T):
            # content moves by +D: sample the source at (p - D)
            frames[t] = map_coordinates(block[t], [(yy + m - D[t, 1]).ravel(),
                                                   (xx + m - D[t, 0]).ravel()],
                                        order=self.order, mode="nearest").reshape(S, S)
        med, mad = self._norm[i]
        frames = (frames - med) / mad
        motion = np.diff(D, axis=0, append=D[-1:] + (D[-1:] - D[-2:-1]))
        return self.pack(frames, motion[:, None, :], label=0)


def load_calcium_archive(path):
    """Prepared archive (scripts/prepare_calcium.py): /movies/<name> (T, H, W) and optional
    /traces/<name> (T, 2). Returns (movies, traces) with movies read lazily per worker."""
    import h5py
    with h5py.File(path, "r") as f:
        traces = ([np.asarray(f["traces"][n]) for n in sorted(f["traces"].keys())]
                  if "traces" in f else None)
        fps = float(f.attrs.get("fps", 30.0))
    return _LazyMovies(path), traces, fps


class _LazyMovies:
    """Pickle-safe list of h5py datasets (re-opened per worker process)."""

    def __init__(self, path):
        self.path = str(path)
        self._pid = None
        import h5py
        with h5py.File(self.path, "r") as f:
            self.names = sorted(f["movies"].keys())
            self.shapes = [f["movies"][n].shape for n in self.names]

    def _open(self):
        import h5py
        if self._pid != os.getpid():
            self._f = h5py.File(self.path, "r")
            self._pid = os.getpid()
        return self._f

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        return self._open()["movies"][self.names[i]]

    def __getstate__(self):
        d = dict(self.__dict__)
        d.pop("_f", None)
        d["_pid"] = None
        return d
