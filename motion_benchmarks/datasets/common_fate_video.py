"""
Common fate on REAL silhouettes: every frame is pure noise, the subject exists only as a
coherently moving region (claude/common_fate_real_video.md).

    frame_t = composite( mask_t filled with texture A translated by D_fg(t),
                         over texture B translated by D_bg(t) )

The mask and its texture share one velocity, so the figure layer is a group orbit of R^2 up to
the articulation of the silhouette itself -- the part outside G. Sub-pixel throughout: textures
move by Fourier shift (exact, band-limited), masks by bilinear sampling with circular wrap.

Construction rules (all measured): both layers move and never with opposite velocities (a static
background leaks the figure through frame differencing, 0.36; v_bg = -v_fg is a 0.90 shortcut);
equal speeds (no per-pixel decorrelation cue); variance-renormalised blending (else the alpha
edge paints the outline into every still frame); figure area >= 25% (the phase-correlation
margin collapses under articulation below that).

Datasets
--------
WeizmannCommonFate   Weizmann "Actions as Space-Time Shapes" silhouettes, 10 actions:
                     bend jack jump pjump run side skip walk wave1 wave2, 9 subjects.
                     Masks: https://www.wisdom.weizmann.ac.il/~vision/VideoAnalysis/Demos/
                     SpaceTimeActions/DB/classification_masks.mat
                     Without the file, `synthetic_walker` stands in (walk / jack / wave).
DeformingCommonFateMNIST  MNIST digits with smooth, periodic articulation of amplitude `amp`
                     px and optional dilation (the figure-area lever): the sweep that measures
                     how far outside G the data can go before R^2 transport stops paying.

Velocity convention: (vx, vy), motion[t] = displacement t -> t+1; motion[:, 0] is the figure,
motion[:, 1] the background.
"""
import re

import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, map_coordinates

from ..common.fields import band_limited_noise
from ..common.shifts import bilinear_shift_np, cumulative_displacement, fourier_shift_np
from .base import GeneratedSequenceDataset

WEIZMANN_ACTIONS = ["bend", "jack", "jump", "pjump", "run", "side", "skip", "walk",
                    "wave1", "wave2"]
WEIZMANN_SUBJECTS = ["daria", "denis", "eli", "ido", "ira", "lena", "lyova", "moshe", "shahar"]


# ============================================================================ loading
def load_weizmann_masks(path, mask_set="original"):
    """
    {sequence_name: (T, H, W) bool}. classification_masks.mat holds one or two structs
    (original / aligned masks), one field per sequence, stored (H, W, T). The layout is
    auto-detected; `mask_set` picks the struct whose name contains it (else the first).
    MATLAB v7.3 files are read with h5py.
    """
    out = {}
    try:
        from scipy.io import loadmat
        raw = loadmat(path, squeeze_me=False, struct_as_record=False)
        structs = {}
        for key, val in raw.items():
            if key.startswith("__"):
                continue
            if isinstance(val, np.ndarray) and val.dtype == object and val.size == 1:
                st = val[0, 0]
                names = getattr(st, "_fieldnames", None)
                if names:
                    structs[key] = {n: np.asarray(getattr(st, n)) for n in names}
            elif isinstance(val, np.ndarray) and val.ndim == 3:
                structs.setdefault("_top", {})[key] = val
        pick = [k for k in structs if mask_set in k.lower()] or list(structs)
        for n, a in structs[pick[0]].items():
            if a.ndim == 3:
                out[n] = np.transpose(a, (2, 0, 1)).astype(bool)
    except NotImplementedError:          # v7.3 -> HDF5
        import h5py
        with h5py.File(path, "r") as f:
            groups = [k for k in f.keys() if not k.startswith("#")]
            pick = [k for k in groups if mask_set in k.lower()] or groups
            g = f[pick[0]]
            for n in g.keys():
                a = np.asarray(g[n])
                if a.ndim == 3:                     # h5py sees MATLAB (H, W, T) as (T, W, H)
                    out[n] = np.transpose(a, (0, 2, 1)).astype(bool)
    if not out:
        raise RuntimeError(f"no 3-D mask arrays found in {path}")
    return out


def parse_weizmann_name(name):
    """'daria_walk' -> ('daria', 'walk'); 'lena_walk1' -> ('lena', 'walk'); wave1/2 kept."""
    name = name.split("/")[-1].lower()
    subj, _, act = name.partition("_")
    if act not in ("wave1", "wave2"):
        act = re.sub(r"\d+$", "", act)
    return subj, act


def synthetic_walker(T=60, H=144, W=180, action="walk", seed=0):
    """Stand-in silhouettes (filled, articulated): walk / jack / wave. NOT the dataset."""
    rng = np.random.default_rng(seed)
    Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
    masks = np.zeros((T, H, W), bool)
    cy, cx = H * 0.52, W * 0.5
    period = rng.uniform(10, 14)
    phase0 = rng.uniform(0, 2 * np.pi)

    def blob(m, y0, x0, ry, rx, ang=0.0):
        c, s = np.cos(ang), np.sin(ang)
        yy, xx = (Y - y0), (X - x0)
        u, v = c * yy + s * xx, -s * yy + c * xx
        return m | ((u / ry) ** 2 + (v / rx) ** 2 <= 1.0)

    for t in range(T):
        ph = 2 * np.pi * t / period + phase0
        m = np.zeros((H, W), bool)
        m = blob(m, cy - 34, cx, 11, 9)
        m = blob(m, cy - 8, cx, 22, 11)
        if action == "walk":
            la, ra = 0.35 * np.sin(ph), 0.35 * np.sin(ph + np.pi)
            m = blob(m, cy + 24, cx + 7 * np.sin(ph), 20, 6, la)
            m = blob(m, cy + 24, cx + 7 * np.sin(ph + np.pi), 20, 6, ra)
            m = blob(m, cy - 10, cx - 9 * np.sin(ph), 16, 5, -la)
            m = blob(m, cy - 10, cx + 9 * np.sin(ph), 16, 5, -ra)
        elif action == "jack":
            a = 0.5 + 0.5 * np.sin(ph)
            m = blob(m, cy + 24, cx - 12 * a, 20, 6, -0.5 * a)
            m = blob(m, cy + 24, cx + 12 * a, 20, 6, 0.5 * a)
            m = blob(m, cy - 22 - 14 * a, cx - 20 * a, 16, 5, 1.1 * a)
            m = blob(m, cy - 22 - 14 * a, cx + 20 * a, 16, 5, -1.1 * a)
        else:  # wave
            a = np.sin(ph)
            m = blob(m, cy + 24, cx - 6, 20, 6)
            m = blob(m, cy + 24, cx + 6, 20, 6)
            m = blob(m, cy - 10, cx - 14, 16, 5, 0.3)
            m = blob(m, cy - 26 - 8 * a, cx + 16 + 4 * a, 16, 5, -1.0 - 0.4 * a)
        masks[t] = binary_fill_holes(m)
    return masks


def crop_and_resize(masks, out=64, target_area=0.30):
    """
    Square crop around the subject (union bbox over time), resized so the silhouette covers
    ~target_area of the frame (bisection on the crop half-width). Returns (T, out, out) float.
    """
    T = len(masks)
    ys, xs = np.where(masks.any(0))
    if len(ys) == 0:
        raise ValueError("empty mask sequence")
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) / 2 + 1

    def render(h, ts=None):
        gy = np.linspace(cy - h, cy + h, out)
        gx = np.linspace(cx - h, cx + h, out)
        G = np.stack(np.meshgrid(gy, gx, indexing="ij"))
        idx = range(T) if ts is None else ts
        return np.stack([map_coordinates(masks[t].astype(np.float32), [G[0].ravel(), G[1].ravel()],
                                         order=1, mode="constant").reshape(out, out) for t in idx])

    probe = list(range(0, T, max(1, T // 8)))
    lo, hi = half * 0.5, half * 4.0
    for _ in range(22):
        h = 0.5 * (lo + hi)
        if (render(h, probe) > 0.5).mean() > target_area:
            lo = h
        else:
            hi = h
    return np.clip(render(0.5 * (lo + hi)), 0, 1).astype(np.float32)


# ============================================================================ rendering
def two_velocities(T, speed, rng, time_varying=True, hold=(3, 7), min_sep=1.5, max_tries=200):
    """Figure and background velocities (T, 2) each: equal speed, random directions, never
    closer than min_sep and never opposite (|v_fg + v_bg| >= min_sep), never zero."""
    def schedule():
        out, t = [], 0
        while t < T:
            th = rng.uniform(0, 2 * np.pi)
            v = speed * np.array([np.cos(th), np.sin(th)])
            k = int(rng.integers(hold[0], hold[1] + 1)) if time_varying else T
            out += [v] * k
            t += k
        return np.array(out[:T])
    for _ in range(max_tries):
        a, b = schedule(), schedule()
        if (np.linalg.norm(a - b, axis=1).min() >= min_sep
                and np.linalg.norm(a + b, axis=1).min() >= min_sep):
            return a, b
    return a, b


def render_common_fate(masks, speed=2.0, corr_len=1.0, time_varying=True, renormalise=True,
                       rng=None, velocities=None):
    """
    masks (T, H, W) in [0, 1] -> dict(frames (T, H, W), masks (T, H, W) as rendered,
    motion (T, 2, 2) [figure, background] in (vx, vy)).
    """
    rng = rng or np.random.default_rng()
    T, H, W = masks.shape
    A = band_limited_noise(H, W, corr_len, rng)
    B = band_limited_noise(H, W, corr_len, rng)
    v_fg, v_bg = velocities if velocities is not None else two_velocities(T, speed, rng,
                                                                          time_varying)
    D_fg, D_bg = cumulative_displacement(v_fg), cumulative_displacement(v_bg)
    frames = np.empty((T, H, W), np.float32)
    shown = np.empty((T, H, W), np.float32)
    for t in range(T):
        al = np.clip(bilinear_shift_np(masks[t], D_fg[t]), 0, 1)
        shown[t] = al
        f = al * fourier_shift_np(A, D_fg[t]) + (1 - al) * fourier_shift_np(B, D_bg[t])
        if renormalise:
            f = f / np.sqrt(al ** 2 + (1 - al) ** 2 + 1e-8)
        frames[t] = f
    return dict(frames=frames, masks=shown, motion=np.stack([v_fg, v_bg], axis=1))


class WeizmannCommonFate(GeneratedSequenceDataset):
    """
    sequences : list of (masks (T, S, S) float in [0, 1] already cropped, label int)
    Each item: a random clip of seq_len frames (temporal stride `stride`) of a random sequence,
    rendered with fresh textures and velocities -- the textures change every draw, so a model
    cannot memorise appearance; only the motion-defined shape carries the label.
    """

    def __init__(self, sequences, length, seq_len, speed=2.0, corr_len=1.0, time_varying=True,
                 renormalise=True, stride=1, return_masks=False, seed=0, random=True,
                 shuffle_time=False):
        super().__init__(length, seed, random)
        self.seqs = sequences
        self.seq_len = int(seq_len)
        self.speed = float(speed)
        self.corr_len = float(corr_len)
        self.time_varying = bool(time_varying)
        self.renormalise = bool(renormalise)
        self.stride = int(stride)
        self.return_masks = return_masks
        self.shuffle_time = shuffle_time
        span = (self.seq_len - 1) * self.stride + 1
        short = [len(m) for m, _ in sequences if len(m) < span]
        if short:
            raise ValueError(f"{len(short)} sequences shorter than seq_len*stride = {span}")
        self._span = span
        self.meta = dict(name="weizmann_common_fate", in_channels=1, n_motions=2,
                         has_motion=True, periodic=True, n_classes=len(WEIZMANN_ACTIONS))

    def generate(self, rng, index):
        masks, label = self.seqs[int(rng.integers(len(self.seqs)))]
        t0 = int(rng.integers(len(masks) - self._span + 1))
        clip = masks[t0:t0 + self._span:self.stride]
        if rng.random() < 0.5:                                  # mirror: same action
            clip = clip[:, :, ::-1]
        r = render_common_fate(np.ascontiguousarray(clip), self.speed, self.corr_len,
                               self.time_varying, self.renormalise, rng)
        frames, motion = r["frames"], r["motion"]
        if self.shuffle_time:                                   # control: destroys common fate
            perm = rng.permutation(len(frames))
            frames, motion = frames[perm], motion[perm]
        out = self.pack(frames, motion, label=label)
        if self.return_masks:
            return out + (r["masks"],)
        return out


def build_weizmann_sequences(mat_path=None, image_size=64, target_area=0.30, mask_set="original",
                             subjects=None, n_synthetic=12, seed=0):
    """
    -> list of (masks (T, S, S), label, subject). Without mat_path: synthetic stand-ins
    (labels walk / jack / wave mapped onto the Weizmann indices of walk / jack / wave1).
    """
    out = []
    if mat_path:
        for name, m in sorted(load_weizmann_masks(mat_path, mask_set).items()):
            subj, act = parse_weizmann_name(name)
            if act not in WEIZMANN_ACTIONS:
                continue
            if subjects is not None and subj not in subjects:
                continue
            out.append((crop_and_resize(m, image_size, target_area),
                        WEIZMANN_ACTIONS.index(act), subj))
        return out
    lab = {"walk": WEIZMANN_ACTIONS.index("walk"), "jack": WEIZMANN_ACTIONS.index("jack"),
           "wave": WEIZMANN_ACTIONS.index("wave1")}
    for i in range(n_synthetic):
        act = ["walk", "jack", "wave"][i % 3]
        subj = f"synth{i // 3}"
        if subjects is not None and subj not in subjects:
            continue
        m = synthetic_walker(T=60, action=act, seed=seed + i)
        out.append((crop_and_resize(m, image_size, target_area), lab[act], subj))
    return out


# ============================================================================ deforming MNIST
def smooth_field(H, W, corr, rng):
    z = band_limited_noise(H, W, corr, rng)
    return z / (z.std() + 1e-8)


def _deform(mask, dy, dx):
    H, W = mask.shape
    Y, X = np.mgrid[0:H, 0:W]
    return map_coordinates(mask, [((Y + dy) % H).ravel(), ((X + dx) % W).ravel()],
                           order=1, mode="wrap").reshape(H, W)


class DeformingCommonFateMNIST(GeneratedSequenceDataset):
    """
    CF-MNIST whose digit ARTICULATES: two smooth displacement fields oscillated in time
    (period `period` frames, amplitude `amp` px) deform the mask inside the figure's translating
    frame. amp = 0 is rigid CF-MNIST. `dilate` grows the stroke (figure-area lever: thin MNIST
    strokes are ~11% of the canvas; >= 25% keeps the tracker locked under articulation).
    `digits`: (N, 28, 28) uint8/float glyphs, `labels`: (N,).
    """

    def __init__(self, digits, labels, length, seq_len, image_size=64, amp=0.0, period=8,
                 defo_corr=6.0, scale=2, dilate=0, speed=2.0, corr_len=1.0, time_varying=True,
                 seed=0, random=True):
        super().__init__(length, seed, random)
        self.digits = digits
        self.labels = np.asarray(labels)
        self.seq_len = int(seq_len)
        self.S = int(image_size)
        self.amp, self.period, self.defo_corr = float(amp), float(period), float(defo_corr)
        self.scale, self.dilate = int(scale), int(dilate)
        self.speed, self.corr_len, self.time_varying = float(speed), float(corr_len), time_varying
        self.meta = dict(name="deforming_cf_mnist", in_channels=1, n_motions=2, has_motion=True,
                         periodic=True, n_classes=10)

    def _mask(self, glyph, rng):
        d = np.asarray(glyph, dtype=np.float32)
        d = d / 255.0 if d.max() > 1.5 else d
        if self.scale != 1:
            d = np.kron(d, np.ones((self.scale, self.scale), np.float32))
        m = d > 0.3
        if self.dilate:
            m = binary_dilation(m, iterations=self.dilate)
        S = self.S
        canvas = np.zeros((S, S), np.float32)
        h, w = m.shape
        canvas[:h, :w] = m
        return np.roll(canvas, (int(rng.integers(S)), int(rng.integers(S))), axis=(0, 1))

    def generate(self, rng, index):
        i = int(rng.integers(len(self.digits)))
        S, T = self.S, self.seq_len
        m0 = self._mask(self.digits[i], rng)
        fy, fx = smooth_field(S, S, self.defo_corr, rng), smooth_field(S, S, self.defo_corr, rng)
        ph0 = rng.uniform(0, 2 * np.pi)
        masks = np.empty((T, S, S), np.float32)
        for t in range(T):
            ph = 2 * np.pi * t / self.period + ph0
            masks[t] = (_deform(m0, self.amp * np.sin(ph) * fy, self.amp * np.cos(ph) * fx)
                        if self.amp > 0 else m0)
        r = render_common_fate(masks, self.speed, self.corr_len, self.time_varying, True, rng)
        return self.pack(r["frames"], r["motion"], label=int(self.labels[i]))
