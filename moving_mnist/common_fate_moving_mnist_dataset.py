"""
Common-Fate Moving MNIST: generator + PyTorch Dataset.

The figure is defined by MOTION, not by intensity. Both the figure and the
background carry the same band-limited noise statistics, so no single frame
contains the digit -- a per-frame model cannot see it at all. Only the
difference between two velocities segments the scene.

Two variants, and the difference is decisive for MEConvLSTM:

  variant='moving_mask'  (figure translates; mask AND its texture move at v_fg)
      layer_fg(t) = roll(m * A, d_fg(t))                  <- a PURE translation
      layer_bg(t) = roll(1-m, d_fg(t)) * roll(B, d_bg(t)) <- aperture at v_fg, texture at v_bg
      In the frame co-moving at v_fg the figure is STATIC -> transport recovers
      the SHAPE.

  variant='static_mask'  (classic "motion-defined form"; mask fixed, textures scroll)
      frame(t) = m * roll(A, d_fg(t)) + (1-m) * roll(B, d_bg(t))
      No global translation makes the figure static -> transport recovers
      TEXTURE, not shape.

Everything lives on a torus with integer velocities, so roll() is the exact
group action -- the same setting the rest of this repo's datasets use, which
is what makes a warp-based model exactly equivariant here rather than
approximately.

The measurement tools that go with this data (phase correlation, co-moving
accumulation, the residual bootstrap that pulls the minority motion out from
under the dominant one) live in `common_fate_diagnostics.py`.

Velocity conventions -- READ THIS
---------------------------------
The array-level helpers below (`velocity_schedule`, `make_sequence`) speak
NUMPY AXIS ORDER: a velocity is (dy, dx), matching `np.roll(x, v, axis=(0, 1))`.

`CommonFateMovingMNISTDataset` returns motion in the REPO convention used by
TDMovingMNISTDataset and the velocity heads: (vx, vy), i.e. index 0 is the
horizontal component. The conversion happens in one place, `_yx_to_xy`. Mixing
the two silently transposes every velocity, so the two worlds are kept apart
deliberately.
"""
import warnings

import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision.datasets import MNIST


# ----------------------------------------------------------------------------- data
def band_limited_noise(H, W, corr_len, rng):
    """Noise with a given correlation length, unit variance, zero mean."""
    z = rng.standard_normal((H, W))
    if corr_len > 0:
        ky = np.fft.fftfreq(H)[:, None]
        kx = np.fft.fftfreq(W)[None, :]
        k2 = ky ** 2 + kx ** 2
        z = np.real(np.fft.ifft2(np.fft.fft2(z) * np.exp(-2 * (np.pi * corr_len) ** 2 * k2)))
    return (z - z.mean()) / (z.std() + 1e-8)


def digit_mask(digit28, H, W, rng, thresh=0.3, scale=1):
    """
    Binary mask of the digit, randomly placed on the HxW torus.

    `digit28` is the raw MNIST glyph. Integer dtypes are taken as 0..255 and
    divided through; floating dtypes are taken as ALREADY in [0, 1]. Getting
    this wrong is silent -- a uint8 glyph read as [0, 1] thresholds to an
    all-ones mask, a float glyph read as 0..255 to an all-zeros one -- so the
    branch is explicit rather than inferred from the value range.
    """
    d = np.asarray(digit28)
    d = d.astype(np.float32) if np.issubdtype(d.dtype, np.floating) \
        else d.astype(np.float32) / 255.0

    if scale != 1:
        d = np.kron(d, np.ones((scale, scale), np.float32))

    h, w = d.shape
    if h > H or w > W:
        raise ValueError(
            f"digit is {h}x{w} after scale={scale} but the canvas is {H}x{W}; "
            f"raise image_size or lower digit_scale."
        )

    m = np.zeros((H, W), np.float32)
    m[:h, :w] = (d > thresh).astype(np.float32)
    return np.roll(m, (rng.integers(H), rng.integers(W)), axis=(0, 1))


def velocity_schedule(T, vmax, rng, time_varying=False, tau_lo=3, tau_hi=6):
    """
    Integer velocities in numpy axis order (dy, dx); piecewise-constant when
    time_varying. (0, 0) is off the grid, so a layer never stops -- a stopped
    layer has no common fate to share.
    """
    def draw():
        while True:
            v = rng.integers(-vmax, vmax + 1, size=2)
            if np.any(v):
                return v
    if not time_varying:
        return np.tile(draw(), (T, 1))
    out, t = [], 0
    while t < T:
        v, hold = draw(), rng.integers(tau_lo, tau_hi + 1)
        out += [v] * int(hold)
        t += hold
    return np.array(out[:T])


def make_sequence(digit28, T=20, H=64, W=64, vmax=3, corr_len=1.0, scale=1,
                  variant="moving_mask", time_varying=False, min_dv=2,
                  thresh=0.3, max_vel_tries=200, rng=None):
    """
    One common-fate sequence.

    Returns a dict with the frames (T, H, W) float32, the mask, the two
    velocity schedules and their cumulative displacements -- all in numpy axis
    order (dy, dx) -- plus `velocity_separated`, which says whether the
    |v_fg - v_bg| >= min_dv rejection actually succeeded within max_vel_tries.

    That flag matters: the loop falls back to the last draw rather than
    raising, so without it a too-aggressive min_dv (near 2*vmax, or min_dv >= 3
    with time_varying=True, where every one of the T steps must satisfy the
    constraint) would quietly seed the set with sequences whose two layers
    share a velocity and therefore contain no figure at all.
    """
    rng = rng or np.random.default_rng()
    m = digit_mask(digit28, H, W, rng, thresh=thresh, scale=scale)
    A = band_limited_noise(H, W, corr_len, rng)
    B = band_limited_noise(H, W, corr_len, rng)

    separated = False
    for _ in range(max_vel_tries):            # separate the two velocities
        v_fg = velocity_schedule(T, vmax, rng, time_varying)
        v_bg = velocity_schedule(T, vmax, rng, time_varying)
        if np.min(np.abs(v_fg - v_bg).max(axis=1)) >= min_dv:
            separated = True
            break
    d_fg = np.cumsum(v_fg, 0)
    d_bg = np.cumsum(v_bg, 0)

    frames = np.empty((T, H, W), np.float32)
    for t in range(T):
        if variant == "moving_mask":
            mt = np.roll(m, tuple(d_fg[t]), (0, 1))
            frames[t] = np.where(mt > 0.5,
                                 np.roll(A, tuple(d_fg[t]), (0, 1)),
                                 np.roll(B, tuple(d_bg[t]), (0, 1)))
        elif variant == "static_mask":
            frames[t] = np.where(m > 0.5,
                                 np.roll(A, tuple(d_fg[t]), (0, 1)),
                                 np.roll(B, tuple(d_bg[t]), (0, 1)))
        else:
            raise ValueError(variant)
    return dict(frames=frames, mask=m, v_fg=v_fg, v_bg=v_bg, d_fg=d_fg, d_bg=d_bg,
                velocity_separated=separated)


# ----------------------------------------------------------------------- conventions
def _yx_to_xy(v):
    """(..., 2) in numpy axis order (dy, dx) -> repo order (vx, vy)."""
    return np.ascontiguousarray(np.asarray(v)[..., ::-1])


# --------------------------------------------------------------------------- dataset
class CommonFateMovingMNISTDataset(Dataset):
    """
    PyTorch Dataset over `make_sequence`, shaped like TDMovingMNISTDataset so
    the same training loop can consume it.

    Yields, in this order:
        seq     (T, 1, H, W) float32
        label   int (the MNIST digit class defining the mask)
        motion  (T, 2, 2) long   -- only if return_motion   [see below]
        mask    (T, H, W) float32 -- only if return_mask    [see below]

    motion follows the (T, N, 2) layout the velocity heads already expect, with
    the two "slots" being the two LAYERS rather than two digits:

        motion[t, 0] = (vx, vy) of the FIGURE at step t
        motion[t, 1] = (vx, vy) of the BACKGROUND at step t

    mask is returned per-frame because in 'moving_mask' the mask travels: it is
    roll(m, d_fg[t]), i.e. where the figure actually is in frame t. In
    'static_mask' every frame carries the same m; the time axis is kept anyway
    so the two variants have one interface.

    Parameters that differ from TDMovingMNISTDataset
    -----------------------------------------------
    variant       : 'moving_mask' | 'static_mask'  -- see the module docstring.
                    This is the experimental variable; everything else is
                    nuisance.
    corr_len      : correlation length of both textures, in pixels. 0 gives
                    white noise, whose phase-correlation peak is sharp but
                    whose aperture problem is trivial; ~1 gives a texture with
                    real spatial structure. Both layers share it, so texture
                    statistics cannot leak the figure.
    digit_scale   : integer upscaling of the 28x28 glyph via pixel replication.
    mask_threshold: glyph intensity above which a pixel belongs to the figure.
    time_varying  : piecewise-constant velocities instead of constant ones.
    min_dv        : required separation between the two velocities, in the
                    max-norm, at EVERY step. Below 1 the layers can move
                    together and the figure vanishes.
    normalize     : how the zero-mean unit-variance textures are mapped into
                    the [0, 1] range the models and the BCE/MSE losses expect.
                    'affine' (default) clips at +-`clip` sigma and rescales,
                    which is a FIXED map -- contrast means the same thing in
                    every sequence. 'minmax' rescales per sequence, which makes
                    contrast sequence-dependent and lets the extremes leak
                    information; 'none' hands back raw sigma units.

    Randomness
    ----------
    random=True  : a fresh OS-seeded Generator per item. This is deliberate --
                   a seeded Generator held on the dataset would be COPIED into
                   every DataLoader worker on fork, so k workers would emit the
                   same k-fold-duplicated stream.
    random=False : one stateful seeded Generator, advanced per item, with
                   reset_rng() to rewind -- the fixed-benchmark contract the
                   rest of the repo uses. Load these with
                   persistent_workers=False and call reset_rng() before each
                   pass, exactly as train.py already does for its fixed sets.
    """

    def __init__(
        self,
        root,
        train=True,
        seq_len=20,
        image_size=64,
        num_digits=1,

        variant="moving_mask",
        max_speed=3,
        corr_len=1.0,
        digit_scale=1,
        mask_threshold=0.3,
        time_varying=False,
        min_dv=2,

        normalize="affine",
        clip=3.0,

        return_motion=True,
        return_mask=False,

        transform=None,
        download=True,

        random=True,
        seed=42,

        max_velocity_tries=200,
    ):
        super().__init__()

        if variant not in ("moving_mask", "static_mask"):
            raise ValueError(
                f"variant must be 'moving_mask' or 'static_mask', got {variant!r}")
        if normalize not in ("affine", "minmax", "none"):
            raise ValueError(
                f"normalize must be 'affine', 'minmax' or 'none', got {normalize!r}")
        if num_digits != 1:
            # Two figures at two velocities is a different experiment (it asks
            # about grouping, not about common fate) and the generator has no
            # occlusion order for overlapping masks. Refuse rather than render
            # something that looks right and is not.
            raise ValueError(
                "CommonFateMovingMNISTDataset renders a single figure against a "
                "single background; num_digits must be 1.")
        if min_dv > 2 * max_speed:
            raise ValueError(
                f"min_dv={min_dv} is unsatisfiable with max_speed={max_speed} "
                f"(the largest possible separation is {2 * max_speed}).")

        self.mnist = MNIST(root=root, train=train, download=download)

        self.seq_len    = seq_len
        self.image_size = image_size
        self.num_digits = num_digits

        self.variant        = variant
        self.max_speed      = max_speed
        self.corr_len       = corr_len
        self.digit_scale    = digit_scale
        self.mask_threshold = mask_threshold
        self.time_varying   = time_varying
        self.min_dv         = min_dv

        self.normalize = normalize
        self.clip      = float(clip)

        self.return_motion = return_motion
        self.return_mask   = return_mask

        self.transform = transform
        self.random    = random
        self.max_velocity_tries = max_velocity_tries

        self.seed = seed
        self.rng  = np.random.default_rng(seed)

        # One warning per dataset, not one per sample: a failing rejection is a
        # property of the configuration, so the first hit says everything.
        self._warned_unseparated = False

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def reset_rng(self):
        """
        Rewind the private Generator to its initial seed. With random=False the
        RNG is stateful (every __getitem__ advances it), so a fixed benchmark
        set requires resetting before each full pass -- otherwise successive
        evaluations silently see different sequences.
        """
        self.rng = np.random.default_rng(self.seed)

    def __len__(self):
        return len(self.mnist)

    def _gen(self):
        return np.random.default_rng() if self.random else self.rng

    def __getitem__(self, index):
        rng = self._gen()

        idx = int(rng.integers(len(self.mnist)))
        img_pil, label = self.mnist[idx]
        glyph = np.asarray(img_pil, dtype=np.uint8)

        out = make_sequence(
            glyph,
            T=self.seq_len,
            H=self.image_size,
            W=self.image_size,
            vmax=self.max_speed,
            corr_len=self.corr_len,
            scale=self.digit_scale,
            variant=self.variant,
            time_varying=self.time_varying,
            min_dv=self.min_dv,
            thresh=self.mask_threshold,
            max_vel_tries=self.max_velocity_tries,
            rng=rng,
        )

        if not out["velocity_separated"] and not self._warned_unseparated:
            self._warned_unseparated = True
            warnings.warn(
                f"could not separate the two velocities by min_dv={self.min_dv} "
                f"within {self.max_velocity_tries} tries "
                f"(max_speed={self.max_speed}, time_varying={self.time_varying}); "
                f"such sequences contain little or no motion-defined figure. "
                f"Lower min_dv, raise max_speed, or raise max_velocity_tries.",
                UserWarning, stacklevel=2)

        seq = torch.from_numpy(self._normalize(out["frames"])).unsqueeze(1)  # (T,1,H,W)

        if self.transform:
            seq = self.transform(seq)

        result = [seq, label]

        if self.return_motion:
            motion = np.stack([_yx_to_xy(out["v_fg"]), _yx_to_xy(out["v_bg"])], axis=1)
            result.append(torch.from_numpy(motion).long())          # (T, 2, 2)

        if self.return_mask:
            result.append(torch.from_numpy(self._mask_track(out)))  # (T, H, W)

        return tuple(result)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize(self, frames):
        """
        Textures come out zero-mean unit-variance; the models want [0, 1].

        'affine' is the default because it is the only FIXED map here: the same
        sigma always lands on the same grey level, so contrast is not a free
        variable across the set. Clipping at 3 sigma touches ~0.3% of pixels.
        """
        if self.normalize == "none":
            return frames.astype(np.float32)

        if self.normalize == "affine":
            c = self.clip
            return (np.clip(frames, -c, c) / (2 * c) + 0.5).astype(np.float32)

        lo, hi = float(frames.min()), float(frames.max())
        return ((frames - lo) / (hi - lo + 1e-8)).astype(np.float32)

    def _mask_track(self, out):
        """
        Where the figure is in each frame.

        In 'moving_mask' the mask travels with the figure, so the target is
        roll(m, d_fg[t]); in 'static_mask' the mask is fixed by construction and
        the same m is repeated. Returned for both so a segmentation probe has
        one interface.
        """
        m, T = out["mask"], self.seq_len
        if self.variant == "static_mask":
            return np.repeat(m[None], T, axis=0).astype(np.float32)
        return np.stack(
            [np.roll(m, tuple(out["d_fg"][t]), (0, 1)) for t in range(T)]
        ).astype(np.float32)
