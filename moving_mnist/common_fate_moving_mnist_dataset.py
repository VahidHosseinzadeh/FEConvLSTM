"""
Common-Fate Moving MNIST: generator + PyTorch Dataset.

The figure is defined by MOTION, not by intensity. Both the figures and the
background carry the same band-limited noise statistics, so no single frame
contains a digit -- a per-frame model cannot see it at all. Only a difference
between velocities segments the scene.

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

Two levels of API
-----------------
The array helpers (`band_limited_noise`, `digit_mask`, `velocity_schedule`,
`render_sequence`, `make_sequence`) are plain numpy and stand alone -- use them
for one-off sequences and diagnostics.

`CommonFateMovingMNISTDataset` subclasses `TDMovingMNISTDataset`, so it inherits
that class's ENTIRE motion vocabulary -- constant / piecewise / stochastic /
accelerate, transition_mode, motion_difficulty, neighbor_kernel, freeze_after --
and applies it to textured layers instead of intensity digits. Reach for it
whenever the motion needs to be anything richer than a constant velocity, or
whenever more than one figure is in play.

Velocity conventions -- READ THIS
---------------------------------
The array helpers speak NUMPY AXIS ORDER: a velocity is (dy, dx), matching
`np.roll(x, v, axis=(0, 1))`.

`CommonFateMovingMNISTDataset` speaks the REPO convention used by
TDMovingMNISTDataset and the velocity heads: (vx, vy), i.e. index 0 is the
horizontal component. The conversion happens in `_xy_to_yx` / `_yx_to_xy` and
nowhere else. Mixing the two silently transposes every velocity.

Displacement indexing follows TDMovingMNISTDataset exactly:

    displacement[0] = 0,   displacement[t] = sum(motion[0 .. t-1])

so `motion[t]` is the step taking frame t to frame t+1. Getting this wrong is
invisible under a constant velocity and shifts every supervision target by one
frame the moment the velocity varies.
"""
import warnings

import numpy as np
import torch

from time_dependent_moving_mnist_dataset import TDMovingMNISTDataset


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


def glyph_to_mask(glyph, scale=1, thresh=0.3):
    """
    Raw MNIST glyph -> binary (h*scale, w*scale) mask, still at its own size.

    Integer dtypes are taken as 0..255 and divided through; floating dtypes are
    taken as ALREADY in [0, 1]. Getting this wrong is silent -- a uint8 glyph
    read as [0, 1] thresholds to all ones, a float glyph read as 0..255 to all
    zeros -- so the branch is explicit rather than inferred from the range.
    """
    d = np.asarray(glyph)
    d = d.astype(np.float32) if np.issubdtype(d.dtype, np.floating) \
        else d.astype(np.float32) / 255.0
    if scale != 1:
        d = np.kron(d, np.ones((scale, scale), np.float32))
    return (d > thresh).astype(np.float32)


def digit_mask(digit28, H, W, rng, thresh=0.3, scale=1):
    """Binary mask of the digit, randomly placed on the HxW torus."""
    d = glyph_to_mask(digit28, scale=scale, thresh=thresh)
    h, w = d.shape
    if h > H or w > W:
        raise ValueError(
            f"digit is {h}x{w} after scale={scale} but the canvas is {H}x{W}; "
            f"raise image_size or lower digit_scale.")
    m = np.zeros((H, W), np.float32)
    m[:h, :w] = d
    return np.roll(m, (rng.integers(H), rng.integers(W)), axis=(0, 1))


def velocity_schedule(T, vmax, rng, time_varying=False, tau_lo=3, tau_hi=6):
    """
    Integer velocities in numpy axis order (dy, dx); piecewise-constant when
    time_varying. (0, 0) is off the grid, so a layer never stops -- a stopped
    layer has no common fate to share.

    This is the standalone helper. `CommonFateMovingMNISTDataset` ignores it and
    uses TDMovingMNISTDataset's far richer motion machinery instead.
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


def cumulative_displacement(v):
    """
    Velocities (T, ..., 2) -> displacements, with displacement[0] = 0 and
    displacement[t] = sum(v[0 .. t-1]).

    This is TDMovingMNISTDataset's convention, and it is the reason v[t] means
    "the step taking frame t to frame t+1". The naive cumsum(v) puts frame 0 at
    v[0] instead, which makes v[t] the step from t-1 to t -- an off-by-one that
    a constant velocity hides completely.
    """
    v = np.asarray(v)
    return np.concatenate([np.zeros((1,) + v.shape[1:], v.dtype),
                           np.cumsum(v, axis=0)[:-1]], axis=0)


def render_sequence(masks, textures, bg_texture, d_fig, d_bg, variant):
    """
    Composite the layered scene.

    masks      : (N, H, W) binary, each figure at its starting position
    textures   : (N, H, W) one texture per figure
    bg_texture : (H, W)
    d_fig      : (T, N, 2) cumulative displacements, numpy order (dy, dx)
    d_bg       : (T, 2)
    variant    : 'moving_mask' | 'static_mask'

    Returns frames (T, H, W) float32 and mask_track (T, N, H, W) float32, the
    latter being where each figure actually is in each frame.

    Painting order is background, then figure 0, 1, ... so the HIGHEST index
    occludes. With one figure the order is irrelevant; with several it must be
    fixed and stated, or overlapping figures render differently run to run.
    """
    masks = np.asarray(masks, np.float32)
    textures = np.asarray(textures, np.float32)
    N, H, W = masks.shape
    T = len(d_bg)
    if variant not in ("moving_mask", "static_mask"):
        raise ValueError(variant)

    frames = np.empty((T, H, W), np.float32)
    track = np.empty((T, N, H, W), np.float32)
    for t in range(T):
        frame = np.roll(bg_texture, tuple(d_bg[t]), (0, 1))
        for i in range(N):
            tex = np.roll(textures[i], tuple(d_fig[t, i]), (0, 1))
            # moving_mask: the aperture travels with its texture, so the figure
            # is static in the co-moving frame. static_mask: the aperture is
            # pinned and only the texture scrolls through it.
            mt = (np.roll(masks[i], tuple(d_fig[t, i]), (0, 1))
                  if variant == "moving_mask" else masks[i])
            frame = np.where(mt > 0.5, tex, frame)
            track[t, i] = mt
        frames[t] = frame
    return frames, track


def make_sequence(digit28, T=20, H=64, W=64, vmax=3, corr_len=1.0, scale=1,
                  variant="moving_mask", time_varying=False, min_dv=2,
                  thresh=0.3, max_vel_tries=200, rng=None):
    """
    One single-figure common-fate sequence, self-contained numpy.

    Returns a dict with the frames (T, H, W) float32, the mask, the two velocity
    schedules and their cumulative displacements -- all in numpy axis order
    (dy, dx) -- plus `velocity_separated`, which says whether the
    |v_fg - v_bg| >= min_dv rejection actually succeeded within max_vel_tries.

    That flag matters: the loop falls back to the last draw rather than raising,
    so without it a too-aggressive min_dv (near 2*vmax, or min_dv >= 3 with
    time_varying=True, where every one of the T steps must satisfy the
    constraint) would quietly seed the set with sequences whose two layers share
    a velocity and therefore contain no figure at all.

    For anything richer than constant or piecewise motion, or for more than one
    figure, use CommonFateMovingMNISTDataset instead.
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

    d_fg = cumulative_displacement(v_fg)
    d_bg = cumulative_displacement(v_bg)

    frames, track = render_sequence(m[None], A[None], B,
                                    d_fig=d_fg[:, None], d_bg=d_bg,
                                    variant=variant)
    return dict(frames=frames, mask=m, mask_track=track[:, 0],
                v_fg=v_fg, v_bg=v_bg, d_fg=d_fg, d_bg=d_bg,
                velocity_separated=separated)


# ----------------------------------------------------------------------- conventions
def _yx_to_xy(v):
    """(..., 2) numpy axis order (dy, dx) -> repo order (vx, vy)."""
    return np.ascontiguousarray(np.asarray(v)[..., ::-1])


def _xy_to_yx(v):
    """(..., 2) repo order (vx, vy) -> numpy axis order (dy, dx)."""
    return np.ascontiguousarray(np.asarray(v)[..., ::-1])


class _ParentRNG:
    """
    The np.random.Generator surface the array helpers expect, backed by
    TDMovingMNISTDataset's own RNG discipline.

    The parent draws through _randint/_choice/_random, which dispatch to the
    global np.random when random=True and to a seeded RandomState when
    random=False. The helpers here want a Generator (`standard_normal`,
    `integers`). Adapting rather than holding a second, independent Generator is
    what keeps `reset_rng()` and the fixed-benchmark contract meaningful -- two
    RNGs would mean resetting one and silently advancing the other.
    """

    def __init__(self, ds):
        self._ds = ds

    def _src(self):
        return np.random if self._ds.random else self._ds.rng

    def standard_normal(self, size=None):
        return self._src().standard_normal(size)

    def integers(self, low, high=None, size=None):
        lo, hi = (0, low) if high is None else (low, high)
        return self._src().randint(lo, hi, size=size)


# --------------------------------------------------------------------------- dataset
class CommonFateMovingMNISTDataset(TDMovingMNISTDataset):
    """
    Common-fate sequences with TDMovingMNISTDataset's full motion vocabulary.

    Subclassing buys the entire motion family -- motion_mode of constant /
    piecewise / stochastic / accelerate, transition_mode, neighbor_kernel,
    motion_difficulty, freeze_after -- plus the toroidal position sampler and the
    random / seeded RNG contract. Only the RENDERING differs: textured layers
    seen through digit-shaped apertures, instead of digit intensities on black.

    Layers, not digits
    ------------------
    The parent counts "digits"; this class counts independently moving LAYERS,
    of which the background is one. So `self.num_digits == num_figures + 1`, and
    the background is the LAST motion slot. Figure slots therefore line up with
    label indices, which is what a classification head wants.

    Yields, in this order:
        seq       (T, 1, H, W) float32
        label     int if num_figures == 1, else (N,) long
        motion    (T, N+1, 2) long   -- if return_motion    [figures, then bg]
        positions (T, N, 2) long     -- if return_positions [figures only]
        mask      (T, N, H, W) f32   -- if return_mask      [per figure]

    motion[t, i] is (vx, vy) and is the step taking frame t to frame t+1, the
    same meaning it has in TDMovingMNISTDataset. mask is per-frame because in
    'moving_mask' the aperture travels; in 'static_mask' every frame repeats the
    same mask, and the time axis is kept so both variants share one interface.

    Classification after T frames
    -----------------------------
    `labels[i]` names the digit whose aperture is figure slot i, so
    `motion[:, i]`, `positions[:, i]`, `mask[:, i]` and `labels[i]` all refer to
    the same object. With require_distinct_digits=True (the default) the labels
    in a sequence are distinct, which is what makes a set-prediction loss
    well posed.

    Parameters beyond TDMovingMNISTDataset's
    ----------------------------------------
    num_figures   : how many motion-defined digits. Each gets its own texture,
                    so they remain distinguishable only by motion and shape.
    variant       : 'moving_mask' | 'static_mask' -- the experimental variable.
    corr_len      : correlation length of every texture, in pixels. All layers
                    share it, so no INTENSITY statistic can leak a figure -- but
                    the SEAM between two independently drawn smooth fields still
                    can. Measured on frame 0 alone, a local gradient magnitude
                    finds the mask boundary with AUC 0.49 at corr_len <= 0.5,
                    0.53 at 1.0 and 0.62 at 3.0. Above ~1.0 the correlation
                    length exceeds the stroke width, the outline becomes visible
                    in a single frame, and the task stops being purely
                    motion-defined. Keep corr_len <= 1.0 for the "no single frame
                    contains the figure" claim.
    digit_scale   : integer upscaling of the 28x28 glyph by pixel replication.
    mask_threshold: glyph intensity above which a pixel belongs to the figure.
    min_dv        : required max-norm gap between every figure and the
                    background, at EVERY step. Below 1 a figure can travel with
                    the background and vanish. 0 disables the check.
    separate_figures : also require that gap pairwise BETWEEN figures. Off by
                    default: two figures sharing a velocity are one motion group
                    but still two shapes, which is fine for classification and
                    only degenerate for motion grouping. Costs acceptance --
                    see max_velocity_tries.
    max_velocity_tries : whole trajectories are redrawn until the separation
                    holds, which keeps the parent's motion process EXACT rather
                    than biasing it by resampling individual steps. Acceptance
                    at min_dv=2, T=20, max_speed=3 runs from 88% (constant, one
                    figure) down to 21% (stochastic, two figures), so the
                    default 200 leaves an enormous margin.
    normalize     : how the zero-mean unit-variance textures are mapped into the
                    [0, 1] range the models and BCE/MSE losses expect. 'affine'
                    (default) clips at +-`clip` sigma and rescales, a FIXED map
                    -- contrast means the same thing in every sequence. 'minmax'
                    rescales per sequence, making contrast sequence-dependent
                    and letting the extremes leak information; 'none' hands back
                    raw sigma units.
    motion_mode   : defaults to 'constant' here rather than the parent's
                    'piecewise', because a constant velocity is what the
                    co-moving-transport analysis assumes. Pass any of the
                    parent's modes to override.
    """

    def __init__(
        self,
        root,
        train=True,
        seq_len=20,
        image_size=64,

        # --- common-fate specific ---
        num_figures=1,
        variant="moving_mask",
        corr_len=1.0,
        digit_scale=1,
        mask_threshold=0.3,
        min_dv=2,
        separate_figures=False,
        max_velocity_tries=200,
        normalize="affine",
        clip=3.0,

        # --- inherited motion vocabulary (see TDMovingMNISTDataset) ---
        max_speed=3,
        motion_mode=None,
        transition_mode="smooth",
        min_segment=3,
        max_segment=6,
        p_change=0.25,
        smooth_probability=0.8,
        neighbor_kernel="legacy",
        motion_difficulty=None,
        freeze_after=None,

        min_center_distance=20,
        reject_overlap=True,
        require_distinct_velocities=True,
        require_distinct_digits=True,

        return_motion=True,
        return_positions=False,
        return_mask=False,

        transform=None,
        download=True,
        random=True,
        seed=42,
        max_tries=200,
    ):
        if variant not in ("moving_mask", "static_mask"):
            raise ValueError(
                f"variant must be 'moving_mask' or 'static_mask', got {variant!r}")
        if normalize not in ("affine", "minmax", "none"):
            raise ValueError(
                f"normalize must be 'affine', 'minmax' or 'none', got {normalize!r}")
        if num_figures < 1:
            raise ValueError(f"num_figures must be >= 1, got {num_figures}")
        if min_dv > 2 * max_speed:
            raise ValueError(
                f"min_dv={min_dv} is unsatisfiable with max_speed={max_speed} "
                f"(the largest possible separation is {2 * max_speed}).")

        # 'constant' unless the caller says otherwise -- but when
        # motion_difficulty is in play, hand the parent its own default so its
        # "difficulty overrides the motion family" warning does not fire on a
        # default this class chose rather than the user.
        if motion_mode is None:
            motion_mode = "piecewise" if motion_difficulty is not None else "constant"

        # The parent's "digit" is this class's moving LAYER, and the background
        # is one of them -- the last slot. Everything the parent generates per
        # digit (velocities above all) is thus generated for the background too.
        super().__init__(
            root=root,
            train=train,
            seq_len=seq_len,
            image_size=image_size,
            num_digits=num_figures + 1,
            max_speed=max_speed,
            motion_mode=motion_mode,
            transition_mode=transition_mode,
            min_segment=min_segment,
            max_segment=max_segment,
            p_change=p_change,
            smooth_probability=smooth_probability,
            neighbor_kernel=neighbor_kernel,
            motion_difficulty=motion_difficulty,
            freeze_after=freeze_after,
            min_center_distance=min_center_distance,
            reject_overlap=reject_overlap,
            require_distinct_velocities=require_distinct_velocities,
            require_distinct_digits=require_distinct_digits,
            return_motion=return_motion,
            return_positions=return_positions,
            transform=transform,
            download=download,
            random=random,
            seed=seed,
            max_tries=max_tries,
        )

        self.num_figures = num_figures
        self.variant = variant
        self.corr_len = corr_len
        self.digit_scale = digit_scale
        self.mask_threshold = mask_threshold
        self.min_dv = min_dv
        self.separate_figures = separate_figures
        self.max_velocity_tries = max_velocity_tries
        self.normalize = normalize
        self.clip = float(clip)
        self.return_mask = return_mask

        self._rng = _ParentRNG(self)
        # One warning per dataset, not one per sample: a failing rejection is a
        # property of the configuration, so the first hit says everything.
        self._warned_unseparated = False

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __getitem__(self, index):
        S, N = self.image_size, self.num_figures

        glyphs, labels = self._sample_glyphs()
        positions = self._sample_initial_positions(N)      # figures only
        motions, separated = self._sample_separated_motion()

        if not separated and not self._warned_unseparated:
            self._warned_unseparated = True
            warnings.warn(
                f"could not separate the layer velocities by min_dv={self.min_dv} "
                f"within {self.max_velocity_tries} tries (motion_mode="
                f"{self.motion_mode!r}, max_speed={self.max_speed}, "
                f"num_figures={N}, separate_figures={self.separate_figures}); "
                f"such sequences contain little or no motion-defined figure. "
                f"Lower min_dv, raise max_speed, or raise max_velocity_tries.",
                UserWarning, stacklevel=2)

        disp = cumulative_displacement(motions.numpy())    # (T, N+1, 2) as (vx, vy)

        masks = np.stack([self._figure_mask(g, cx, cy)
                          for g, (cx, cy) in zip(glyphs, positions)])
        textures = np.stack([band_limited_noise(S, S, self.corr_len, self._rng)
                             for _ in range(N)])
        bg_texture = band_limited_noise(S, S, self.corr_len, self._rng)

        frames, track = render_sequence(
            masks, textures, bg_texture,
            d_fig=_xy_to_yx(disp[:, :N]),
            d_bg=_xy_to_yx(disp[:, N]),
            variant=self.variant)

        seq = torch.from_numpy(self._normalize(frames)).unsqueeze(1)   # (T,1,H,W)
        if self.transform:
            seq = self.transform(seq)

        out = [seq, (labels[0] if N == 1 else torch.tensor(labels, dtype=torch.long))]
        if self.return_motion:
            out.append(motions)                                        # (T, N+1, 2)
        if self.return_positions:
            pos = torch.from_numpy(disp[:, :N]).long() + torch.tensor(
                positions, dtype=torch.long).unsqueeze(0)
            out.append(pos % S)                                        # (T, N, 2)
        if self.return_mask:
            out.append(torch.from_numpy(track))                        # (T, N, H, W)
        return tuple(out)

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_glyphs(self):
        """N raw uint8 glyphs and their labels, honouring require_distinct_digits."""
        glyphs, labels, used = [], [], set()
        for _ in range(self.num_figures):
            while True:
                idx = self._randint(0, len(self.mnist))
                img, lbl = self.mnist[idx]
                if (not self.require_distinct_digits) or (lbl not in used):
                    break
            glyphs.append(np.asarray(img, dtype=np.uint8))
            labels.append(lbl)
            used.add(lbl)
        return glyphs, labels

    def _figure_mask(self, glyph, cx, cy):
        """
        Binary aperture for one glyph, centred then rolled to (cx, cy).

        Centre-then-roll is the parent's placement convention (_pad_digit plus
        torch.roll by (cy, cx)), so a position means the same thing in both
        datasets and `positions` is directly comparable.
        """
        S = self.image_size
        g = glyph_to_mask(glyph, scale=self.digit_scale, thresh=self.mask_threshold)
        h, w = g.shape
        if h > S or w > S:
            raise ValueError(
                f"digit is {h}x{w} after digit_scale={self.digit_scale} but the "
                f"canvas is {S}x{S}; raise image_size or lower digit_scale.")
        m = np.zeros((S, S), np.float32)
        top, left = (S - h) // 2, (S - w) // 2
        m[top:top + h, left:left + w] = g
        return np.roll(m, (cy, cx), (0, 1))

    def _velocities_separated(self, motions):
        """
        Every figure at least min_dv from the background at every step, in the
        max norm -- and pairwise between figures too when separate_figures.
        """
        if self.min_dv <= 0:
            return True
        N = self.num_figures
        fig, bg = motions[:, :N], motions[:, N:]
        if int((fig - bg).abs().amax(dim=2).min()) < self.min_dv:
            return False
        if self.separate_figures and N > 1:
            for i in range(N):
                for j in range(i + 1, N):
                    gap = (fig[:, i] - fig[:, j]).abs().amax(dim=1).min()
                    if int(gap) < self.min_dv:
                        return False
        return True

    def _sample_separated_motion(self):
        """
        Redraw WHOLE trajectories until the separation holds.

        Rejecting the whole trajectory keeps the parent's motion process exact.
        Resampling only the offending steps would be cheaper and would quietly
        change the velocity statistics this class inherits -- which is the one
        thing a subclass of TDMovingMNISTDataset must not do.
        """
        motions = None
        for _ in range(self.max_velocity_tries):
            motions = self._generate_motion_trajectory()   # (T, N+1, 2)
            if self._velocities_separated(motions):
                return motions, True
        return motions, False

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
