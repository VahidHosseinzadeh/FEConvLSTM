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
from contextlib import contextmanager

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
        return self._ds._src_rng()

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
    digit_indices : which MNIST indices this dataset may draw figure glyphs from.
                    Pass disjoint lists to two instances to get a genuine
                    train/val split AT THE GLYPH LEVEL -- necessary for a
                    classification task, where sharing glyphs between train and
                    val makes the val number optimistic.

                    Without it the dataset IGNORES its index and samples a glyph
                    at random from the whole split on every access, so splitting
                    by index (random_split and friends) separates nothing. That
                    is harmless for next-frame prediction, which is what the
                    parent class was built for, and wrong for classification.
    variant       : 'moving_mask' | 'static_mask' -- the experimental variable.
    corr_len      : correlation length of every texture, in pixels. DEFAULT 0
                    (white noise), and anything above ~0.5 breaks the premise of
                    the dataset.

                    All layers share the statistic, so no INTENSITY cue can leak
                    a figure -- but the SEAM can. With corr_len > 0 pixels WITHIN
                    a region are correlated while pixels ACROSS the boundary are
                    independent, so the figure's outline is marked by a local
                    statistics discontinuity in EVERY frame. A single-frame CNN,
                    given no temporal information whatsoever, reaches:

                        corr_len  0.0   0.5   1.0   2.0
                        accuracy  11%   13%   24%   36%      (chance 10%)

                    At corr_len 0 the noise is independent everywhere including
                    at the boundary, so there is no local statistic to find.

                    An earlier version of this docstring said "keep corr_len <=
                    1.0", inferred from a hand-crafted gradient detector that
                    scored only AUC 0.53 there. That underestimated the leak
                    badly: a trained CNN extracts far more from the same cue. The
                    numbers above come from the CNN, which is the adversary that
                    matters.

                    corr_len 0 is also simply better for the experiment: measured
                    transport IoU 0.661 vs 0.522, and slot_hit_fig 100% vs 96%.
    digit_scale   : integer upscaling of the 28x28 glyph by pixel replication.
    mask_threshold: glyph intensity above which a pixel belongs to the figure.
    min_dv        : required max-norm gap between every figure and the
                    background, at EVERY step. Below 1 a figure can travel with
                    the background and vanish. 0 disables the check.
    bg_opposite_at_start : require the background to travel in an OPPOSING
                    direction to every figure at t=0 (strictly negative dot
                    product), while staying on the SHARED velocity grid.

                    Prefer this to bg_speed_range whenever felstm is in the
                    comparison. bg_speed_range guarantees separation by putting
                    the background outside the figure grid -- but felstm's
                    copies sit at fixed lattice velocities bounded by v_range,
                    so an off-grid background is a motion felstm structurally
                    CANNOT represent while melstm's tracked slots can. That is a
                    difference in expressive power confounded with the effect
                    being measured. Separating by direction keeps both motions
                    representable by both models.
    bg_mirror     : the background moves at EXACTLY the negative of the figure's
                    velocity, v_bg(t) = -v_fig(t), at every step. Only the figure's
                    trajectory is drawn (under the usual motion law); the
                    background switches when, and only when, the figure does. On
                    the shared grid, so felstm represents both, and |v_fig - v_bg|
                    = 2|v_fig| >= 2, so min_dv <= 2 holds by construction. Note
                    what it gives away: the background's velocity -- the dominant
                    phase-correlation peak -- determines the figure's exactly.
                    One figure only.
    bg_speed_range : (lo, hi) -- give the BACKGROUND its own velocity grid,
                    every integer (vx, vy) with lo <= max(|vx|,|vy|) <= hi.
                    Requires lo > max_speed, which makes the grid disjoint from
                    the figures': the background then can NEVER coincide with a
                    figure, by construction rather than by rejection. When
                    lo - max_speed >= min_dv the separation constraint is a
                    theorem and the rejection loop skips it. The background
                    keeps the same motion_mode and transition statistics as the
                    figures -- only its alphabet of velocities differs.
                    None (default) draws the background from the figure grid and
                    enforces separation by rejection instead.
    bg_velocity   : (vx, vy) -- the background moves at exactly this velocity at
                    every step of every sequence; only the figures' motion is
                    drawn. Any integer velocity, on or off the figure grid. The
                    motion law (motion_mode, p_change, ...) then applies to the
                    figures alone, so a motion sweep varies the figure and nothing
                    else -- by default the background is drawn under the SAME law
                    as the figures and every sweep axis moves both layers.
                    Separation is structural when max(|vx|,|vy|) - max_speed >=
                    min_dv, and by whole-trajectory rejection of the figure
                    otherwise. Exclusive with bg_speed_range and
                    bg_opposite_at_start. Off felstm's lattice (|v| > v_range) it is
                    a motion felstm cannot represent -- the caveat above applies.
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
    stream_seed   : when set, every sample draws ALL its randomness (placement,
                    both layers' motion, every texture, extra glyphs) from its own
                    RNG keyed on (stream_seed, epoch, index). Index the dataset
                    with (epoch, index) to get a fresh sample per epoch; a plain
                    index means epoch 0. A sample is then a function of those three
                    numbers alone -- not of the worker that renders it, the order
                    it is requested in, or the global RNG. That is what makes a
                    random=True training stream identical across model seeds:
                    without it, draws come from the global np.random, which
                    DataLoader seeds per worker from the torch RNG, i.e. from the
                    model seed. None (default) keeps the parent's contract.
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
        corr_len=0.0,
        digit_scale=1,
        mask_threshold=0.3,
        min_dv=2,
        separate_figures=False,
        bg_opposite_at_start=False,
        bg_mirror=False,
        bg_speed_range=None,
        bg_velocity=None,
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

        digit_indices=None,

        return_motion=True,
        return_positions=False,
        return_mask=False,

        transform=None,
        download=True,
        random=True,
        seed=42,
        stream_seed=None,
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
        if bg_velocity is not None:
            bg_velocity = tuple(int(c) for c in bg_velocity)
            if len(bg_velocity) != 2:
                raise ValueError(f"bg_velocity must be (vx, vy), got {bg_velocity!r}")
            if bg_speed_range is not None or bg_opposite_at_start or bg_mirror:
                raise ValueError(
                    "bg_velocity fixes the background's motion outright; it cannot be "
                    "combined with bg_speed_range, bg_opposite_at_start or bg_mirror.")
        if bg_mirror:
            if bg_speed_range is not None or bg_opposite_at_start:
                raise ValueError(
                    "bg_mirror sets the background to -v_fig at every step; it cannot "
                    "be combined with bg_speed_range or bg_opposite_at_start.")
            if num_figures != 1:
                raise ValueError(f"bg_mirror mirrors ONE figure, got num_figures={num_figures}")
        if stream_seed is not None and int(stream_seed) < 0:
            raise ValueError(f"stream_seed must be >= 0, got {stream_seed}")
        max_gap = (max_speed + max(abs(c) for c in bg_velocity)
                   if bg_velocity is not None else 2 * max_speed)
        if min_dv > max_gap and bg_speed_range is None:
            raise ValueError(
                f"min_dv={min_dv} is unsatisfiable with max_speed={max_speed} "
                f"(the largest possible separation is {max_gap}).")

        if bg_speed_range is not None:
            lo, hi = bg_speed_range
            if lo > hi:
                raise ValueError(f"bg_speed_range={bg_speed_range} is empty (lo > hi).")
            if lo <= max_speed:
                raise ValueError(
                    f"bg_speed_range={bg_speed_range} overlaps the figure grid "
                    f"(max_speed={max_speed}); the whole point of this argument is "
                    f"that the background can NEVER coincide with a figure, so its "
                    f"minimum speed must exceed max_speed.")

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
        self.digit_indices = (None if digit_indices is None
                              else list(int(i) for i in digit_indices))
        self.variant = variant
        self.corr_len = corr_len
        self.digit_scale = digit_scale
        self.mask_threshold = mask_threshold
        self.min_dv = min_dv
        self.separate_figures = separate_figures
        self.bg_opposite_at_start = bg_opposite_at_start
        self.bg_mirror = bg_mirror
        self.bg_speed_range = bg_speed_range

        # Background velocities live on their own grid: every integer (vx, vy)
        # whose max-norm falls in [lo, hi]. Because lo > max_speed, this set is
        # DISJOINT from the figure grid, so the background can never coincide
        # with a figure -- no rejection sampling, no residual chance of a
        # figureless sequence.
        self.bg_velocity_grid = None
        self.bg_separation_is_structural = False
        if bg_speed_range is not None:
            lo, hi = bg_speed_range
            self.bg_velocity_grid = [
                (vx, vy)
                for vx in range(-hi, hi + 1)
                for vy in range(-hi, hi + 1)
                if lo <= max(abs(vx), abs(vy)) <= hi
            ]
            # A figure has max-norm <= max_speed and the background >= lo, so
            # their max-norm gap is at least lo - max_speed. When that already
            # covers min_dv the constraint is a theorem, not a sample-time
            # check, and the rejection loop can skip it entirely.
            self.bg_separation_is_structural = (lo - max_speed) >= min_dv
        self.bg_velocity = bg_velocity
        if bg_velocity is not None:
            self.bg_separation_is_structural = (
                max(abs(c) for c in bg_velocity) - max_speed) >= min_dv
        if bg_mirror:
            # |v_fig - (-v_fig)| = 2|v_fig| >= 2, since (0, 0) is off the grid.
            self.bg_separation_is_structural = min_dv <= 2
        self.stream_seed = None if stream_seed is None else int(stream_seed)
        self._sample_rng = None
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

    def __len__(self):
        # With a digit pool, one epoch is one pass over THOSE glyphs.
        return len(self.mnist) if self.digit_indices is None else len(self.digit_indices)

    def __getitem__(self, index):
        epoch = 0
        if isinstance(index, tuple):          # (epoch, index), see stream_seed
            epoch, index = index
        if self.stream_seed is None:
            return self._render(index)
        self._sample_rng = np.random.RandomState(np.random.MT19937(
            np.random.SeedSequence([self.stream_seed, int(epoch), int(index)])))
        try:
            return self._render(index)
        finally:
            self._sample_rng = None

    def _render(self, index):
        S, N = self.image_size, self.num_figures

        glyphs, labels = self._sample_glyphs(index)
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
    # RNG: the parent's three draw helpers, routed through the per-sample
    # stream when one is active. Otherwise identical to the parent's, draw for
    # draw, so fixed benchmarks generated before stream_seed existed reproduce.
    # ------------------------------------------------------------------

    def _src_rng(self):
        # getattr: the parent's __init__ runs before this class sets the attribute.
        rng = getattr(self, "_sample_rng", None)
        if rng is not None:
            return rng
        return np.random if self.random else self.rng

    def _randint(self, low, high):
        return int(self._src_rng().randint(low, high))

    def _choice(self, values):
        return values[self._src_rng().randint(len(values))]

    def _random(self):
        return self._src_rng().rand()

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_glyphs(self, index):
        """
        N raw uint8 glyphs and their labels, honouring require_distinct_digits.

        With `digit_indices`, figure 0's glyph is `digit_indices[index]` -- the
        dataset index actually SELECTS the digit, so splitting the index range
        splits the glyphs. Without it the index is ignored and every glyph is
        drawn at random from the whole split, which is fine for a prediction task
        and WRONG for classification: a train/val split by index would then draw
        both halves from the same pool and the val metric would be measured on
        digits the model had already trained on.

        Any additional figures are drawn from the same pool, so they cannot leak
        across a split either.
        """
        pool = self.digit_indices
        glyphs, labels, used = [], [], set()
        for i in range(self.num_figures):
            if i == 0 and pool is not None:
                mnist_idx = pool[index % len(pool)]
                img, lbl = self.mnist[mnist_idx]
                if self.require_distinct_digits:
                    used.add(lbl)
                glyphs.append(np.asarray(img, dtype=np.uint8))
                labels.append(lbl)
                continue
            while True:
                if pool is None:
                    mnist_idx = self._randint(0, len(self.mnist))
                else:
                    mnist_idx = pool[self._randint(0, len(pool))]
                img, lbl = self.mnist[mnist_idx]
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
        # With a disjoint background grid whose gap already covers min_dv, this
        # check can only ever pass -- skip it rather than pay for it per draw.
        if not self.bg_separation_is_structural:
            if int((fig - bg).abs().amax(dim=2).min()) < self.min_dv:
                return False
        if self.separate_figures and N > 1:
            for i in range(N):
                for j in range(i + 1, N):
                    gap = (fig[:, i] - fig[:, j]).abs().amax(dim=1).min()
                    if int(gap) < self.min_dv:
                        return False

        if self.bg_opposite_at_start:
            # At t=0 the background must travel in an OPPOSING direction to every
            # figure: a strictly negative dot product, so the two are not merely
            # different but visibly counter-moving from the first frame pair.
            #
            # This is the alternative to bg_speed_range for a fair felstm
            # comparison. Putting the background outside the figure grid
            # guarantees they never coincide, but it also puts the background
            # beyond every lattice copy felstm has, so felstm cannot represent
            # that motion at all while melstm's tracked slots can -- a difference
            # in what the models can express, confounded with the thing being
            # measured. Keeping the background ON the shared grid and separating
            # it by DIRECTION instead keeps both models able to represent both
            # motions.
            f0, b0 = motions[0, :N].to(torch.long), motions[0, N].to(torch.long)
            if bool(((f0 * b0.unsqueeze(0)).sum(dim=1) >= 0).any()):
                return False

        return True

    @contextmanager
    def _velocity_grid(self, grid, n_slots):
        """
        Run the parent's trajectory machinery on a different grid and slot count.

        The parent draws every slot from self.velocity_grid, which is exactly
        what must NOT happen when the background has its own disjoint grid.
        Swapping the three attributes it reads -- rather than reimplementing the
        motion modes -- is what keeps the background's motion statistically
        identical to the figures': same motion_mode, same transition_mode, same
        segment lengths, different alphabet.
        """
        saved = (self.velocity_grid, self._velocity_set, self.num_digits)
        self.velocity_grid, self._velocity_set, self.num_digits = (
            grid, set(grid), n_slots)
        try:
            yield
        finally:
            self.velocity_grid, self._velocity_set, self.num_digits = saved

    def _generate_split_grid_motion(self):
        """
        Figures on the figure grid, background on its own grid, fixed at
        bg_velocity, or mirroring the figure -- concatenated into the usual
        (T, N+1, 2) layout with the background last.
        """
        N = self.num_figures
        with self._velocity_grid(self.velocity_grid, N):
            fig = self._generate_motion_trajectory()          # (T, N, 2)
        if self.bg_mirror:
            bg = -fig[:, :1]                                  # v_bg = -v_fig, no draw
        elif self.bg_velocity is not None:
            bg = torch.tensor(self.bg_velocity, dtype=torch.long).expand(
                fig.shape[0], 1, 2).clone()                   # no draw at all
        else:
            with self._velocity_grid(self.bg_velocity_grid, 1):
                bg = self._generate_motion_trajectory()       # (T, 1, 2)
        return torch.cat([fig, bg], dim=1)                    # (T, N+1, 2)

    def _sample_separated_motion(self):
        """
        Redraw WHOLE trajectories until the separation holds.

        Rejecting the whole trajectory keeps the parent's motion process exact.
        Resampling only the offending steps would be cheaper and would quietly
        change the velocity statistics this class inherits -- which is the one
        thing a subclass of TDMovingMNISTDataset must not do.
        """
        split = (self.bg_velocity_grid is not None or self.bg_velocity is not None
                 or self.bg_mirror)
        draw = (self._generate_split_grid_motion if split
                else self._generate_motion_trajectory)
        motions = None
        for _ in range(self.max_velocity_tries):
            motions = draw()                              # (T, N+1, 2)
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
