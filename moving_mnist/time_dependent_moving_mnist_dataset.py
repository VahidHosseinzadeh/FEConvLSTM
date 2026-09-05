import warnings

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class TDMovingMNISTDataset(Dataset):
    """
    Moving MNIST with time-dependent, multi-digit motion.

    Motion modes
    ------------
    "constant"    : velocity is fixed for the whole sequence.
    "piecewise"   : velocity held for a random segment, then updated.
    "stochastic"  : velocity changes with probability p_change each step.
    "accelerate"  : each digit gets an acceleration sign; every
                    min_segment..max_segment steps the velocity is stepped away
                    from or toward zero, so |v| ramps rather than jumping. The
                    sign reverses at the limits, so the speed BREATHES between 1
                    and max_speed instead of saturating and sticking. (0, 0) is
                    off the velocity grid, so a digit never actually stops. Unlike the other modes the velocity sequence is
                    systematic, not a random walk -- use it to test motion whose
                    *magnitude* changes over the context. transition_mode,
                    smooth_probability and p_change are unused here.
    "harmonic"    : velocity = constant drift + sinusoid, per axis, with the
                    amplitude/period/phase drawn per digit and the trajectory
                    then DETERMINISTIC. Covers constant flow, circular orbits,
                    "constant along one axis, oscillating along the other", and
                    Lissajous figures in one family (harmonic_shapes). This is
                    the only mode whose future velocity is predictable from the
                    velocity history alone -- see _generate_harmonic_trajectory
                    for why that distinction matters and why the other four
                    modes do not have it. transition_mode, smooth_probability,
                    p_change and min/max_segment are all unused here.

    Which parameters apply to which mode
    ------------------------------------
        min_segment/max_segment : piecewise, accelerate
        p_change                : stochastic only
        transition_mode         : piecewise, stochastic
        smooth_probability      : only when transition_mode == "smooth"
                                  ("smooth" with probability 0.0 is exactly
                                   equivalent to "uniform")
        harmonic_shapes         : harmonic only
        harmonic_period         : harmonic only
        harmonic_amp            : harmonic only
        harmonic_drift          : harmonic only

    All modes support freeze_after (see below).

    neighbor_kernel
    ---------------
    How "smooth" transitions pick the next velocity. "legacy" (default) draws
    uniformly from the valid neighbours of v, which is degree-biased: interior
    low-speed states have more neighbours and accumulate stationary mass.
    "symmetric" proposes a uniform unit offset and holds if it leaves the grid,
    which is doubly stochastic and leaves the velocity uniform over the grid at
    every switching rate. Default stays "legacy" only because "symmetric" draws
    a different number of RNG values and so cannot reproduce existing fixed
    benchmarks byte-for-byte.

    motion_difficulty
    -----------------
    A single scalar d in [0, 1] controlling HOW OFTEN the velocity changes, for
    sweeps that need one ordered "how hard is the motion" axis. Setting it FORCES
    motion_mode="stochastic" and transition_mode="smooth" (passing a non-default
    value for either alongside it warns).

        d = 0.0  : p_change = 0     -- pure constant flow, 0 bits/frame
        d = 0.5  : p_change = 0.222 -- exactly the TRAINING regime, ~1.2 bits/frame
        d = 1.0  : p_change = 1     -- a change every frame, 80% of them to a
                   lattice neighbour: a random walk in velocity, ~3.4 bits/frame

    Two things are deliberately held OFF the axis:

    smooth_probability is fixed at the training value 0.80 rather than swept to
    0. The top of the scale is therefore NOT the maximum-entropy process (3.4
    bits, not the 4.5 of an i.i.d. uniform velocity). That is the point: an
    i.i.d. velocity makes the digit vibrate in place instead of travelling, which
    collapses net displacement and is an EASIER task, not a harder one. Jump size
    is a separate axis and belongs in a named regime.

    max_speed is untouched: leaving the velocity grid would confound "harder to
    infer" with "impossible to represent".

    Pair d with neighbor_kernel="symmetric" for a sweep: the legacy kernel is
    degree-biased, so mean |v| drifts with d and difficulty gets confounded with
    speed.

    freeze_after
    ------------
    If not None, the velocity at step (freeze_after - 1) is frozen for all
    subsequent steps.  Combined with piecewise/stochastic this gives a
    two-phase sequence:

        steps 0 .. freeze_after-1  : time-varying motion
        steps freeze_after .. T-1  : constant (last velocity is frozen)

    This is the correct data format for the length-generalisation
    experiment: train on full seq_len frames, then evaluate whether the
    model continues correctly with the velocity it learned at the end of
    the encoder.

    Example
    -------
    seq_len=20, input_frames=10, freeze_after=10
        frames 0-9:   piecewise / stochastic motion  (encoder sees these)
        frames 10-19: constant at velocity of frame 9 (decoder predicts these,
                      and length-gen continues beyond 19 at the same velocity)

    Initial position sampling
    --------------------------
    Positions are drawn on a toroidal (periodic) canvas.  The toroidal
    Euclidean distance is used everywhere:

        toroidal_dist(a, b) = sqrt(
            min(|ax-bx|, S-|ax-bx|)^2 + min(|ay-by|, S-|ay-by|)^2
        )

    This correctly handles digits that are close because they wrap around
    the boundary.

    If rejection sampling fails (max_tries exceeded), a guaranteed-
    separation fallback places digits at maximally separated points on the
    torus diagonal:  digit i is offset from digit 0 by (i*S//N, i*S//N).
    For N=2 on a torus of size S this gives toroidal distance S/sqrt(2),
    which is always > digit_size (28) for S >= 40.
    """

    def __init__(
        self,
        root,
        train=True,
        seq_len=20,
        image_size=64,
        num_digits=2,

        max_speed=3,

        motion_mode="piecewise",
        transition_mode="smooth",

        min_segment=3,
        max_segment=6,

        p_change=0.25,
        smooth_probability=0.8,
        neighbor_kernel="legacy",       # "legacy" | "symmetric", see
                                        # _sample_neighbor_velocity

        # motion_mode="harmonic" only -- see _generate_harmonic_trajectory
        harmonic_shapes=("constant", "orbit", "axis", "lissajous"),
        harmonic_period=(12, 30),       # integer period, in frames, inclusive
        harmonic_amp=(0.6, 1.0),        # oscillation amplitude, as a FRACTION
                                        # of max_speed
        harmonic_drift=True,            # add a constant velocity offset on top
                                        # of the oscillation

        motion_difficulty=None,

        # Two-phase motion: run dynamic motion for freeze_after steps,
        # then freeze the velocity.  None = no freezing.
        freeze_after=None,

        min_center_distance=20,
        reject_overlap=True,
        require_distinct_velocities=True,
        require_distinct_digits=True,

        return_motion=True,
        return_positions=False,

        transform=None,
        download=True,

        random=True,
        seed=42,

        max_tries=200,
    ):
        super().__init__()

        self.mnist     = MNIST(root=root, train=train, download=download)
        self.to_tensor = ToTensor()

        self.seq_len    = seq_len
        self.image_size = image_size
        self.num_digits = num_digits
        self.max_speed  = max_speed

        self.motion_mode        = motion_mode
        self.transition_mode    = transition_mode
        self.min_segment        = min_segment
        self.max_segment        = max_segment
        self.p_change           = p_change
        self.smooth_probability = smooth_probability
        self.freeze_after       = freeze_after

        self.min_center_distance         = min_center_distance
        self.reject_overlap              = reject_overlap
        self.require_distinct_velocities = require_distinct_velocities
        self.require_distinct_digits     = require_distinct_digits

        self.return_motion    = return_motion
        self.return_positions = return_positions

        self.transform = transform
        self.random    = random
        self.max_tries = max_tries

        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Velocity grid: all integer (vx, vy) within max_speed, zero excluded
        self.velocity_grid = [
            (vx, vy)
            for vx in range(-max_speed, max_speed + 1)
            for vy in range(-max_speed, max_speed + 1)
            if (vx, vy) != (0, 0)
        ]

        self.harmonic_shapes  = tuple(harmonic_shapes)
        self.harmonic_period  = tuple(harmonic_period)
        self.harmonic_amp     = tuple(harmonic_amp)
        self.harmonic_drift   = harmonic_drift
        if not self.harmonic_shapes:
            raise ValueError("harmonic_shapes must not be empty")
        unknown = set(self.harmonic_shapes) - {"constant", "orbit", "axis", "lissajous"}
        if unknown:
            raise ValueError(f"unknown harmonic shape(s): {sorted(unknown)}")

        self.neighbor_kernel = neighbor_kernel
        self._velocity_set   = set(self.velocity_grid)
        self._unit_offsets   = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)
                                if (a, b) != (0, 0)]

        # Optional difficulty preset
        if motion_difficulty is not None:
            overridden = []
            if motion_mode != "piecewise":
                overridden.append(f"motion_mode={motion_mode!r}")
            if transition_mode != "smooth":
                overridden.append(f"transition_mode={transition_mode!r}")
            if overridden:
                warnings.warn(
                    "motion_difficulty is defined on the stochastic family and "
                    "overrides the motion family, so "
                    + " and ".join(overridden)
                    + " will be ignored in favour of motion_mode='stochastic', "
                      "transition_mode='smooth'.",
                    UserWarning, stacklevel=2)
            self._apply_difficulty(motion_difficulty)

    def _apply_difficulty(self, d):
        """
        One scalar d in [0,1] controlling HOW OFTEN the velocity changes, with the
        jump-size statistics held at the training value throughout.

            d = 0.0  ->  p_change = 0      : velocity never changes. A pure flow --
                                             the setting flow-equivariant models are
                                             built for.
            d = 0.5  ->  p_change = 0.222  : the TRAINING regime, one change every
                                             ~4.5 frames.
            d = 1.0  ->  p_change = 1      : the velocity changes every frame, 80% of
                                             changes to a lattice neighbour -- a
                                             random walk in velocity.

        p(d) = d ** 2.170 is solved so that p(0.5) = 1/4.5, matching the training
        segment range [3, 6].

        smooth_probability is FIXED at 0.80 (the training value), not swept. Driving
        it to 0 makes the velocity i.i.d. at d=1, which collapses net displacement
        and makes the task easier rather than harder. Jump size is a separate axis
        and belongs in the named-regime panel, not in d.

        max_speed is untouched: difficulty must never become the trivial
        'outside the velocity grid' result.

        Pure parameter assignment: this must never consume the RNG, or every
        previously generated fixed benchmark would silently change.
        """
        d = float(np.clip(d, 0.0, 1.0))

        self.motion_mode        = "stochastic"
        self.transition_mode    = "smooth"
        self.p_change           = d ** 2.170
        self.smooth_probability = 0.80

        mean_gap = (1.0 / self.p_change) if self.p_change > 1e-9 else float(self.seq_len)
        mean_gap = min(mean_gap, float(self.seq_len))
        self.min_segment = max(1, int(round(0.70 * mean_gap)))
        self.max_segment = max(self.min_segment + 1, int(round(1.35 * mean_gap)))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def reset_rng(self):
        """
        Rewind the private RandomState to its initial seed. With random=False
        the RNG is stateful (every __getitem__ advances it), so a fixed
        benchmark set requires resetting before each full pass — otherwise
        successive evaluations silently see different sequences.
        """
        self.rng = np.random.RandomState(self.seed)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        S = self.image_size
        N = self.num_digits
        T = self.seq_len

        # 1. Sample and pad digits
        imgs, labels = [], []
        used_labels = set()

        for _ in range(N):

            while True:
                idx = self._randint(0, len(self.mnist))
                img_pil, lbl = self.mnist[idx]

                if (not self.require_distinct_digits) or (lbl not in used_labels):
                    break

            imgs.append(self._pad_digit(self.to_tensor(img_pil), S))
            labels.append(lbl)
            used_labels.add(lbl)

        # 2. Sample initial positions and motion trajectory
        init_pos = self._sample_initial_positions()    # list of N (cx, cy)
        motions  = self._generate_motion_trajectory()  # (T, N, 2) [vx, vy]

        # 3. Cumulative positions
        # positions[t, i] = init_pos[i] + sum(motions[0:t, i])
        cum = torch.cumsum(motions, dim=0)   # (T, N, 2)
        cum = torch.cat([torch.zeros(1, N, 2, dtype=torch.long), cum[:-1]], dim=0)

        init_tensor = torch.tensor(init_pos, dtype=torch.long)  # (N, 2)
        positions   = cum + init_tensor.unsqueeze(0)             # (T, N, 2)

        # 4. Render frames
        seq = torch.zeros(T, 1, S, S, dtype=imgs[0].dtype)
        for t in range(T):
            frame = torch.zeros(1, S, S)
            for i, img in enumerate(imgs):
                cx = int(positions[t, i, 0].item())
                cy = int(positions[t, i, 1].item())
                frame += torch.roll(img, shifts=(cy, cx), dims=(1, 2))
            seq[t] = frame.clamp(0.0, 1.0)

        if self.transform:
            seq = self.transform(seq)

        # 5. Build output
        lbl_out = (labels[0] if N == 1
                   else torch.tensor(labels, dtype=torch.long))
        out = [seq, lbl_out]
        if self.return_motion:
            out.append(motions)           # (T, N, 2)
        if self.return_positions:
            out.append(positions % S)     # (T, N, 2) wrapped
        return tuple(out)

    # ------------------------------------------------------------------
    # Frame helper
    # ------------------------------------------------------------------

    def _pad_digit(self, img, S):
        """Centre a (1,h,w) digit on a (1,S,S) black canvas."""
        _, h, w    = img.shape
        pad_top    = (S - h) // 2
        pad_bottom = S - h - pad_top
        pad_left   = (S - w) // 2
        pad_right  = S - w - pad_left
        return F.pad(img, (pad_left, pad_right, pad_top, pad_bottom))

    # ------------------------------------------------------------------
    # Initial position sampling
    # ------------------------------------------------------------------

    def _toroidal_dist(self, ax, ay, bx, by):
        """
        Euclidean distance on a toroidal canvas of size self.image_size.

        min(|ax-bx|, S-|ax-bx|) picks the shorter of the two paths
        around the torus in each axis independently.  This is the correct
        periodic distance — two digits that are 2 pixels apart because
        one is at x=1 and the other at x=S-1 have distance 2, not S-2.
        """
        S  = self.image_size
        dx = min(abs(ax - bx), S - abs(ax - bx))
        dy = min(abs(ay - by), S - abs(ay - by))
        return np.sqrt(dx * dx + dy * dy)

    def _position_is_valid(self, cx, cy, existing):
        """
        True if (cx, cy) is at least threshold pixels from every existing
        position in toroidal distance.
        """
        digit_sz  = 28
        threshold = float(self.min_center_distance)
        if self.reject_overlap:
            threshold = max(threshold, float(digit_sz))

        for pcx, pcy in existing:
            if self._toroidal_dist(cx, cy, pcx, pcy) < threshold:
                return False
        return True

    def _sample_initial_positions(self):
        """
        Sample (cx, cy) centre positions for each digit.

        Strategy
        --------
        1. Try rejection sampling (up to max_tries per digit).
        2. If that fails, use the guaranteed-separation fallback:
           place digit i at the torus-diagonal offset from digit 0.

           offset_i = (i * S // N, i * S // N)

           For N=2 this gives toroidal distance S*sqrt(2)/2, which
           exceeds the 28-pixel digit size for all S >= 40.

        The fallback NEVER produces an unconstrained placement (the
        previous bug), so overlaps only occur when S is genuinely too
        small to fit N digits.
        """
        S = self.image_size
        N = self.num_digits
        positions = []

        for i in range(N):

            # --- rejection sampling ---
            placed = False
            for _ in range(self.max_tries):
                cx = self._randint(0, S)
                cy = self._randint(0, S)
                if self._position_is_valid(cx, cy, positions):
                    positions.append((cx, cy))
                    placed = True
                    break

            if not placed:
                # --- guaranteed-separation fallback ---
                #
                # Anchor: digit 0's position (or random if i==0, which
                # shouldn't happen but is handled for safety).
                if positions:
                    ax, ay = positions[0]
                else:
                    ax = self._randint(0, S)
                    ay = self._randint(0, S)

                # Offset digit i by (i * S//N, i * S//N) on the torus.
                # For N=2:  digit 1 is at antipodal-ish point, toroidal
                # dist = S/sqrt(2) ≈ 0.707*S, e.g. ~32 px for S=45.
                cx = (ax + i * S // N) % S
                cy = (ay + i * S // N) % S
                positions.append((cx, cy))

        return positions

    # ------------------------------------------------------------------
    # Motion trajectory generation
    # ------------------------------------------------------------------

    def _generate_motion_trajectory(self):
        """
        Generate the (T, N, 2) velocity trajectory.

        freeze_after
        ------------
        If self.freeze_after is not None, the velocity at step
        (freeze_after - 1) is copied to all subsequent steps.

        For the length-generalisation experiment set freeze_after equal
        to input_frames so that:
          - encoder frames 0 .. input_frames-1 have time-varying velocity
          - decoder frames input_frames .. T-1 have constant velocity
          - the constant velocity is whatever was active at the boundary

        The model's v_last (frozen in the decoder at inference) will then
        match the GT continuation exactly.
        """
        T = self.seq_len
        N = self.num_digits

        # Harmonic is parametric rather than step-by-step: the whole
        # trajectory follows from a handful of per-digit constants, so it does
        # not use the initial-velocity draw or the segment machinery below.
        # Branching out here (rather than after them) also keeps it from
        # consuming RNG draws it would only discard, which would shift the
        # stream for every other mode and silently change existing fixed
        # benchmarks -- the same reason accel_sign is drawn conditionally.
        if self.motion_mode == "harmonic":
            return self._apply_freeze(self._generate_harmonic_trajectory(T, N))

        motions = torch.zeros(T, N, 2, dtype=torch.long)

        # Initial velocities (distinct if required)
        forbidden = []
        for i in range(N):
            v = self._sample_velocity(
                forbidden=forbidden if self.require_distinct_velocities else []
            )
            motions[0, i, 0] = v[0]
            motions[0, i, 1] = v[1]
            forbidden.append(v)

        # Constant mode: replicate t=0 across all steps then apply freeze
        if self.motion_mode == "constant":
            motions[1:] = motions[0]
            # freeze_after has no extra effect here but is applied uniformly
            return self._apply_freeze(motions)

        # Piecewise / stochastic / accelerate: evolve step by step
        segment_remaining = [
            self._randint(self.min_segment, self.max_segment + 1)
            for _ in range(N)
        ]
        # accelerate mode: sign per digit, +1 to speed up (|v| ramps outward
        # toward max_speed) or -1 to slow down. Randomly signed acceleration
        # would largely cancel across digits and leave |v| almost flat, which
        # is not what this mode is for.
        #
        # Drawn ONLY in accelerate mode: self._choice advances the RNG, so
        # sampling it unconditionally would shift the random stream for every
        # other mode and silently change previously-generated fixed benchmarks.
        accel_sign = ([self._choice([+1, +1, -1]) for _ in range(N)]
                      if self.motion_mode == "accelerate" else None)

        for t in range(1, T):
            for i in range(N):
                current = (int(motions[t-1, i, 0]), int(motions[t-1, i, 1]))
                new_v   = current

                if self.motion_mode == "piecewise":
                    segment_remaining[i] -= 1
                    if segment_remaining[i] <= 0:
                        new_v = self._update_velocity(current)
                        segment_remaining[i] = self._randint(
                            self.min_segment, self.max_segment + 1
                        )

                elif self.motion_mode == "stochastic":
                    if self._random() < self.p_change:
                        new_v = self._update_velocity(current)

                elif self.motion_mode == "accelerate":
                    segment_remaining[i] -= 1
                    if segment_remaining[i] <= 0:
                        s = accel_sign[i]
                        # step each component away from (s=+1) or toward (s=-1)
                        # zero, so the SPEED ramps rather than the direction
                        # wandering
                        step = lambda c: c + s * (1 if c >= 0 else -1)
                        cand = (min(max(step(current[0]), -self.max_speed), self.max_speed),
                                min(max(step(current[1]), -self.max_speed), self.max_speed))
                        # (0, 0) is not on the velocity grid; hold instead of stopping
                        new_v = current if cand == (0, 0) else cand
                        # reverse at the limits so |v| ramps up and down instead
                        # of saturating at max_speed and sticking there
                        speed = max(abs(new_v[0]), abs(new_v[1]))
                        if speed >= self.max_speed:
                            accel_sign[i] = -1
                        elif speed <= 1:
                            accel_sign[i] = +1
                        segment_remaining[i] = self._randint(
                            self.min_segment, self.max_segment + 1
                        )

                else:
                    raise ValueError(
                        f"Unknown motion_mode '{self.motion_mode}'. Choose from "
                        f"'constant', 'piecewise', 'stochastic', 'accelerate'."
                    )

                motions[t, i, 0] = new_v[0]
                motions[t, i, 1] = new_v[1]

        return self._apply_freeze(motions)


    def _apply_freeze(self, motions):
        """
        Freeze the velocity from step freeze_after - 1 onward, using the value
        at step freeze_after - 2:

            motions[freeze_after - 1 :] = motions[freeze_after - 2]

        Two-phase trajectory, for freeze_after = f:
            phase 1  (steps 0 .. f-2)   : time-varying
            phase 2  (steps f-1 .. T-1) : constant

        Note the indices: with f = input_frames the context is frames
        0 .. f-1, whose transitions are motions[0 .. f-2]. Freezing from f-1
        with the value from f-2 therefore makes the velocity that governs the
        ENTIRE rollout identical to the last transition visible in the context.
        That is deliberate -- it is what lets a decoder that freezes its last
        encoder velocity match the ground-truth continuation exactly, instead
        of having to guess a value it was never shown.
        """
        if self.freeze_after is None:
            return motions

        T = motions.shape[0]
        f = int(np.clip(self.freeze_after, 1, T))

        if f >= 2:
            last_v = motions[f - 2].clone()
            motions[f - 1:] = last_v.unsqueeze(0).expand(T - (f - 1), -1, -1)

        return motions

    # ------------------------------------------------------------------
    # Harmonic (parametric, deterministic-given-parameters) motion
    # ------------------------------------------------------------------

    def _generate_harmonic_trajectory(self, T, N):
        """
        Velocity as a sum of a constant drift and a sinusoid, per axis:

            vx(t) = dx + round( Ax * cos(wx t + px) )
            vy(t) = dy + round( Ay * cos(wy t + py) )

        Why this family, and not another "changing velocity" mode
        ---------------------------------------------------------
        A velocity predictor that is exactly motion-equivariant can only see
        the history of velocity DIFFERENCES du = u_t - u_{t-1}; the absolute
        velocity is invisible to it by construction (that is what buys the
        equivariance). So a dynamics is learnable by such a model iff the next
        increment is a function of past increments.

        Any linear time-invariant recurrence has that property: the difference
        operator commutes with LTI operators, so if u obeys the recurrence then
        du obeys the SAME one. Sinusoids are the canonical such signals, which
        is why this family is the right target and why the older modes are not:

            constant             du = 0            -- learnable, but freezing
                                                      the last velocity is
                                                      already optimal
            piecewise/stochastic increments i.i.d. -- no signal at all; the
                                                      best prediction IS freeze
            accelerate           reverses when |v| -- NOT a function of du:
                                 hits max_speed       the switch depends on the
                                                      absolute speed, which an
                                                      equivariant head cannot
                                                      see

        Rounding makes the quantized system only approximately LTI, but the
        trajectory stays fully determined by the du history: a model-free
        nearest-neighbour predictor matched on context increments alone
        recovers the rollout to ~6 px of end-position error against ~42 px for
        freezing (36 px frame, context 15, rollout 10).

        Shapes (sampled per digit from harmonic_shapes)
        ----------------------------------------------
            "constant"  : Ax = Ay = 0. Pure constant flow -- the degenerate
                          member, kept in the family so one dataset can mix
                          flows and oscillations.
            "orbit"     : both axes share one amplitude and frequency, a
                          quarter period apart, so u ROTATES at a constant rate
                          and the digit traces a circle. |u| is constant; only
                          the direction turns. The cleanest case: u_{t+1} =
                          R(w) u_t implies du_{t+1} = R(w) du_t, so a single
                          parameter is identifiable from two consecutive
                          increments.
            "axis"      : one axis oscillates, the other carries only drift --
                          constant flow in one direction, sinusoidal in the
                          other.
            "lissajous" : both axes oscillate with independent amplitude,
                          frequency and phase. Direction AND magnitude change;
                          two frequencies to identify instead of one.

        Drift
        -----
        The offset (dx, dy) is INTEGER, which makes it exactly separable from
        the rounded oscillation: round(d + A cos) == d + round(A cos). It
        therefore cancels completely in du, so it is invisible to an
        equivariant velocity model -- it costs such a model nothing while
        making the path a looping trochoid and the motion strictly richer for
        any model that has to carry velocity implicitly. That is the whole
        inductive-bias argument, made visible in the data.

        Speed budget
        ------------
        |d| + peak|round(A cos)| <= max_speed BY CONSTRUCTION, so the velocity
        is never clipped. This matters more than it looks: a clip is a rule
        about the ABSOLUTE speed, exactly the kind of non-equivariant switch
        that makes "accelerate" unlearnable here. Reintroducing one would
        undo the point of the mode.

        (0, 0) is allowed, unlike in the random modes: here a momentary pause
        is a predictable part of a deterministic trajectory rather than a
        degenerate "digit stopped" state, and the model can anticipate it.
        """
        t = np.arange(T)
        motions = torch.zeros(T, N, 2, dtype=torch.long)

        seen = []
        for i in range(N):
            # require_distinct_velocities is about the FIRST frame: the K
            # tracking slots are bootstrapped from one frame pair, so digits
            # sharing v(0) cannot be separated there. Later coincidences are
            # fine -- the trajectories diverge again on their own.
            for _ in range(max(1, self.max_tries)):
                vx, vy = self._sample_harmonic_digit(t)
                v0 = (int(vx[0]), int(vy[0]))
                if not self.require_distinct_velocities or v0 not in seen:
                    break
            seen.append(v0)
            motions[:, i, 0] = torch.from_numpy(vx)
            motions[:, i, 1] = torch.from_numpy(vy)

        return motions

    def _sample_harmonic_digit(self, t):
        """One digit's parameters -> its (T,) integer vx, vy. See
        _generate_harmonic_trajectory for what the shapes mean."""
        ms    = self.max_speed
        shape = self._choice(self.harmonic_shapes)

        def amp():
            lo, hi = self.harmonic_amp
            return (lo + (hi - lo) * self._random()) * ms

        def omega():
            # Integer period => the velocity sequence is exactly periodic,
            # which is the most learnable version of this signal.
            period = self._randint(self.harmonic_period[0],
                                   self.harmonic_period[1] + 1)
            return 2.0 * np.pi / period * self._choice([-1, +1])

        def phase():
            return 2.0 * np.pi * self._random()

        if shape == "constant":
            ax = ay = 0.0
            wx = wy = px = py = 0.0
        elif shape == "orbit":
            a, w, ph = amp(), omega(), phase()
            ax = ay = a
            wx = wy = w
            px, py = ph, ph - np.pi / 2      # (cos, sin) -> a rotating vector
        elif shape == "axis":
            a, w, ph = amp(), omega(), phase()
            if self._choice([0, 1]) == 0:
                ax, ay = a, 0.0
            else:
                ax, ay = 0.0, a
            wx = wy = w
            px = py = ph
        else:                                 # "lissajous"
            ax, ay = amp(), amp()
            wx, wy = omega(), omega()
            px, py = phase(), phase()

        osc_x = np.round(ax * np.cos(wx * t + px)).astype(np.int64)
        osc_y = np.round(ay * np.cos(wy * t + py)).astype(np.int64)

        # round() is monotone, so the peak of round(A cos .) is exactly
        # round(A) -- the leftover budget below is therefore tight, not
        # approximate, and no clipping is ever needed.
        dx = self._harmonic_drift(ax, shape)
        dy = self._harmonic_drift(ay, shape)

        if shape == "constant" and (dx, dy) == (0, 0):
            # A "constant" digit with zero drift would not move at all. Draw
            # from the velocity grid instead, which already excludes (0, 0).
            dx, dy = self._sample_velocity()

        return osc_x + dx, osc_y + dy

    def _harmonic_drift(self, amplitude, shape):
        """Integer offset that fits in whatever speed budget the oscillation
        on this axis leaves. Always drawn for "constant" (it IS the velocity
        there); otherwise only when harmonic_drift is on."""
        if shape != "constant" and not self.harmonic_drift:
            return 0
        budget = self.max_speed - int(np.round(amplitude))
        if budget <= 0:
            return 0
        return self._randint(-budget, budget + 1)

    # ------------------------------------------------------------------
    # Velocity sampling helpers
    # ------------------------------------------------------------------

    def _sample_velocity(self, forbidden=None):
        forbidden  = forbidden or []
        candidates = [v for v in self.velocity_grid if v not in forbidden]
        if not candidates:
            return self._choice(self.velocity_grid)
        return self._choice(candidates)

    def _sample_neighbor_velocity(self, velocity):
        """
        "legacy"    : uniform over the valid neighbours of v. Degree-biased --
                      the stationary velocity distribution favours low-speed
                      interior states, so mean |v| drifts with difficulty.
        "symmetric" : propose v + eps with eps uniform over the eight unit
                      offsets, and HOLD if the proposal leaves the grid.
                      Symmetric, hence doubly stochastic, hence uniform
                      stationary distribution -- mean |v| is constant across
                      the whole difficulty sweep.

        A held proposal is not a change, so the realised switch count under
        "symmetric" is below p_change * T. That is expected.

        "legacy" is the default because "symmetric" draws a different number of
        RNG values per call and therefore does not reproduce previously
        generated fixed benchmarks byte-for-byte.
        """
        if self.neighbor_kernel == "symmetric":
            dx, dy = self._choice(self._unit_offsets)
            cand = (velocity[0] + dx, velocity[1] + dy)
            return cand if cand in self._velocity_set else velocity

        vx, vy = velocity
        neighbors = [
            u for u in self.velocity_grid
            if u != velocity
            and abs(u[0] - vx) <= 1
            and abs(u[1] - vy) <= 1
        ]
        if not neighbors:
            return self._sample_velocity(forbidden=[velocity])
        return self._choice(neighbors)

    def _update_velocity(self, velocity):
        if self.transition_mode == "uniform":
            return self._sample_velocity(forbidden=[velocity])
        if self._random() < self.smooth_probability:
            return self._sample_neighbor_velocity(velocity)
        return self._sample_velocity(forbidden=[velocity])

    # ------------------------------------------------------------------
    # RNG helpers
    # ------------------------------------------------------------------

    def _randint(self, low, high):
        if self.random:
            return int(np.random.randint(low, high))
        return int(self.rng.randint(low, high))

    def _choice(self, values):
        idx = (np.random.randint(len(values)) if self.random
               else self.rng.randint(len(values)))
        return values[idx]

    def _random(self):
        return np.random.rand() if self.random else self.rng.rand()

