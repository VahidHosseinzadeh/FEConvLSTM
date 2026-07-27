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
    "accelerate"  : each digit gets a constant acceleration vector; every
                    min_segment..max_segment steps the velocity is incremented
                    by it and clipped to the grid, so |v| ramps rather than
                    jumping. Unlike the other modes the velocity sequence is
                    systematic, not a random walk -- use it to test motion whose
                    *magnitude* changes over the context. transition_mode,
                    smooth_probability and p_change are unused here.

    Which parameters apply to which mode
    ------------------------------------
        min_segment/max_segment : piecewise, accelerate
        p_change                : stochastic only
        transition_mode         : piecewise, stochastic
        smooth_probability      : only when transition_mode == "smooth"
                                  ("smooth" with probability 0.0 is exactly
                                   equivalent to "uniform")

    All modes support freeze_after (see below).

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

        # Optional difficulty preset
        if motion_difficulty is not None:
            d = float(np.clip(motion_difficulty, 0.0, 1.0))
            self.p_change           = 0.02 + 0.48 * d
            self.min_segment        = int(round(8  - 5 * d))
            self.max_segment        = int(round(12 - 6 * d))
            self.smooth_probability = 0.95 - 0.45 * d

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
    # Velocity sampling helpers
    # ------------------------------------------------------------------

    def _sample_velocity(self, forbidden=None):
        forbidden  = forbidden or []
        candidates = [v for v in self.velocity_grid if v not in forbidden]
        if not candidates:
            return self._choice(self.velocity_grid)
        return self._choice(candidates)

    def _sample_neighbor_velocity(self, velocity):
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

