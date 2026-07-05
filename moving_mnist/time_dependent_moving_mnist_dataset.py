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
    "constant"    : all digits move at their initial velocity forever.
    "piecewise"   : velocity is held for a random segment length, then
                    updated.  Segment lengths are uniform in
                    [min_segment, max_segment].
    "stochastic"  : at every step, each digit independently changes
                    velocity with probability p_change.

    Transition modes (how a new velocity is chosen)
    ------------------------------------------------
    "smooth"   : with probability smooth_probability, pick a neighbour
                 velocity (max Chebyshev distance 1); otherwise pick
                 uniformly at random.
    "uniform"  : always pick uniformly at random.

    Bootstrap-friendly initialisation
    ----------------------------------
    For the first two frames (t=0, t=1), velocity estimation by phase
    correlation requires:
        (a) distinct per-digit velocities  → require_distinct_velocities
        (b) minimum spatial separation     → min_center_distance
        (c) no pixel-level overlap         → reject_overlap
    These constraints are enforced at t=0.  Because velocities are
    distinct, digits tend to remain separated at t=1 as well.

    Returns (always):
        seq    : (T, 1, H, W)   float32 in [0, 1]
        labels : int or LongTensor(N)

    Optional extra returns (controlled by flags):
        motions   : (T, N, 2)  LongTensor  [vx, vy] at each time step
                    motions[t, i] = velocity of digit i at time t,
                    i.e. the shift applied to go from frame t to frame t+1.
        positions : (T, N, 2)  LongTensor  [cx, cy] centre positions
                    (wrapped to [0, image_size) ).

    Velocity convention (matches PhaseCorrelation and MEConvLSTM):
        dim 0 → vx (horizontal, col shift)
        dim 1 → vy (vertical,   row shift)
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

        min_center_distance=12,
        reject_overlap=True,
        require_distinct_velocities=True,

        return_motion=True,
        return_positions=False,

        transform=None,
        download=True,

        random=True,
        seed=42,

        max_tries=100,
    ):
        super().__init__()

        self.mnist    = MNIST(root=root, train=train, download=download)
        self.to_tensor = ToTensor()

        self.seq_len    = seq_len
        self.image_size = image_size
        self.num_digits = num_digits
        self.max_speed  = max_speed

        self.motion_mode      = motion_mode
        self.transition_mode  = transition_mode
        self.min_segment      = min_segment
        self.max_segment      = max_segment
        self.p_change         = p_change
        self.smooth_probability = smooth_probability

        self.min_center_distance         = min_center_distance
        self.reject_overlap              = reject_overlap
        self.require_distinct_velocities = require_distinct_velocities

        self.return_motion    = return_motion
        self.return_positions = return_positions

        self.transform = transform
        self.random    = random
        self.max_tries = max_tries

        self.rng = np.random.RandomState(seed)

        # ------------------------------------------------------------------
        # Velocity grid: all integer (vx, vy) within max_speed, zero excluded
        # ------------------------------------------------------------------
        self.velocity_grid = [
            (vx, vy)
            for vx in range(-max_speed, max_speed + 1)
            for vy in range(-max_speed, max_speed + 1)
            if (vx, vy) != (0, 0)
        ]

        # ------------------------------------------------------------------
        # Optional difficulty preset — overrides the relevant parameters
        # ------------------------------------------------------------------
        if motion_difficulty is not None:
            d = float(np.clip(motion_difficulty, 0.0, 1.0))
            self.p_change           = 0.02 + 0.48 * d
            self.min_segment        = int(round(8  - 5 * d))
            self.max_segment        = int(round(12 - 6 * d))
            self.smooth_probability = 0.95 - 0.45 * d

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        S = self.image_size
        N = self.num_digits
        T = self.seq_len

        # ---- 1. Sample and pad digits --------------------------------
        imgs, labels = [], []
        for _ in range(N):
            idx     = self._randint(0, len(self.mnist))
            img_pil, lbl = self.mnist[idx]
            imgs.append(self._pad_digit(self.to_tensor(img_pil), S))
            labels.append(lbl)

        # ---- 2. Sample initial positions and motion trajectory -------
        init_pos = self._sample_initial_positions()   # list of N (cx, cy)
        motions  = self._generate_motion_trajectory() # (T, N, 2) [vx, vy]

        # ---- 3. Compute centre position at every time step -----------
        # positions[t, i] = init_pos[i] + sum(motions[0:t, i])
        # Achieved by prepending a zero row to the cumulative sum.
        cum = torch.cumsum(motions, dim=0)            # (T, N, 2)
        cum = torch.cat([torch.zeros(1, N, 2, dtype=torch.long), cum[:-1]], dim=0)
        # cum[t] = sum(motions[0:t]),  cum[0] = 0

        init_tensor = torch.tensor(init_pos, dtype=torch.long)   # (N, 2)
        positions   = cum + init_tensor.unsqueeze(0)              # (T, N, 2)

        # ---- 4. Render frames ----------------------------------------
        seq = torch.zeros(T, 1, S, S, dtype=imgs[0].dtype)

        for t in range(T):
            frame = torch.zeros(1, S, S)
            for i, img in enumerate(imgs):
                cx = int(positions[t, i, 0].item())
                cy = int(positions[t, i, 1].item())
                # torch.roll handles wrap-around automatically
                frame += torch.roll(img, shifts=(cy, cx), dims=(1, 2))
            seq[t] = frame.clamp(0.0, 1.0)

        if self.transform:
            seq = self.transform(seq)

        # ---- 5. Build output tuple -----------------------------------
        lbl_out = (labels[0] if N == 1
                   else torch.tensor(labels, dtype=torch.long))

        out = [seq, lbl_out]

        if self.return_motion:
            out.append(motions)                    # (T, N, 2) [vx, vy]

        if self.return_positions:
            out.append(positions % S)              # (T, N, 2) wrapped to canvas

        return tuple(out)

    # ------------------------------------------------------------------
    # Frame helpers
    # ------------------------------------------------------------------

    def _pad_digit(self, img, S):
        """
        Centre a (1, h, w) MNIST digit on a (1, S, S) black canvas.
        After padding, torch.roll(img, shifts=(cy, cx), dims=(1,2))
        places the digit centre at pixel (cy % S, cx % S).
        """
        _, h, w    = img.shape
        pad_top    = (S - h) // 2
        pad_bottom = S - h - pad_top
        pad_left   = (S - w) // 2
        pad_right  = S - w - pad_left
        return F.pad(img, (pad_left, pad_right, pad_top, pad_bottom))

    # ------------------------------------------------------------------
    # Initial position sampling
    # ------------------------------------------------------------------

    def _sample_initial_positions(self):
        """
        Sample (cx, cy) centre positions for each digit on a toroidal
        canvas of size image_size × image_size.

        Enforces:
            min_center_distance : minimum Chebyshev / Euclidean distance
                                  (periodic) between any two centres.
            reject_overlap      : additionally require distance >= 28
                                  (MNIST digit width) to prevent overlap.

        Falls back to unconstrained placement if max_tries is exceeded
        (avoids infinite loops when constraints are very tight).
        """
        S         = self.image_size
        digit_sz  = 28
        positions = []

        for _ in range(self.num_digits):

            placed = False

            for _try in range(self.max_tries):

                cx = self._randint(0, S)
                cy = self._randint(0, S)

                valid = True

                for pcx, pcy in positions:
                    # Periodic (toroidal) Euclidean distance
                    dx   = min(abs(cx - pcx), S - abs(cx - pcx))
                    dy   = min(abs(cy - pcy), S - abs(cy - pcy))
                    dist = np.sqrt(dx * dx + dy * dy)

                    threshold = float(self.min_center_distance)
                    if self.reject_overlap:
                        threshold = max(threshold, float(digit_sz))

                    if dist < threshold:
                        valid = False
                        break

                if valid:
                    positions.append((cx, cy))
                    placed = True
                    break

            if not placed:
                # Constraints could not be satisfied; fall back gracefully.
                positions.append((self._randint(0, S), self._randint(0, S)))

        return positions   # list of N (cx, cy)

    # ------------------------------------------------------------------
    # Motion trajectory generation
    # ------------------------------------------------------------------

    def _generate_motion_trajectory(self):
        """
        Generate the complete per-digit velocity trajectory.

        Returns
        -------
        motions : LongTensor (T, N, 2)
            motions[t, i] = (vx, vy) of digit i at time step t.
            This is the shift applied to go from frame t to frame t+1,
            so position[t+1] = position[t] + motions[t].
        """
        T = self.seq_len
        N = self.num_digits

        motions = torch.zeros(T, N, 2, dtype=torch.long)

        # ------ initial velocities (distinct if requested) ------
        forbidden = []
        for i in range(N):
            v = self._sample_velocity(
                forbidden=forbidden if self.require_distinct_velocities else []
            )
            motions[0, i, 0] = v[0]   # vx
            motions[0, i, 1] = v[1]   # vy
            forbidden.append(v)

        # ------ constant velocity: done ------
        if self.motion_mode == "constant":
            motions[1:] = motions[0]
            return motions

        # ------ piecewise / stochastic ------
        # For piecewise, track how many steps remain in each segment.
        segment_remaining = [
            self._randint(self.min_segment, self.max_segment + 1)
            for _ in range(N)
        ]

        for t in range(1, T):
            for i in range(N):

                current = (
                    int(motions[t - 1, i, 0]),
                    int(motions[t - 1, i, 1]),
                )
                new_v = current   # default: keep velocity

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

                else:
                    raise ValueError(
                        f"Unknown motion_mode '{self.motion_mode}'. "
                        f"Choose from 'constant', 'piecewise', 'stochastic'."
                    )

                motions[t, i, 0] = new_v[0]
                motions[t, i, 1] = new_v[1]

        return motions

    # ------------------------------------------------------------------
    # Velocity sampling helpers
    # ------------------------------------------------------------------

    def _sample_velocity(self, forbidden=None):
        """
        Uniform random velocity from the grid, excluding forbidden entries.
        """
        forbidden  = forbidden or []
        candidates = [v for v in self.velocity_grid if v not in forbidden]
        if not candidates:
            # Forbidden covers everything — just sample freely.
            return self._choice(self.velocity_grid)
        return self._choice(candidates)

    def _sample_neighbor_velocity(self, velocity):
        """
        Sample a velocity within Chebyshev distance 1 of the current one,
        excluding the current velocity itself.  Falls back to random if
        no neighbour exists on the grid.
        """
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
        """
        Choose the next velocity according to the transition mode.

        "uniform" : pick uniformly at random (excluding current).
        "smooth"  : with probability smooth_probability pick a neighbour;
                    otherwise pick uniformly at random.
        """
        if self.transition_mode == "uniform":
            return self._sample_velocity(forbidden=[velocity])

        # smooth (default)
        if self._random() < self.smooth_probability:
            return self._sample_neighbor_velocity(velocity)
        return self._sample_velocity(forbidden=[velocity])

    # ------------------------------------------------------------------
    # RNG helpers (support both reproducible and fully random modes)
    # ------------------------------------------------------------------

    def _randint(self, low, high):
        """Uniform integer in [low, high)."""
        if self.random:
            return int(np.random.randint(low, high))
        return int(self.rng.randint(low, high))

    def _choice(self, values):
        """Uniform random choice from a sequence."""
        idx = (np.random.randint(len(values)) if self.random
               else self.rng.randint(len(values)))
        return values[idx]

    def _random(self):
        """Uniform float in [0, 1)."""
        return np.random.rand() if self.random else self.rng.rand()
