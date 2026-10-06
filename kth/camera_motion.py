"""
Camera motion for KTH clips: one trajectory of whole-frame translations per clip.

Conventions -- the repo's (moving_mnist/common_fate_moving_mnist_dataset.py)
--------------------------------------------------------------------------
A velocity is (vx, vy) in pixels per model frame. `motion[t]` is the step taking frame t to
frame t+1, the displacement is d[0] = 0, d[t] = sum(motion[:t]), and frame t is shown rolled by
d[t] (torch.roll(frame, (dy, dx))): the scene content moves by +v, i.e. a camera moving by -v.
Shifts are whole pixels and circular, exactly like Keller's KTH protocol, so for every model in
this repo (circular padding everywhere) the camera motion is an exact group action.

Modes
-----
none        no camera motion (Keller's V_0).
constant    one velocity per clip from V_R = {-R..R}^2, (0, 0) included. With keller_draws (the
            dataset handles it, because it needs the clip order) the velocities are Keller's own:
            np.random.RandomState(42), vx then vy, one draw per clip in split order -- so the
            constant condition reproduces his frames bit for bit.
piecewise / stochastic / accelerate
            the Moving MNIST velocity process itself (TDMovingMNISTDataset's
            _generate_motion_trajectory), run through a thin adapter that skips its MNIST
            loading. Two settings differ from Moving MNIST, both on purpose:
              * (0, 0) is ON the grid. Moving MNIST excludes it; V_R includes it, and a camera
                at rest is the most natural camera state there is.
              * transition 'uniform' is the KTH default. Simulated on V_1 with (0, 0), 15 steps
                per clip: 'smooth' + the legacy kernel (Moving MNIST's setting) puts 0.167 of all
                steps at rest instead of 1/9 = 0.111; 'smooth' + 'symmetric' is uniform but holds
                so often that 9% of clips never change; 'uniform' is uniform AND changes at every
                segment end (2.74 changes per clip), so the constant and piecewise conditions
                differ only in time variation.
shake       a periodic shake per axis, d(t) = A sin(2 pi t / P + phi), rounded to whole pixels
            (floor(x + 1/2)). A (px), P (frames) and phi are drawn per clip and per axis. With
            shake_vmax, A is capped at vmax / (2 sin(pi / P)), which bounds every continuous step
            by vmax and therefore every ROUNDED step too: vmax = 1 keeps the shake on V_1, i.e. on
            FEConvLSTM's 9-copy lattice. At the model's 12.5 frames/s (25 fps, step 2) a period
            of P frames is 12.5 / P Hz.
"""
import math

import numpy as np

from motion_benchmarks import _repo  # noqa: F401  (puts moving_mnist/ on sys.path)
from time_dependent_moving_mnist_dataset import TDMovingMNISTDataset  # noqa: E402

GENERATOR_MODES = ("piecewise", "stochastic", "accelerate")
MODES = ("none", "constant") + GENERATOR_MODES + ("shake",)


class _VelocityProcess(TDMovingMNISTDataset):
    """
    TDMovingMNISTDataset's velocity process without its MNIST.

    TDMovingMNISTDataset.__init__ loads MNIST, which a camera trajectory has no use for, so it is
    deliberately not called: only the attributes _generate_motion_trajectory and its helpers read
    are set. One moving layer (num_digits = 1); random=False, so every draw goes through
    self.rng -- which draw() points at the caller's RandomState (or at the np.random module).
    """

    def __init__(self, seq_len, max_speed, motion_mode, transition_mode, min_segment,
                 max_segment, smooth_probability=0.8, p_change=0.25, neighbor_kernel="legacy",
                 include_zero=True):
        self.seq_len = seq_len
        self.num_digits = 1
        self.max_speed = max_speed
        self.motion_mode = motion_mode
        self.transition_mode = transition_mode
        self.min_segment = min_segment
        self.max_segment = max_segment
        self.smooth_probability = smooth_probability
        self.p_change = p_change
        self.freeze_after = None
        self.require_distinct_velocities = False
        self.random = False
        self.rng = np.random.RandomState(0)
        self.velocity_grid = [(vx, vy)
                              for vx in range(-max_speed, max_speed + 1)
                              for vy in range(-max_speed, max_speed + 1)
                              if include_zero or (vx, vy) != (0, 0)]
        self.neighbor_kernel = neighbor_kernel
        self._velocity_set = set(self.velocity_grid)
        self._unit_offsets = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)
                              if (a, b) != (0, 0)]

    def draw(self, rng):
        """(seq_len, 2) int64 velocities (vx, vy) drawn from rng."""
        self.rng = rng
        return self._generate_motion_trajectory()[:, 0].numpy().astype(np.int64)


class CameraMotion:
    """
    A camera-motion law. draw(rng) returns one clip's (seq_len, 2) int64 velocity sequence.

    rng is a np.random.RandomState for fixed (seeded) trajectories, or the np.random module for
    fresh ones in DataLoader workers (PyTorch seeds numpy per worker).
    """

    def __init__(self, mode="none", v_range=1, seq_len=16, transition="uniform",
                 min_segment=3, max_segment=6, neighbor_kernel="legacy",
                 smooth_probability=0.8, p_change=0.25,
                 shake_amp=(1.0, 3.0), shake_period=(6.0, 16.0), shake_axes="both",
                 shake_vmax=1, keller_draws=True):
        if mode not in MODES:
            raise ValueError(f"unknown camera mode {mode!r}; expected one of {MODES}")
        if shake_axes not in ("both", "x", "y"):
            raise ValueError(f"shake_axes {shake_axes!r}: expected both, x or y")
        self.mode = mode
        self.v_range = int(v_range)
        self.seq_len = int(seq_len)
        self.keller_draws = bool(keller_draws)
        self.shake_amp = (float(shake_amp[0]), float(shake_amp[1]))
        self.shake_period = (float(shake_period[0]), float(shake_period[1]))
        self.shake_axes = shake_axes
        self.shake_vmax = shake_vmax
        self._process = None
        if mode in GENERATOR_MODES:
            self._process = _VelocityProcess(
                seq_len, self.v_range, mode, transition, min_segment, max_segment,
                smooth_probability=smooth_probability, p_change=p_change,
                neighbor_kernel=neighbor_kernel, include_zero=True)
        self.transition = transition
        self.min_segment, self.max_segment = min_segment, max_segment

    # ------------------------------------------------------------------
    @property
    def uses_keller_draws(self):
        return self.mode == "constant" and self.keller_draws

    def draw(self, rng):
        T = self.seq_len
        if self.mode == "none":
            return np.zeros((T, 2), dtype=np.int64)
        if self.mode == "constant":
            v = (rng.randint(-self.v_range, self.v_range + 1),
                 rng.randint(-self.v_range, self.v_range + 1))
            return np.tile(np.asarray(v, dtype=np.int64), (T, 1))
        if self.mode == "shake":
            v = np.zeros((T, 2), dtype=np.int64)
            for axis in (0, 1):
                if self.shake_axes == "both" or self.shake_axes == "xy"[axis]:
                    v[:, axis] = self._shake_axis(rng, T)
            return v
        return self._process.draw(rng)

    def _shake_axis(self, rng, T):
        """One axis of the shake: T whole-pixel steps of a rounded sinusoid."""
        a_lo, a_hi = self.shake_amp
        for _ in range(100):
            P = rng.uniform(*self.shake_period)
            cap = a_hi
            if self.shake_vmax:
                cap = min(a_hi, self.shake_vmax / (2.0 * math.sin(math.pi / P)))
            if cap >= a_lo:
                break
        A = rng.uniform(a_lo, cap) if cap > a_lo else cap
        phi = rng.uniform(0.0, 2.0 * math.pi)
        t = np.arange(T + 1)
        d = np.floor(A * np.sin(2.0 * math.pi * t / P + phi) + 0.5)
        return np.diff(d).astype(np.int64)

    def describe(self):
        if self.mode == "none":
            return "none (static camera)"
        if self.mode == "constant":
            src = "Keller's draws" if self.keller_draws else "seeded draws"
            return f"constant, one v per clip from V_{self.v_range} ({src})"
        if self.mode == "shake":
            return (f"shake, A in [{self.shake_amp[0]:g}, {self.shake_amp[1]:g}] px, "
                    f"P in [{self.shake_period[0]:g}, {self.shake_period[1]:g}] frames, "
                    f"axes {self.shake_axes}, |v| <= {self.shake_vmax or 'inf'}")
        return (f"{self.mode} on V_{self.v_range} incl. (0,0), transition {self.transition}, "
                f"segments {self.min_segment}-{self.max_segment}")


def keller_constant_velocities(n_clips, v_range):
    """
    Keller's constant-velocity draws, converted to the repo convention.

    His KTHVideoClips draws (vx, vy) = rng.randint(-R, R + 1) twice per clip from
    RandomState(42), in clip order, and shows frame t as frame[(y + vy t) mod H, (x + vx t) mod W]
    -- content moving by (-vx, -vy). Hence the sign flip. Returns (n_clips, 2) int64.
    """
    rng = np.random.RandomState(42)
    out = np.zeros((n_clips, 2), dtype=np.int64)
    for i in range(n_clips):
        vx = rng.randint(-v_range, v_range + 1)
        vy = rng.randint(-v_range, v_range + 1)
        out[i] = (-vx, -vy)
    return out


def displacements(v):
    """(T, 2) per-step velocities -> (T, 2) displacement of each frame, d[0] = 0."""
    d = np.zeros_like(v)
    d[1:] = np.cumsum(v[:-1], axis=0)
    return d
