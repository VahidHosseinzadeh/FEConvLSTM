"""
Continuous-valued velocity schedules, (T, 2) arrays of (vx, vy) in pixels per step.

Repo convention throughout: motion[t] is the displacement taking frame t to frame t+1 (the same
as TDMovingMNISTDataset's `motions`), so frame t sits at cumulative_displacement(motion)[t].

The Moving MNIST experiments use an INTEGER velocity lattice with i.i.d. switches -- the thing a
reviewer will call artificial. These schedules are the continuous counterparts:

  piecewise  -- held for hold[0]..hold[1] steps, then redrawn (optionally to a nearby velocity);
                the continuous analogue of motion_mode='piecewise'.
  ou         -- Ornstein-Uhlenbeck: smooth, correlated drift with time constant tau.
  rotating   -- constant speed, direction turning at a random rate: the roundabout case, where the
                velocity changes EVERY frame but smoothly.
  constant   -- one velocity for the whole sequence (a flow, not a motion).

`apply_freeze` reproduces TDMovingMNISTDataset._apply_freeze exactly, so the length-
generalisation protocol (the rollout continues at the last velocity visible in the context)
carries over unchanged.
"""
import numpy as np


def draw_velocity(rng, max_speed, min_speed=0.0, integer=False):
    """Uniform over the annulus min_speed <= |v| <= max_speed (area-uniform)."""
    if integer:
        m = int(np.floor(max_speed))
        while True:
            v = rng.integers(-m, m + 1, size=2).astype(np.float64)
            s = np.hypot(*v)
            if s > 0 and min_speed <= s <= max_speed:     # (0, 0) is never drawn, as in
                return v                                  # TDMovingMNIST's velocity grid
    r = np.sqrt(rng.uniform(min_speed ** 2, max_speed ** 2))
    th = rng.uniform(0, 2 * np.pi)
    return np.array([r * np.cos(th), r * np.sin(th)])


def piecewise(T, rng, max_speed, min_speed=0.0, hold=(3, 6), smooth_prob=0.0, smooth_step=1.0,
              integer=False, v0=None):
    """
    Held for a random hold[0]..hold[1] (inclusive) steps, then redrawn. With probability
    smooth_prob a change is a perturbation of at most smooth_step per component (clipped back
    into the speed annulus) instead of an unrelated jump.
    """
    out = np.zeros((T, 2))
    v = draw_velocity(rng, max_speed, min_speed, integer) if v0 is None else np.asarray(v0, float)
    left = int(rng.integers(hold[0], hold[1] + 1))
    for t in range(T):
        if left <= 0:
            if rng.random() < smooth_prob:
                for _ in range(50):
                    step = (rng.integers(-1, 2, size=2) if integer
                            else rng.uniform(-smooth_step, smooth_step, size=2))
                    cand = v + step
                    s = np.hypot(*cand)
                    if min_speed <= s <= max_speed and np.any(cand != v):
                        v = cand.astype(np.float64)
                        break
            else:
                v = draw_velocity(rng, max_speed, min_speed, integer)
            left = int(rng.integers(hold[0], hold[1] + 1))
        out[t] = v
        left -= 1
    return out


def ou(T, rng, max_speed, tau=6.0, sigma=None, v0=None):
    """
    Ornstein-Uhlenbeck velocity with mean 0, correlation time tau steps and stationary std
    sigma per component (default max_speed / 2); the norm is soft-clipped at max_speed.
    """
    sigma = max_speed / 2.0 if sigma is None else sigma
    a = np.exp(-1.0 / tau)
    v = rng.normal(0, sigma, 2) if v0 is None else np.asarray(v0, float)
    out = np.zeros((T, 2))
    for t in range(T):
        s = np.hypot(*v)
        out[t] = v if s <= max_speed else v * (max_speed / s)
        v = a * v + np.sqrt(1 - a * a) * rng.normal(0, sigma, 2)
    return out


def rotating(T, rng, max_speed, min_speed=None, omega_max=0.35, omega_change_prob=0.1):
    """Constant speed, heading turning by omega rad/step; omega occasionally redrawn."""
    min_speed = 0.5 * max_speed if min_speed is None else min_speed
    speed = rng.uniform(min_speed, max_speed)
    th = rng.uniform(0, 2 * np.pi)
    om = rng.uniform(-omega_max, omega_max)
    out = np.zeros((T, 2))
    for t in range(T):
        out[t] = speed * np.array([np.cos(th), np.sin(th)])
        th += om
        if rng.random() < omega_change_prob:
            om = rng.uniform(-omega_max, omega_max)
    return out


def constant(T, rng, max_speed, min_speed=0.0, integer=False):
    return np.tile(draw_velocity(rng, max_speed, min_speed, integer), (T, 1))


SCHEDULES = {"piecewise": piecewise, "ou": ou, "rotating": rotating, "constant": constant}


def make_schedule(kind, T, rng, max_speed, **kw):
    """Dispatch by name; unknown keyword arguments for a given schedule are ignored."""
    import inspect
    fn = SCHEDULES[kind]
    params = inspect.signature(fn).parameters
    return fn(T, rng, max_speed, **{k: v for k, v in kw.items() if k in params})


def apply_freeze(motion, freeze_after):
    """
    TDMovingMNISTDataset._apply_freeze, verbatim in meaning:

        motion[freeze_after - 1:] = motion[freeze_after - 2]

    With freeze_after = input_frames the context is frames 0..f-1, whose transitions are
    motion[0..f-2]; freezing from f-1 with the value at f-2 makes the velocity that governs the
    whole rollout identical to the last transition visible in the context.
    """
    if freeze_after is None:
        return motion
    motion = np.array(motion, copy=True)
    T = motion.shape[0]
    f = int(np.clip(freeze_after, 1, T))
    if f >= 2:
        motion[f - 1:] = motion[f - 2]
    return motion
