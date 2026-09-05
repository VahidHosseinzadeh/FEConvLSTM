"""
Contract tests for motion_mode="harmonic".

The point of this motion family is that its future velocity is predictable
from the VELOCITY HISTORY ALONE -- specifically from the history of velocity
differences du, which is the only thing an exactly motion-equivariant dynamics
head is allowed to see (velocity_dynamics_model.py). Every test here guards
one of the properties that makes that true; if one fails, the mode has stopped
being a valid target for the head and the experiment it exists for is invalid.

    cd moving_mnist && pytest tests/test_harmonic_motion.py -v
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from time_dependent_moving_mnist_dataset import TDMovingMNISTDataset  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SHAPES = ("constant", "orbit", "axis", "lissajous")
CTX, ROLL = 15, 10


def build(n_digits=2, max_speed=4, seed=0, **kw):
    cfg = dict(root=ROOT, train=True, seq_len=CTX + ROLL, num_digits=n_digits,
               image_size=36, max_speed=max_speed, motion_mode="harmonic",
               freeze_after=None, min_center_distance=20, reject_overlap=True,
               require_distinct_velocities=True, return_motion=True,
               download=False, random=False, seed=seed, max_tries=300)
    cfg.update(kw)
    return TDMovingMNISTDataset(**cfg)


def motions(ds, n=120):
    """(n * n_digits, T, 2) integer velocity trajectories, one row per digit."""
    m = torch.stack([ds[i][2] for i in range(n)])          # (n, T, digits, 2)
    return m.permute(0, 2, 1, 3).reshape(-1, m.shape[1], 2).numpy()


# ----------------------------------------------------------------------
# Representability: integers, and never clipped
# ----------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("max_speed", [2, 4])
def test_velocities_are_integers_within_the_speed_budget(shape, max_speed):
    """
    Two things at once, and the second is the load-bearing one.

    Integer: the renderer places digits with an integer torch.roll, so a
    fractional velocity would be a ground truth the pixels never actually
    show, and the head would be scored against a fiction.

    Never clipped: a clip is a rule about the ABSOLUTE speed, which is exactly
    the kind of non-equivariant switch that makes "accelerate" unlearnable for
    a du-only head. The drift budget is meant to make clipping unnecessary by
    construction -- if this fails, the mode has quietly acquired the same
    defect it was built to avoid.
    """
    m = motions(build(max_speed=max_speed, harmonic_shapes=(shape,)))
    assert m.dtype == np.int64
    assert np.abs(m).max() <= max_speed, \
        f"{shape}: |v| reached {np.abs(m).max()} > max_speed={max_speed}"


def test_constant_shape_never_stands_still():
    """A "constant" digit whose drift happened to be (0, 0) would not move at
    all, which is not a motion sample. It is redrawn from the velocity grid."""
    m = motions(build(harmonic_shapes=("constant",)))
    assert (np.abs(m).sum(-1) > 0).all()
    # and it is genuinely constant
    assert (np.diff(m, axis=1) == 0).all()


# ----------------------------------------------------------------------
# The equivariance-relevant properties
# ----------------------------------------------------------------------

@pytest.mark.parametrize("shape", ["orbit", "axis", "lissajous"])
def test_drift_cancels_exactly_in_the_velocity_differences(shape, monkeypatch):
    """
    The drift must be invisible to a du-only model, EXACTLY.

    This holds because the offset is an integer and is added after rounding:
    round(d + A cos) == d + round(A cos). If someone later rounds the sum
    instead, the drift starts leaking into du -- the head would then be asked
    to predict something that depends on the absolute velocity, which it
    cannot see, and the equivariance guarantee would quietly stop buying
    anything. Same RNG stream both times, so the two runs differ ONLY in drift.
    """
    def run(drift_value):
        ds = build(harmonic_shapes=(shape,))
        monkeypatch.setattr(ds, "_harmonic_drift",
                            lambda amplitude, shp: drift_value)
        return motions(ds, n=60)

    m0, m1 = run(0), run(2)
    assert not np.array_equal(m0, m1), "drift had no effect at all -- test is vacuous"
    assert np.array_equal(np.diff(m0, axis=1), np.diff(m1, axis=1)), \
        "drift leaked into du: it is no longer invisible to an equivariant head"


@pytest.mark.parametrize("shape", ["orbit", "axis", "lissajous"])
def test_trajectory_is_locally_predictable(shape):
    """
    First half of "there is something to learn": the velocity must be smooth
    enough that even a trivially lagged estimate tracks it, unlike the random
    modes where the best any predictor can do IS to freeze.

    This one is not sample-limited, so it is the honest evidence that the
    signal exists; the du-only test below then shows the signal survives
    throwing away the absolute velocity.
    """
    u = motions(build(harmonic_shapes=(shape,)), n=200)
    true, frozen = u[:, CTX:], np.repeat(u[:, CTX - 1:CTX], ROLL, axis=1)
    lag = u[:, CTX - 1:CTX + ROLL - 1]

    def end_err(p):
        return np.linalg.norm(np.cumsum(p - true, axis=1)[:, -1], axis=-1).mean()

    assert end_err(lag) < 0.35 * end_err(frozen), (
        f"{shape}: even a 1-step-lagged velocity gives {end_err(lag):.1f} px vs "
        f"frozen {end_err(frozen):.1f} px -- the motion is not smoothly predictable")


# NN in du-space is curse-of-dimensionality limited, and the shapes differ in
# how many parameters have to be matched: orbit and axis have 3 (amplitude,
# period, phase), lissajous has 6. Measured against a 800-digit pool, NN lands
# at 0.36x / 0.43x / 0.67x of the frozen error respectively, while the
# not-sample-limited lag bound is 0.15x for all three -- i.e. lissajous is
# harder to MATCH, not less predictable. Thresholds are set per shape with
# margin rather than at one optimistic value that lissajous would fail for a
# reason that says nothing about the data.
_NN_MAX_FRACTION_OF_FROZEN = {"orbit": 0.50, "axis": 0.55, "lissajous": 0.80}


@pytest.mark.parametrize("shape", ["orbit", "axis", "lissajous"])
def test_future_is_predictable_from_the_du_history_alone(shape):
    """
    The whole reason this mode exists.

    A model-free nearest-neighbour predictor that sees ONLY the context
    velocity DIFFERENCES -- no absolute velocity, no images -- must clearly
    beat a frozen decoder. That is exactly the information an equivariant
    dynamics head is restricted to, so if this fails no such head can win here
    and running the experiment on this data is pointless.

    A trained model fits the parametric family instead of memorising
    neighbours, so this is a lower bound on what is achievable, not an upper
    one: the same NN at a 20k-digit pool reaches 0.13x of frozen for orbit.
    """
    u = motions(build(harmonic_shapes=(shape,), seed=0), n=200)
    pool = motions(build(harmonic_shapes=(shape,), seed=7), n=400)

    def ctx_du(x):
        return np.diff(x[:, :CTX], axis=1).reshape(len(x), -1).astype(float)

    idx = np.argmin(((ctx_du(u)[:, None] - ctx_du(pool)[None]) ** 2).sum(-1), axis=1)
    pred = u[:, CTX - 1:CTX] + np.cumsum(np.diff(pool[:, CTX - 1:], axis=1)[idx], axis=1)
    frozen = np.repeat(u[:, CTX - 1:CTX], ROLL, axis=1)
    true = u[:, CTX:]

    def end_err(p):
        return np.linalg.norm(np.cumsum(p - true, axis=1)[:, -1], axis=-1).mean()

    limit = _NN_MAX_FRACTION_OF_FROZEN[shape]
    assert end_err(pred) < limit * end_err(frozen), (
        f"{shape}: du-history predictor {end_err(pred):.1f} px vs "
        f"frozen {end_err(frozen):.1f} px -- not enough headroom to test a "
        f"velocity dynamics head")


def test_constant_shape_has_no_headroom_by_design():
    """The sanity anchor at the other end: for pure constant flow, freezing
    the last encoder velocity is EXACTLY right, so the head has nothing to win
    and must not lose anything either. A nonzero number here would mean the
    trajectory is not actually constant."""
    u = motions(build(harmonic_shapes=("constant",)))
    frozen = np.repeat(u[:, CTX - 1:CTX], ROLL, axis=1)
    assert np.array_equal(frozen, u[:, CTX:])


# ----------------------------------------------------------------------
# Determinism, mixing, and not disturbing the other modes
# ----------------------------------------------------------------------

def test_periodic_and_deterministic_given_the_seed():
    """random=False must give byte-identical data across passes after
    reset_rng -- the fixed benchmark sets depend on it."""
    ds = build()
    a = motions(ds, n=40)
    ds.reset_rng()
    assert np.array_equal(a, motions(ds, n=40))


def test_mixed_shapes_actually_mix():
    """With all four shapes in the pool, some digits must be constant and some
    must not -- otherwise the sampler has collapsed onto one shape."""
    m = motions(build(harmonic_shapes=SHAPES), n=200)
    is_const = (np.diff(m, axis=1) == 0).all(axis=(1, 2))
    assert is_const.any() and (~is_const).any(), \
        f"shape mixing broken: {is_const.mean():.2f} constant"


def test_distinct_initial_velocities():
    """The K tracking slots are bootstrapped from one frame pair, so digits
    sharing v(0) cannot be separated there."""
    ds = build(n_digits=2, require_distinct_velocities=True)
    for i in range(60):
        v0 = ds[i][2][0]                      # (digits, 2)
        assert not torch.equal(v0[0], v0[1])


def test_other_modes_are_byte_identical():
    """Harmonic branches out before the shared initial-velocity draw, so it
    must not have shifted the RNG stream for any pre-existing mode. These are
    the fixed benchmark sets -- a shift here silently changes stored results."""
    for mode in ("constant", "piecewise", "stochastic", "accelerate"):
        ds = TDMovingMNISTDataset(
            root=ROOT, train=True, seq_len=CTX + ROLL, num_digits=2, image_size=36,
            max_speed=2, motion_mode=mode, transition_mode="smooth",
            min_segment=3, max_segment=6, freeze_after=None, min_center_distance=20,
            reject_overlap=True, require_distinct_velocities=True,
            return_motion=True, download=False, random=False, seed=123, max_tries=300)
        got = torch.stack([ds[i][2] for i in range(5)])
        # Regenerating from the same seed must reproduce it exactly; the point
        # is that constructing a harmonic dataset in between changes nothing.
        build(harmonic_shapes=SHAPES)[0]
        ds.reset_rng()
        assert torch.equal(got, torch.stack([ds[i][2] for i in range(5)])), mode


def test_rejects_unknown_shape():
    with pytest.raises(ValueError, match="unknown harmonic shape"):
        build(harmonic_shapes=("spiral",))
    with pytest.raises(ValueError, match="must not be empty"):
        build(harmonic_shapes=())
