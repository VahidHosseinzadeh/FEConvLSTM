"""
Characterise the `motion_difficulty` axis of TDMovingMNISTDataset.

`motion_difficulty=d` collapses the whole motion family onto one ordered scalar
by forcing motion_mode="stochastic" and varying the entropy of the velocity
process. The exponents are solved so that d = 0.5 IS the training regime, which
gives the sweep a reference point the previous linear preset did not have.

Run directly to print the table for the paper appendix:

    python moving_mnist/tests/test_motion_difficulty.py
"""
import math
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from time_dependent_moving_mnist_dataset import TDMovingMNISTDataset  # noqa: E402

DATA_ROOT = os.environ.get("MNIST_ROOT", str(_PKG / "data"))

D_GRID = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
N_TRAJ = 4000
SEQ_LEN = 100
MAX_SPEED = 2
NUM_DIGITS = 2
FREEZE = 15
SEED = 20240521

# Transitions motions[0 .. f-2] are the context ones (see _apply_freeze), so the
# free transitions are t = 1 .. f-2 and the boundary age of a change at index k
# is (f-2) - k + 1. A trajectory that never changes counts its initial draw at
# index 0, giving the maximum age f-1.
LAST_FREE_T = FREEZE - 2

# Simulated regression targets; see the module docstring in the dataset class.
EXPECTED = {
    #      p_change  p_smooth  mean_gap  entropy  age    d_v    switches
    0.00: (0.000, 1.000, math.inf, 0.000, 14.00, 0.00, 0.00),
    0.20: (0.030, 0.931, 32.86, 0.282, 11.48, 1.10, 0.40),
    0.35: (0.102, 0.871, 9.76, 0.791, 7.65, 1.18, 1.33),
    0.50: (0.222, 0.800, 4.50, 1.498, 4.39, 1.27, 2.90),
    0.65: (0.393, 0.713, 2.55, 2.362, 2.52, 1.39, 5.08),
    0.80: (0.616, 0.596, 1.62, 3.331, 1.63, 1.55, 8.01),
    1.00: (1.000, 0.000, 1.00, 4.523, 1.00, 2.42, 13.00),
}


def _dataset(d):
    return TDMovingMNISTDataset(
        root=DATA_ROOT, train=False, seq_len=SEQ_LEN, num_digits=NUM_DIGITS,
        image_size=36, max_speed=MAX_SPEED, motion_difficulty=d,
        freeze_after=FREEZE, download=False, random=False, seed=SEED,
        return_motion=True,
    )


def _conditional_entropy(joint):
    """H(v_t | v_{t-1}) in bits from an empirical joint count matrix."""
    row = joint.sum(axis=1)
    total = row.sum()
    if total == 0:
        return 0.0
    h = 0.0
    for k in np.flatnonzero(row):
        p = joint[k][joint[k] > 0] / row[k]
        h += (row[k] / total) * float(-(p * np.log2(p)).sum())
    return h


def measure(d, n_traj=N_TRAJ):
    """Velocity-process statistics over the pre-freeze (context) transitions."""
    ds = _dataset(d)
    ds.reset_rng()

    index = {v: i for i, v in enumerate(ds.velocity_grid)}
    k = len(ds.velocity_grid)
    joint = np.zeros((k, k), dtype=np.int64)

    ages, jumps, switches = [], [], []
    n_change = n_trans = 0
    for _ in range(n_traj):
        motions = ds._generate_motion_trajectory().numpy()
        for i in range(NUM_DIGITS):
            v = motions[:, i, :]
            last_change, count = 0, 0
            for t in range(1, LAST_FREE_T + 1):
                a, b = (int(v[t - 1][0]), int(v[t - 1][1])), (int(v[t][0]), int(v[t][1]))
                joint[index[a], index[b]] += 1
                n_trans += 1
                if a != b:
                    n_change += 1
                    count += 1
                    last_change = t
                    jumps.append(max(abs(b[0] - a[0]), abs(b[1] - a[1])))
            switches.append(count)
            ages.append(LAST_FREE_T - last_change + 1)

    p_emp = n_change / n_trans
    return dict(
        d=d,
        p_change=ds.p_change,                  # configured, exact
        p_smooth=ds.smooth_probability,        # configured, exact
        p_change_emp=p_emp,
        mean_gap=(1.0 / p_emp) if p_emp > 0 else math.inf,
        entropy=_conditional_entropy(joint),
        mean_age=float(np.mean(ages)),
        mean_dv=float(np.mean(jumps)) if jumps else 0.0,
        switches=float(np.mean(switches)),
        min_segment=ds.min_segment,
        max_segment=ds.max_segment,
    )


def _all():
    return [measure(d) for d in D_GRID]


def format_table(rows):
    out = ["|   d  | p_change | p_smooth | mean gap | H bits/frame | E[age] | E[|dv|] | switches |",
           "|------|----------|----------|----------|--------------|--------|---------|----------|"]
    for r in rows:
        gap = "   inf  " if math.isinf(r["mean_gap"]) else f"{r['mean_gap']:8.2f}"
        out.append(
            f"| {r['d']:.2f} | {r['p_change']:8.3f} | {r['p_smooth']:8.3f} | {gap} |"
            f" {r['entropy']:12.3f} | {r['mean_age']:6.2f} | {r['mean_dv']:7.2f} |"
            f" {r['switches']:8.2f} |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_anchor_points():
    """d = 0 is a pure flow, d = 0.5 is training, d = 1 is maximum entropy."""
    zero = measure(0.0)
    assert zero["p_change"] == 0.0
    assert zero["entropy"] == 0.0
    assert zero["mean_dv"] == 0.0
    assert zero["switches"] == 0.0

    train = measure(0.5)
    assert math.isclose(train["p_change"], 1 / 4.5, rel_tol=1e-3)
    assert math.isclose(train["p_smooth"], 0.80, rel_tol=1e-3)
    assert math.isclose(train["mean_gap"], 4.5, rel_tol=0.05)
    # d = 0.5 must also reproduce the training SEGMENT bounds, so a piecewise
    # run at the same d switches at the same expected rate
    assert (train["min_segment"], train["max_segment"]) == (3, 6)

    top = measure(1.0)
    assert top["p_change"] == 1.0
    assert math.isclose(top["entropy"], math.log2(23), rel_tol=0.02)


def test_monotonicity():
    rows = _all()
    up = ["entropy", "mean_dv", "switches"]
    down = ["mean_age"]
    for key in up:
        vals = [r[key] for r in rows]
        assert all(a < b for a, b in zip(vals, vals[1:])), f"{key} not increasing: {vals}"
    for key in down:
        vals = [r[key] for r in rows]
        assert all(a > b for a, b in zip(vals, vals[1:])), f"{key} not decreasing: {vals}"
    gaps = [r["mean_gap"] for r in rows]
    assert all(a > b for a, b in zip(gaps, gaps[1:])), f"mean gap not decreasing: {gaps}"


def test_regression_targets():
    for r in _all():
        exp = EXPECTED[round(r["d"], 2)]
        got = (r["p_change"], r["p_smooth"], r["mean_gap"], r["entropy"],
               r["mean_age"], r["mean_dv"], r["switches"])
        for name, g, e in zip(("p_change", "p_smooth", "mean_gap", "entropy",
                               "mean_age", "mean_dv", "switches"), got, exp):
            if math.isinf(e):
                assert math.isinf(g), f"d={r['d']} {name}: {g} != inf"
            elif e == 0.0:
                assert abs(g) < 1e-9, f"d={r['d']} {name}: {g} != 0"
            else:
                assert math.isclose(g, e, rel_tol=0.05), f"d={r['d']} {name}: {g} vs {e}"


def test_max_speed_is_off_the_difficulty_axis():
    """Difficulty must never become 'the velocity left the grid'."""
    ref = _dataset(0.0)
    for d in D_GRID:
        ds = _dataset(d)
        assert ds.max_speed == ref.max_speed
        assert ds.velocity_grid == ref.velocity_grid


def test_apply_difficulty_does_not_consume_rng():
    """Any RNG draw here would shift every previously generated benchmark."""
    ds = _dataset(0.0)
    ds.reset_rng()
    before = ds.rng.get_state()
    ds._apply_difficulty(0.73)
    after = ds.rng.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_forced_mode_warns_only_on_conflict():
    import warnings

    def build(**kw):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            TDMovingMNISTDataset(
                root=DATA_ROOT, train=False, seq_len=SEQ_LEN, num_digits=NUM_DIGITS,
                image_size=36, max_speed=MAX_SPEED, download=False, random=False,
                seed=SEED, **kw)
        return [w for w in caught if issubclass(w.category, UserWarning)]

    assert not build(motion_difficulty=0.5)
    assert not build(motion_mode="accelerate")            # no difficulty -> no override
    for kw in (dict(motion_difficulty=0.5, motion_mode="accelerate"),
               dict(motion_difficulty=0.5, transition_mode="uniform")):
        got = build(**kw)
        assert got, f"expected a UserWarning for {kw}"
        msg = str(got[0].message)
        assert "motion_mode" in msg and "transition_mode" in msg

    ds = _dataset(0.5)
    assert ds.motion_mode == "stochastic"
    assert ds.transition_mode == "smooth"


if __name__ == "__main__":
    rows = _all()
    print(format_table(rows))
    print()
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name} ok")
