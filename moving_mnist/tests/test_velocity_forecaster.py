"""
Contract tests for the velocity forecasters.

Two things are pinned here because the rest of the design depends on them:

  * the `equivariant=True` variants are EXACTLY motion-equivariant, and the
    others are not -- that flag has to mean something, in both directions;
  * `GRUForecaster` at initialisation reproduces the frozen rollout bit for bit,
    so turning it on cannot regress a baseline (the same nesting guarantee
    VelocityDynamicsHead has).

    cd moving_mnist && pytest tests/test_velocity_forecaster.py -v
"""

import copy
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from velocity_forecaster import (HarmonicForecaster, LTIForecaster,  # noqa: E402
                                 GRUForecaster, frozen_rollout, momentum_rollout)

CTX, V = 15, 4.0


def context(b=8, t=CTX, seed=0):
    torch.manual_seed(seed)
    return torch.randint(-4, 5, (b, t, 2)).float()


def models(v_max=V, equivariant=False):
    return [("harmonic", HarmonicForecaster(v_max=v_max, equivariant=equivariant)),
            ("lti", LTIForecaster(v_max=v_max, equivariant=equivariant)),
            ("gru", GRUForecaster(v_max=v_max, equivariant=equivariant))]


@pytest.mark.parametrize("horizon", [1, 10, 30])
def test_shapes(horizon):
    v = context()
    for name, m in models() + models(equivariant=True):
        out = m(v, horizon)
        assert out.shape == (v.shape[0], horizon, 2), f"{name}: {tuple(out.shape)}"


@pytest.mark.parametrize("name_model", models(v_max=None, equivariant=True),
                         ids=lambda nm: nm[0])
def test_equivariant_variants_are_exactly_equivariant(name_model):
    """g(u + c) == g(u) + c for every constant c.

    This is the property the whole MEConvLSTM construction rests on; a model
    that claims it and does not have it voids Theorem 2 rather than degrading
    it. v_max is off because a clamp is not equivariant once it binds.

    The tolerance is precision-aware, and deliberately so. The models that
    ITERATE add an invariant increment to an equivariant base once per step, and
    (v + c) + d does not round to (v + d) + c, so float32 leaves a few ulp per
    step. Measured, to show that is all it is:

        model      float32              float64
        lti        1.53e-05 (16 ulp)    2.84e-14
        gru        0.00e+00 ( 0 ulp)    0.00e+00   (zero-init => frozen rollout)
        harmonic   4.77e-07 (0.5 ulp)   8.88e-16   (closed form, never iterates)

    Nine orders of magnitude of improvement from float32 to float64 is round-off
    scaling with precision, not a broken construction -- so the property is
    pinned in float64, where it is unambiguous, and float32 only has to stay
    within a sane bound.
    """
    name, m = name_model
    v = context()
    c = torch.tensor([2.0, -3.0])
    with torch.no_grad():
        dev32 = (m(v + c, 10) - m(v, 10) - c).abs().max().item()
    assert dev32 < 1e-4, f"{name}: float32 max |g(u+c) - g(u) - c| = {dev32:.2e}"

    m64 = copy.deepcopy(m).double()
    v64, c64 = v.double(), c.double()
    with torch.no_grad():
        dev64 = (m64(v64 + c64, 10) - m64(v64, 10) - c64).abs().max().item()
    assert dev64 < 1e-10, f"{name}: float64 max |g(u+c) - g(u) - c| = {dev64:.2e}"


def test_the_flag_actually_changes_something():
    """The non-equivariant harmonic head must NOT be equivariant -- otherwise
    the comparison between the two is vacuous.

    Only the harmonic head is checked: GRUForecaster has a zero-initialised
    readout, so at init it emits the frozen rollout, which IS equivariant. That
    is tested separately below rather than treated as a failure here.
    """
    torch.manual_seed(0)
    m = HarmonicForecaster(v_max=None, equivariant=False)
    v = context()
    c = torch.tensor([2.0, -3.0])
    with torch.no_grad():
        dev = (m(v + c, 10) - m(v, 10) - c).abs().max().item()
    assert dev > 1e-3, f"expected non-equivariance, got {dev:.2e}"


def test_gru_at_init_is_exactly_the_frozen_rollout():
    """Zero-initialised readout => delta == 0 => u_next == u_t, which IS the
    frozen-velocity rollout. Nesting the baseline exactly means an untrained
    head cannot make anything worse."""
    for equivariant in (False, True):
        torch.manual_seed(0)
        m = GRUForecaster(v_max=None, equivariant=equivariant)
        v = context()
        with torch.no_grad():
            assert torch.equal(m(v, 12), frozen_rollout(v, 12))


def test_harmonic_forward_matches_its_own_parameters():
    """forward() must be the closed form evaluated at t = 1..H, since the point
    of this head is that a step is computed FROM PARAMETERS rather than from its
    own previous output -- that is what makes H=30 as accurate as H=10."""
    torch.manual_seed(0)
    m = HarmonicForecaster(v_max=None)
    v = context()
    H = 20
    with torch.no_grad():
        w, a_c, a_s, d = m.parameters_for(v)
        t = torch.arange(1, H + 1, dtype=v.dtype)
        wt = w.unsqueeze(1) * t.view(1, -1, 1)
        expect = (d.unsqueeze(1) + a_c.unsqueeze(1) * torch.cos(wt)
                  + a_s.unsqueeze(1) * torch.sin(wt))
        assert torch.allclose(m(v, H), expect, atol=1e-6)


def test_harmonic_frequency_stays_in_its_band():
    """An unconstrained omega makes the loss surface violently multimodal, so
    the band is a real part of the design, not a cosmetic clamp."""
    torch.manual_seed(0)
    m = HarmonicForecaster(v_max=None, period_range=(8.0, 40.0))
    for seed in range(4):
        w, _, _, _ = m.parameters_for(context(seed=seed))
        assert (w >= m.w_min - 1e-6).all() and (w <= m.w_max + 1e-6).all()


@pytest.mark.parametrize("name_model", models(v_max=2.0), ids=lambda nm: nm[0])
def test_v_max_is_respected(name_model):
    name, m = name_model
    torch.manual_seed(0)
    with torch.no_grad():
        out = m(context() * 2, 25)
    assert out.abs().max() <= 2.0 + 1e-6, f"{name}: {out.abs().max().item()}"


def test_baselines_are_what_they_claim():
    v = context()
    fr = frozen_rollout(v, 7)
    assert torch.equal(fr, v[:, -1:].expand(-1, 7, -1))
    mo = momentum_rollout(v, 3)
    step = v[:, -1] - v[:, -2]
    assert torch.allclose(mo[:, 0], v[:, -1] + step)
    assert torch.allclose(mo[:, 2], v[:, -1] + 3 * step)


def test_gradients_reach_every_model():
    v = context()
    for name, m in models() + models(equivariant=True):
        out = m(v, 10)
        out.square().mean().backward()
        total = sum(p.grad.abs().sum() for p in m.parameters() if p.grad is not None)
        assert total > 0, f"{name}: no gradient reached the parameters"
