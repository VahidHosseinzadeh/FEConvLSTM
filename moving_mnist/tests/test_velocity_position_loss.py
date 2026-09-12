"""
Contract tests for the position loss -- the properties it is FOR.

The loss exists because the pixel loss saturates: past ~3 px a correctly drawn
digit costs the same as a blank frame, so it cannot say how far off a rollout
is. Everything asserted here is one of the properties that fixes, so a failure
means the loss has stopped being usable for training a velocity model, not that
a tolerance drifted.

    cd moving_mnist && pytest tests/test_velocity_position_loss.py -v
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from velocity_position_loss import (PositionRolloutLoss, displacement_targets,  # noqa: E402
                                    huber_on_norm, toroidal_wrap)
from velocity_predictor_model import PhaseCorrelation  # noqa: E402


def velocities(b=32, h=10, seed=0):
    torch.manual_seed(seed)
    return torch.randint(-4, 5, (b, h, 2)).float()


# ---------------------------------------------------------------- the contract

def test_zero_exactly_when_the_digits_coincide():
    """The headline property: matched trajectories cost nothing at all."""
    v = velocities()
    for image_size in (None, 48):
        loss = PositionRolloutLoss(image_size=image_size)(v, v)
        assert loss.item() == 0.0, f"image_size={image_size}: {loss.item()}"


@pytest.mark.parametrize("offset", [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0])
def test_positive_whenever_they_do_not(offset):
    v = velocities()
    loss = PositionRolloutLoss(image_size=None)(v + torch.tensor([offset, 0.0]), v)
    assert loss.item() > 0.0


def test_strictly_monotone_in_the_separation_with_no_plateau():
    """
    The property the pixel loss does not have. Measured on this data, pixel MSE
    goes 0.039 -> 0.060 -> 0.073 between 3 and 20 px while a blank frame scores
    0.037: it is flat exactly where a rollout lives. This must not be.
    """
    v = velocities()
    loss_fn = PositionRolloutLoss(image_size=None)
    prev = -1.0
    for offset in [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        cur = loss_fn(v + torch.tensor([offset, 0.0]), v).item()
        assert cur > prev, f"not increasing at offset {offset}: {cur} <= {prev}"
        prev = cur


def test_displacement_is_the_cumulative_velocity_error():
    """A constant offset of o px/frame must put the digit exactly H*o px out.

    This is the identity the whole loss rests on -- the dataset renders motion
    with an integer roll, so position IS the running sum of velocity.
    """
    v = velocities(h=10)
    loss_fn = PositionRolloutLoss(image_size=None)
    for offset in [0.1, 0.5, 1.0, 2.0]:
        r = loss_fn.displacement_error(v + torch.tensor([offset, 0.0]), v)
        assert torch.allclose(r[:, -1], torch.full_like(r[:, -1], 10 * offset), atol=1e-4)


def test_gradient_reaches_the_velocities_and_charges_early_steps_more():
    """
    dL/dv_s collects a term from every step t >= s, so an early error is charged
    for the rest of the horizon. That is the compounding a per-step velocity
    loss misses, and it is why this loss is shaped for rollouts.
    """
    v = velocities()
    pred = (v + 1.0).clone().requires_grad_(True)
    PositionRolloutLoss(image_size=None)(pred, v).backward()
    g = pred.grad
    assert g is not None and g.abs().sum() > 0
    assert g[:, 0].abs().mean() > g[:, -1].abs().mean(), \
        "early rollout steps must carry more gradient than late ones"


# ---------------------------------------------------------------- the torus

def test_a_full_frame_of_drift_is_the_same_picture():
    """On a torus a displacement of exactly S px is not an error at all, and the
    loss has to agree with the pixels rather than with the arithmetic."""
    S, H = 48, 10
    v = velocities(h=H)
    off = torch.tensor([S / H, 0.0])
    wrapped = PositionRolloutLoss(image_size=S).displacement_error(v + off, v)
    plain = PositionRolloutLoss(image_size=None).displacement_error(v + off, v)
    assert wrapped[:, -1].abs().max() < 1e-3, wrapped[:, -1].abs().max()
    assert torch.allclose(plain[:, -1], torch.full_like(plain[:, -1], float(S)), atol=1e-3)


@pytest.mark.parametrize("size", [36, 48, 64])
def test_toroidal_wrap_stays_in_the_half_open_box(size):
    d = torch.linspace(-3 * size, 3 * size, 501).unsqueeze(-1).expand(-1, 2)
    w = toroidal_wrap(d, size)
    assert w.abs().max() <= size / 2 + 1e-4


# ---------------------------------------------------------------- details

def test_huber_is_smooth_at_zero_and_linear_far_out():
    delta = 2.0
    r = torch.tensor([0.0, 0.5, 1.0, 2.0, 10.0, 30.0])
    h = huber_on_norm(r, delta)
    assert h[0] == 0.0
    # quadratic below the knee, linear above it: the far-field slope is 1 px per px
    assert torch.allclose(h[-1] - h[-2], torch.tensor(20.0), atol=1e-4)


def test_final_weighting_only_scores_the_last_step():
    v = velocities(h=10)
    bad = v.clone()
    bad[:, 0] += 5.0            # a large error that is cancelled straight away
    bad[:, 1] -= 5.0
    loss = PositionRolloutLoss(image_size=None, weighting="final")(bad, v)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        PositionRolloutLoss()(velocities(h=10), velocities(h=9))


def test_targets_measured_from_frames_recover_the_true_shifts():
    """displacement_targets must read the motion off the pixels exactly, since
    that is what lets the loss be computed without ground-truth labels."""
    torch.manual_seed(0)
    S, T = 48, 8
    img = torch.zeros(1, 1, S, S)
    img[0, 0, 10:24, 12:28] = torch.rand(14, 16) * 0.6 + 0.4      # one sparse blob
    shifts = torch.tensor([[2., 1.], [-1., 3.], [0., -2.], [3., 3.],
                           [-2., 0.], [1., -1.], [4., 2.]])
    frames, cur = [img], img
    for vx, vy in shifts:
        cur = torch.roll(cur, shifts=(int(vy), int(vx)), dims=(2, 3))
        frames.append(cur)
    seq = torch.stack(frames, dim=1)                               # (1, T, 1, S, S)
    got = displacement_targets(seq, PhaseCorrelation(n_modes=1))
    assert torch.equal(got[0], shifts), f"{got[0].tolist()} != {shifts.tolist()}"
