"""
Contract tests for the shape / location decomposition of the pixel loss.

The identity being relied on is exact, so these are equalities, not tolerances
chosen to make things pass:

    |X - Y|^2 == (|X| - |Y|)^2 + 2|X||Y|(1 - cos dphi)

    cd moving_mnist && pytest tests/test_fourier_loss.py -v
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fourier_loss import FourierShapePhaseLoss  # noqa: E402


def digits(b=2, t=3, h=32, w=32, seed=0):
    """Sparse bright-on-black blocks -- the regime the loss is designed for."""
    torch.manual_seed(seed)
    x = torch.zeros(b, t, 1, h, w)
    for i in range(b):
        for j in range(t):
            r, c = torch.randint(4, h - 14, (2,))
            x[i, j, 0, r:r + 10, c:c + 8] = torch.rand(10, 8) * 0.6 + 0.4
    return x


def test_the_two_terms_sum_to_mse():
    """Parseval, and the algebra, and the normalisation -- all at once."""
    for seed in range(4):
        x, y = digits(seed=seed), digits(seed=seed + 100)
        s, l = FourierShapePhaseLoss.decompose(y, x)
        assert torch.allclose(s + l, F.mse_loss(y, x), rtol=1e-5, atol=1e-7), \
            f"{(s + l).item():.6f} vs {F.mse_loss(y, x).item():.6f}"


def test_default_weights_are_exactly_mse():
    x, y = digits(), digits(seed=7)
    loss, parts = FourierShapePhaseLoss()(y, x)
    assert torch.allclose(loss, F.mse_loss(y, x), rtol=1e-5, atol=1e-7)
    assert torch.allclose(parts["mse"], F.mse_loss(y, x), rtol=1e-5, atol=1e-7)


def test_both_terms_are_non_negative():
    for seed in range(4):
        s, l = FourierShapePhaseLoss.decompose(digits(seed=seed), digits(seed=seed + 50))
        assert s >= 0 and l >= 0


@pytest.mark.parametrize("shift", [(1, 0), (0, 5), (7, -3), (16, 16)])
def test_pure_translation_has_zero_shape_error(shift):
    """
    The claim that makes the split meaningful. A cyclic shift multiplies every
    Fourier coefficient by a unit-modulus phase, so magnitudes are untouched
    and the shape term must vanish -- for ANY displacement, however large.
    """
    x = digits()
    y = torch.roll(x, shifts=shift, dims=(-2, -1))
    s, l = FourierShapePhaseLoss.decompose(y, x)
    assert s < 1e-9, f"shift {shift} leaked {s.item():.3e} into the shape term"
    assert torch.allclose(l, F.mse_loss(y, x), rtol=1e-5, atol=1e-7), \
        "a pure translation must put ALL of the error in the location term"


def test_blur_and_blankness_land_in_the_shape_term():
    """
    The failure mode this loss is meant to price. A blank prediction has zero
    magnitude everywhere, so its entire error is 'shape' -- it cannot be
    excused by having the location approximately right.
    """
    x = digits()
    blank = torch.zeros_like(x)
    s_blank, l_blank = FourierShapePhaseLoss.decompose(blank, x)
    assert l_blank < 1e-9
    assert torch.allclose(s_blank, F.mse_loss(blank, x), rtol=1e-5, atol=1e-7)

    blurred = F.avg_pool2d(F.pad(x.flatten(0, 1), (2,) * 4, mode="circular"),
                           5, stride=1).view_as(x)
    s_blur, _ = FourierShapePhaseLoss.decompose(blurred, x)
    assert s_blur > 0.2 * F.mse_loss(blurred, x), \
        "blur must show up substantially in the shape term"


def test_weights_do_what_they_say():
    x, y = digits(), digits(seed=3)
    s, l = FourierShapePhaseLoss.decompose(y, x)
    loss, _ = FourierShapePhaseLoss(shape_weight=3.0, location_weight=0.5)(y, x)
    assert torch.allclose(loss, 3.0 * s + 0.5 * l, rtol=1e-5, atol=1e-7)


def test_is_differentiable_and_gradient_is_finite():
    x = digits()
    y = digits(seed=11).requires_grad_(True)
    loss, _ = FourierShapePhaseLoss(shape_weight=2.0, location_weight=1.0)(y, x)
    loss.backward()
    assert y.grad is not None and torch.isfinite(y.grad).all()
    assert y.grad.abs().sum() > 0
