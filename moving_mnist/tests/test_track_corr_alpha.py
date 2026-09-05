"""
Contract tests for the tracking correlator's whitening exponent.

Two properties, and both are load-bearing:

  * the DEFAULT must reproduce classic phase correlation bit-for-bit, so every
    run and checkpoint predating this option is unaffected;
  * alpha must not move the peak's LOCATION, only how reliably it is found --
    otherwise it would be changing the velocity estimate itself, and the
    model's motion equivariance rests on that estimate transforming correctly.

    cd moving_mnist && pytest tests/test_track_corr_alpha.py -v
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from velocity_predictor_model import PhaseCorrelation                # noqa: E402
from velocity_model_based_MEConvLSTM_model import Seq2SeqMEConvLSTM  # noqa: E402

H = W = 32


def blob(seed=0, n=3):
    torch.manual_seed(seed)
    x = torch.zeros(n, 1, H, W)
    x[:, 0, 8:16, 6:18] = torch.rand(n, 8, 12) + 0.4
    return x


def test_default_is_bitwise_classic_phase_correlation():
    """alpha defaults to 1.0 and takes the original divide, not pow(.,1)."""
    a, b = blob(1), torch.roll(blob(1), shifts=(3, -2), dims=(2, 3))
    assert PhaseCorrelation(n_modes=1).alpha == 1.0
    v_new, s_new = PhaseCorrelation(n_modes=1)(a, b)

    # recompute the pre-change expression by hand
    F1, F2 = torch.fft.rfft2(a.mean(1)), torch.fft.rfft2(b.mean(1))
    R = F1 * torch.conj(F2)
    R = R / (R.abs() + 1e-8)
    corr = torch.fft.irfft2(R, s=(H, W)).reshape(len(a), -1)
    s_ref, idx = torch.topk(corr, 1, dim=1)
    y, x = (idx // W).float(), (idx % W).float()
    x = torch.where(x > W / 2, x - W, x)
    y = torch.where(y > H / 2, y - H, y)
    v_ref = torch.stack((-x, -y), dim=-1)

    assert torch.equal(v_new, v_ref)
    assert torch.equal(s_new, s_ref)


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 1.0])
def test_alpha_never_moves_the_peak_for_a_rigid_shift(alpha):
    """For an exact rigid shift the displacement is recovered at every alpha.

    This is the property the equivariance argument needs: |R| is invariant to a
    shift of either input, so dividing by any function of it cannot move where
    the peak sits. alpha trades noise robustness, never correctness.
    """
    pc = PhaseCorrelation(n_modes=1, alpha=alpha)
    for dx, dy in ((1, 0), (0, -1), (4, 3), (-5, 6)):
        a = blob(2)
        b = torch.roll(a, shifts=(dy, dx), dims=(2, 3))
        v, _ = pc(a, b)
        assert v[0, 0].tolist() == [dx, dy], f"alpha={alpha} at ({dx},{dy}) -> {v[0,0].tolist()}"


def test_alpha_zero_survives_a_smooth_template():
    """The whole point. A smooth template kills alpha=1 and not alpha=0.

    Phase correlation divides each frequency by its own magnitude, so the bands
    a blurred template barely occupies are amplified to unit gain -- noise, at
    the same weight as signal. This is what track(h, X_t) hits, because
    h.mean(dim=2) is smooth.
    """
    import torch.nn.functional as F
    sharp = blob(3, n=24)
    shifted = torch.roll(sharp, shifts=(2, 3), dims=(2, 3))
    smooth = torch.tanh(3 * F.avg_pool2d(F.pad(sharp, (2,) * 4, mode="circular"), 5, stride=1))

    def hit(pc, tmpl):
        v, _ = pc(tmpl, shifted)
        return (v[:, 0] == torch.tensor([3.0, 2.0])).all(-1).float().mean().item()

    pc1, pc0 = PhaseCorrelation(n_modes=1, alpha=1.0), PhaseCorrelation(n_modes=1, alpha=0.0)
    assert hit(pc1, sharp) == 1.0, "alpha=1 must be exact on a sharp template"
    assert hit(pc0, sharp) == 1.0, "alpha=0 must also be exact on a sharp template"
    assert hit(pc0, smooth) > hit(pc1, smooth), (
        f"alpha=0 ({hit(pc0, smooth):.2f}) should beat alpha=1 "
        f"({hit(pc1, smooth):.2f}) on a smooth template")


def test_model_default_leaves_both_correlators_identical():
    m = Seq2SeqMEConvLSTM(input_channels=1, hidden_channels=8, n_slots=2, decoder_layers=1)
    assert m.track_corr_alpha is None
    assert m.phase_corr_track.alpha == m.phase_corr_bootstrap.alpha == 1.0


def test_model_decouples_only_the_tracking_correlator():
    m = Seq2SeqMEConvLSTM(input_channels=1, hidden_channels=8, n_slots=2, decoder_layers=1,
                          track_corr_alpha=0.0)
    assert m.phase_corr_track.alpha == 0.0
    assert m.phase_corr_bootstrap.alpha == 1.0, "the bootstrap must not be touched"


def test_forward_is_unchanged_at_the_default():
    """Bitwise identical outputs with the option absent vs explicitly at 1.0."""
    torch.manual_seed(5)
    x = torch.rand(2, 5, 1, H, W)
    def build(**kw):
        torch.manual_seed(9)
        return Seq2SeqMEConvLSTM(input_channels=1, hidden_channels=8, n_slots=2,
                                 decoder_layers=1, **kw).eval()
    with torch.no_grad():
        a = build()(x, pred_len=3, target_seq=None, track_decoder_velocity=False)
        b = build(track_corr_alpha=1.0)(x, pred_len=3, target_seq=None,
                                        track_decoder_velocity=False)
    assert torch.equal(a, b)
