"""
Contract tests for --vel_dyn_loss {velocity, position, both}.

The one that matters most is the first: 'velocity' is the DEFAULT and must be
the pre-existing objective untouched, or every previous run silently stops being
comparable. The rest pin what the new modes actually compute.

    cd moving_mnist && pytest tests/test_vel_dyn_loss_modes.py -v
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from velocity_model_based_MEConvLSTM_model import Seq2SeqMEConvLSTM  # noqa: E402
from velocity_position_loss import huber_on_norm  # noqa: E402

T_IN, PRED, S = 6, 3, 16


def model(loss="velocity", **kw):
    torch.manual_seed(0)
    return Seq2SeqMEConvLSTM(
        input_channels=1, hidden_channels=4, n_slots=1, decoder_layers=1,
        decoder_channels=4, use_velocity_dynamics=True, vel_dyn_state_dim=8,
        vel_dyn_openloop_k=PRED, vel_dyn_decoder_supervision="none",
        vel_dyn_loss=loss, **kw)


def batch(b=2, seed=0):
    torch.manual_seed(seed)
    seq = torch.zeros(b, T_IN + PRED, 1, S, S)
    for i in range(b):                       # one moving blob per sequence
        blob = torch.zeros(1, S, S)
        blob[0, 3:8, 4:9] = torch.rand(5, 5) * 0.6 + 0.4
        for t in range(T_IN + PRED):
            seq[i, t] = torch.roll(blob, shifts=(t, 2 * t), dims=(1, 2))
    return seq[:, :T_IN], seq[:, T_IN:]


# ---------------------------------------------------------------- the default

def test_velocity_is_the_default():
    """Anything else would silently change every existing run."""
    assert model().vel_dyn_loss == "velocity"


def test_velocity_mode_is_plain_smooth_l1():
    m = model("velocity")
    a = torch.randn(2, 1, 2)
    b = torch.randn(2, 1, 2)
    term, _ = m._dyn_term(a, b)
    assert torch.allclose(term, F.smooth_l1_loss(a, b))


# ---------------------------------------------------------------- the new modes

@pytest.mark.parametrize("mode", ["velocity", "position", "both"])
def test_zero_when_the_prediction_matches(mode):
    """Every mode must agree on the one point that matters."""
    m = model(mode)
    x = torch.randn(2, 1, 2)
    term, _ = m._dyn_term(x, x)
    assert term.item() == pytest.approx(0.0, abs=1e-7)


def test_position_mode_is_in_pixels_and_accumulates():
    """A constant per-step error e must be charged as k*|e| px on step k --
    that accumulation is the whole reason the position form exists."""
    m = model("position")
    e = torch.tensor([[[0.5, 0.0]]])          # 0.5 px/frame, one slot
    zero = torch.zeros_like(e)
    cum = None
    for k in (1, 2, 3, 4):
        term, cum = m._dyn_term(e, zero, cum)
        expected = huber_on_norm(torch.tensor(0.5 * k), m.vel_dyn_pos_delta)
        assert torch.allclose(term, expected, atol=1e-6), f"step {k}"


def test_both_is_exactly_the_sum_of_the_two():
    mv, mp, mb = model("velocity"), model("position"), model("both")
    a, b = torch.randn(2, 1, 2), torch.randn(2, 1, 2)
    tv, _ = mv._dyn_term(a, b)
    tp, _ = mp._dyn_term(a, b)
    tb, _ = mb._dyn_term(a, b)
    assert torch.allclose(tb, tv + mb.vel_dyn_pos_weight * tp, atol=1e-6)


def test_bad_mode_is_rejected():
    with pytest.raises(ValueError):
        model("pixels")


# ---------------------------------------------------------------- end to end

@pytest.mark.parametrize("mode", ["velocity", "position", "both"])
def test_forward_produces_a_finite_loss_and_trains_the_head(mode):
    m = model(mode)
    inp, tgt = batch()
    out, dyn = m(inp, PRED, target_seq=tgt, track_decoder_velocity=True,
                 return_dyn_loss=True)
    assert torch.isfinite(dyn) and dyn.item() > 0, f"{mode}: {dyn}"
    dyn.backward()
    g = sum(p.grad.abs().sum() for p in m.vel_dyn.parameters() if p.grad is not None)
    assert g > 0, f"{mode}: no gradient reached the velocity head"


def test_the_flag_changes_the_objective():
    """If position and velocity gave the same number the switch would be a no-op."""
    inp, tgt = batch()
    vals = {}
    for mode in ("velocity", "position"):
        m = model(mode)
        _, dyn = m(inp, PRED, target_seq=tgt, track_decoder_velocity=True,
                   return_dyn_loss=True)
        vals[mode] = dyn.item()
    assert vals["velocity"] != pytest.approx(vals["position"], rel=1e-3), vals


def test_the_image_path_is_untouched_by_the_choice():
    """The head's objective must not change the predicted FRAMES -- with a fixed
    gain the head is not on the pixel path at all, and that has to stay true."""
    inp, tgt = batch()
    outs = []
    for mode in ("velocity", "position", "both"):
        m = model(mode)
        out, _ = m(inp, PRED, target_seq=tgt, track_decoder_velocity=True,
                   return_dyn_loss=True)
        outs.append(out)
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2])
