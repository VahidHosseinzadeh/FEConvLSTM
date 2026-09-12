"""
Which velocity drives the DECODER, and whether the head is on the pixel path.

This is the question that decides whether the velocity head learns to predict
motion at all. During training the decoder velocity is
v = track(h, target_seq[:, t]) -- an oracle re-measured against the true future
frame at every step -- unless scheduled sampling fires. So with
--decoder_sampling_p 0 the head NEVER drives the rollout while training, gets no
gradient from the image loss, and the ConvLSTM never learns to cope with a
predicted velocity. These tests pin that, and pin that raising the flag fixes it.

    cd moving_mnist && pytest tests/test_decoder_protocol.py -v
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from velocity_model_based_MEConvLSTM_model import Seq2SeqMEConvLSTM  # noqa: E402

T_IN, PRED, S = 6, 3, 16


def model(**kw):
    torch.manual_seed(0)
    m = Seq2SeqMEConvLSTM(
        input_channels=1, hidden_channels=4, n_slots=1, decoder_layers=1,
        decoder_channels=4, use_velocity_dynamics=True, vel_dyn_state_dim=8,
        vel_dyn_openloop_k=PRED, vel_dyn_decoder_supervision="none", **kw)
    # The head is zero-initialised, so its output is exactly the frozen velocity
    # and its gradient path is degenerate. Give it a real function first.
    with torch.no_grad():
        m.vel_dyn.out.weight.normal_(0, 0.05)
        m.vel_dyn.out.bias.normal_(0, 0.05)
    return m


def batch(b=2, seed=0):
    torch.manual_seed(seed)
    seq = torch.zeros(b, T_IN + PRED, 1, S, S)
    blob = torch.zeros(1, S, S)
    blob[0, 3:8, 4:9] = torch.rand(5, 5) * 0.6 + 0.4
    for i in range(b):
        for t in range(T_IN + PRED):
            seq[i, t] = torch.roll(blob, shifts=(t, 2 * t), dims=(1, 2))
    return seq[:, :T_IN], seq[:, T_IN:]


# ------------------------------------------------- which protocol actually ran

def test_training_without_sampling_is_the_oracle():
    """The default. The head is a bystander in the rollout."""
    m = model().train()
    inp, tgt = batch()
    m(inp, PRED, target_seq=tgt, track_decoder_velocity=True, decoder_sampling_p=0.0)
    assert m.last_decoder_protocol == "tracked"


def test_sampling_puts_the_head_in_charge():
    m = model().train()
    inp, tgt = batch()
    m(inp, PRED, target_seq=tgt, track_decoder_velocity=True, decoder_sampling_p=1.0)
    assert m.last_decoder_protocol == "predicted"


def test_eval_defaults_to_frozen():
    m = model().eval()
    inp, _ = batch()
    m(inp, PRED, target_seq=None, track_decoder_velocity=False)
    assert m.last_decoder_protocol == "frozen"


def test_eval_can_ask_for_the_head():
    m = model().eval()
    inp, _ = batch()
    m(inp, PRED, target_seq=None, track_decoder_velocity=False,
      predict_decoder_velocity=True)
    assert m.last_decoder_protocol == "predicted"


# ------------------------------------------------- is the head on the pixel path

def _head_grad_from_pixel_loss(sampling_p):
    """|grad| reaching the velocity head from the IMAGE loss alone."""
    m = model().train()
    inp, tgt = batch()
    out = m(inp, PRED, target_seq=tgt, track_decoder_velocity=True,
            decoder_sampling_p=sampling_p)
    ((out - tgt) ** 2).mean().backward()
    return sum(p.grad.abs().sum().item() for p in m.vel_dyn.parameters()
               if p.grad is not None)


def test_pixel_loss_does_not_reach_the_head_without_sampling():
    """With the oracle protocol every velocity fed to the warp is a phase-
    correlation argmax, which carries no gradient -- so the image loss gives the
    head EXACTLY zero. This is the thing that has to be fixed, pinned so that a
    future change cannot quietly un-fix it."""
    assert _head_grad_from_pixel_loss(0.0) == 0.0


def test_pixel_loss_reaches_the_head_once_sampling_fires():
    """With the head driving the rollout the warp is differentiable in its
    output, so the image loss finally trains it -- the whole point of raising
    --decoder_sampling_p."""
    g = _head_grad_from_pixel_loss(1.0)
    assert g > 0.0, "image loss still does not reach the velocity head"


def test_sampling_only_applies_while_training():
    """Evaluation must stay a fixed, honest protocol regardless of the flag, or
    val numbers would become random across epochs."""
    m = model().eval()
    inp, tgt = batch()
    m(inp, PRED, target_seq=tgt, track_decoder_velocity=True, decoder_sampling_p=1.0)
    assert m.last_decoder_protocol == "tracked"
