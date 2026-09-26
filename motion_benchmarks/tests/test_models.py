"""
Model variants: bit-identical to the originals by default, correct velocity sources, and the
paper's property -- exact motion equivariance under time-dependent integer translations.
"""
import copy
import math

import numpy as np
import pytest
import torch

from motion_benchmarks.common.shifts import roll_torch
from motion_benchmarks.models import (FEConvLSTMPlus, MEConvLSTMPlus, MeanFlowConnection,
                                      OracleStabilized, Persistence, bias_for_tau)
from motion_benchmarks.models.cf_classifier import MotionVideoClassifier
from channel_based_FEConvLSTM_model import Seq2SeqFEConvLSTM
from motion_classification_model import MotionDigitClassifier
from velocity_model_based_MEConvLSTM_model import Seq2SeqMEConvLSTM


def moving_seq(B=2, T=8, C=1, S=24, seed=0, integer=True):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(B, C, S, S, generator=g)
    m = torch.randint(-2, 3, (B, T, 1, 2), generator=g).float()
    if not integer:
        m = m + 0.5
    D = torch.cat([torch.zeros(B, 1, 2), torch.cumsum(m[:, :, 0], 1)[:, :-1]], 1)
    seq = torch.stack([roll_torch(base, D[:, t]) for t in range(T)], 1)
    return seq, m


def test_melstm_plus_default_is_parent():
    x, _ = moving_seq(T=7)
    torch.manual_seed(3)
    a = Seq2SeqMEConvLSTM(1, 8, n_slots=2)
    torch.manual_seed(3)
    b = MEConvLSTMPlus(1, 8, n_slots=2)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
    inp, tgt = x[:, :4], x[:, 4:]
    for track in (True, False):
        ya, va = a(inp, 3, target_seq=tgt, track_decoder_velocity=track, return_velocity=True)
        yb, vb = b(inp, 3, target_seq=tgt, track_decoder_velocity=track, return_velocity=True)
        assert torch.equal(ya, yb) and torch.equal(va, vb)


def test_felstm_plus_default_is_parent():
    x, _ = moving_seq(T=6)
    torch.manual_seed(1)
    a = Seq2SeqFEConvLSTM(1, 6, v_range=1)
    torch.manual_seed(1)
    b = FEConvLSTMPlus(1, 6, v_range=1)
    assert torch.equal(a(x[:, :4], 2), b(x[:, :4], 2))


def test_video_classifier_default_is_digit_classifier():
    x, _ = moving_seq(T=5, S=32)
    for model in ("lstm", "melstm"):
        torch.manual_seed(2)
        a = MotionDigitClassifier(model=model, hidden_channels=6, n_slots=2, head_channels=8)
        torch.manual_seed(2)
        b = MotionVideoClassifier(model=model, hidden_channels=6, n_slots=2, head_channels=8)
        a.eval(), b.eval()
        assert torch.equal(a(x), b(x))


def test_video_classifier_trajectory_readout_runs():
    x, _ = moving_seq(T=6, S=32)
    m = MotionVideoClassifier(model="melstm", hidden_channels=6, n_slots=3, head_channels=8,
                              readout_steps=3, forget_bias=0.44, forget_bias_long=2.97,
                              phase_corr_kwargs=dict(subpixel=True, suppress_radius=1))
    m.eval()
    out, aux = m(x, return_aux=True)
    assert out.shape == (2, 10) and aux["velocities"].shape == (2, 5, 3, 2)


def test_forget_bias():
    assert bias_for_tau(2) == pytest.approx(0.44, abs=0.01)
    assert bias_for_tau(20) == pytest.approx(2.97, abs=0.01)
    m = MEConvLSTMPlus(1, 8, forget_bias=0.44, forget_bias_long=2.97, long_fraction=0.25)
    f = m.cell.conv.bias[8:16]
    assert torch.allclose(f[:2], torch.tensor(2.97)) and torch.allclose(f[2:], torch.tensor(0.44))


def test_external_velocities_are_used():
    x, m = moving_seq(T=7)
    model = MEConvLSTMPlus(1, 6, n_slots=1, velocity_source="external")
    _, v = model(x[:, :4], 3, target_seq=x[:, 4:], track_decoder_velocity=True,
                 return_velocity=True, velocities=m)
    assert torch.equal(v[:, :, 0], m[:, :6, 0])          # encoder t uses m[t-1], decoder too


def test_frame_pair_velocities_match_truth():
    x, m = moving_seq(T=7, S=32)
    model = MEConvLSTMPlus(1, 6, n_slots=1, velocity_source="frame_pair")
    _, v = model(x[:, :5], 2, target_seq=x[:, 5:], track_decoder_velocity=True,
                 return_velocity=True)
    assert torch.equal(v[:, :, 0], m[:, :6, 0])


def _transformed(seed=4, T=9, S=24):
    x, m = moving_seq(B=2, T=T, S=S, seed=seed)
    g = torch.Generator().manual_seed(9)
    u = torch.randint(-2, 3, (2, T, 1, 2), generator=g).float()        # extra frame motion
    D = torch.cat([torch.zeros(2, 1, 2), torch.cumsum(u[:, :, 0], 1)[:, :-1]], 1)
    xt = torch.stack([roll_torch(x[:, t], D[:, t]) for t in range(T)], 1)
    return x, m, xt, m + u, D


def test_encoder_is_exactly_motion_equivariant():
    x, m, xt, mt, D = _transformed()
    torch.manual_seed(0)
    me = MEConvLSTMPlus(1, 6, n_slots=1, velocity_source="external")
    h = me.encode(x[:, :5], velocities=m)[0]
    ht = me.encode(xt[:, :5], velocities=mt)[0]
    assert torch.allclose(ht[:, 0], roll_torch(h[:, 0], D[:, 4]), atol=1e-5)


@pytest.mark.parametrize("src", ["external", "frame_pair"])
def test_rollout_equivariance_needs_aligned_decoder_input(src):
    """psi > X with a time-dependent integer translation. With decoder_input='warped' the whole
    rollout moves with it (Theorem 2). With the original 'previous' input the decoder combines
    the UNWARPED previous frame with the warped state -- misaligned by v, and by v + u after the
    transformation -- so the rollout is not exactly equivariant. ConvLSTM is not either way."""
    x, m, xt, mt, D = _transformed()
    kw = lambda mm: dict(velocities=mm) if src == "external" else {}   # noqa: E731
    errs = {}
    for dec in ("warped", "previous", "zeros"):
        torch.manual_seed(0)
        me = MEConvLSTMPlus(1, 6, n_slots=1, velocity_source=src, decoder_input=dec)
        y = me(x[:, :5], 4, target_seq=x[:, 5:], track_decoder_velocity=True, **kw(m))
        yt = me(xt[:, :5], 4, target_seq=xt[:, 5:], track_decoder_velocity=True, **kw(mt))
        back = torch.stack([roll_torch(y[:, t], D[:, 5 + t]) for t in range(4)], 1)
        errs[dec] = float((yt - back).abs().max())
    assert errs["warped"] < 1e-4 and errs["zeros"] < 1e-4
    assert errs["previous"] > 1e-3
    torch.manual_seed(0)
    lstm = FEConvLSTMPlus(1, 6, v_range=0)
    y, yt = lstm(x[:, :5], 4), lstm(xt[:, :5], 4)
    back = torch.stack([roll_torch(y[:, t], D[:, 5 + t]) for t in range(4)], 1)
    assert float((yt - back).abs().max()) > 1e-3


def test_fe_warped_decoder_runs_and_lstm_unchanged():
    x, _ = moving_seq(T=6)
    torch.manual_seed(1)
    a = FEConvLSTMPlus(1, 6, v_range=0)
    torch.manual_seed(1)
    b = FEConvLSTMPlus(1, 6, v_range=0, decoder_input="warped")   # v = (0, 0): same thing
    assert torch.allclose(a(x[:, :4], 2), b(x[:, :4], 2), atol=1e-6)
    c = FEConvLSTMPlus(1, 6, v_range=1, decoder_input="warped", residual="eulerian")
    assert c(x[:, :4], 2).shape == (2, 2, 1, 24, 24)


def test_lagrangian_residual_is_persistence_with_zero_decoder():
    x, m = moving_seq(T=7, S=32)
    model = MEConvLSTMPlus(1, 6, n_slots=1, velocity_source="external", residual="lagrangian",
                           residual_shift="bilinear")
    for p in model.decoder.parameters():
        torch.nn.init.zeros_(p)
    y = model(x[:, :4], 3, target_seq=x[:, 4:], track_decoder_velocity=True, velocities=m)
    assert torch.allclose(y, x[:, 4:], atol=1e-5)


def test_mean_flow_connection_and_gauge():
    B, C, S = 2, 3, 16
    mean = torch.tensor([0.5, 0.0, 0.0])
    std = torch.tensor([0.2, 0.3, 0.3])
    conn = MeanFlowConnection([1], [2], mean, std, px_per_unit=(2.0, 4.0))
    u_phys = torch.tensor([[0.25, -0.5], [1.0, 0.1]])
    x = torch.randn(B, C, S, S)
    x[:, 1] = x[:, 1] - x[:, 1].mean((-2, -1), keepdim=True) + u_phys[:, 0, None, None] / 0.3
    x[:, 2] = x[:, 2] - x[:, 2].mean((-2, -1), keepdim=True) + u_phys[:, 1, None, None] / 0.3
    assert torch.allclose(conn.physical_mean(x), u_phys, atol=1e-5)
    assert torch.allclose(conn.velocity(x), u_phys * torch.tensor([2.0, 4.0]), atol=1e-5)
    g = conn.gauge_fix(x)
    assert torch.allclose(conn.physical_mean(g), torch.zeros(B, 2), atol=1e-5)
    assert torch.allclose(conn.add_mean(g, u_phys), x, atol=1e-5)
    model = MEConvLSTMPlus(C, 6, n_slots=1, velocity_source="mean_flow", mean_flow=conn)
    seq = x[:, None].expand(B, 5, C, S, S).contiguous()
    _, v = model(seq[:, :3], 2, target_seq=seq[:, 3:], return_velocity=True)
    assert torch.allclose(v[:, 0, 0], u_phys * torch.tensor([2.0, 4.0]), atol=1e-4)


def test_oracle_stabilized_eulerian_is_lagrangian_oracle():
    x, m = moving_seq(T=7, S=24)
    stab = OracleStabilized(Persistence("eulerian"), shift="bilinear")
    y1 = stab(x[:, :4], 3, m)
    y2 = Persistence("lagrangian", "oracle", shift="bilinear")(x[:, :4], 3, m)
    assert torch.allclose(y1, y2, atol=1e-5)
    assert torch.allclose(y1, x[:, 4:], atol=1e-5)


def test_track_gate_and_window_run():
    x, m = moving_seq(T=6, S=32)
    model = MEConvLSTMPlus(1, 6, n_slots=2, velocity_source="track", pc_window=True,
                           track_gate=dict(radius=3, min_conf=2.0, erode_quantile=0.8),
                           phase_corr_kwargs=dict(subpixel=True, suppress_radius=1))
    y, v = model(x[:, :4], 2, target_seq=x[:, 4:], return_velocity=True)
    assert y.shape == (2, 2, 1, 32, 32) and torch.isfinite(v).all()
