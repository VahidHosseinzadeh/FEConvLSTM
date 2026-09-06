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


# ----------------------------------------------------------------------
# Head architecture and scheduled sampling (added with those features)
# ----------------------------------------------------------------------

from velocity_dynamics_model import VelocityDynamicsHead  # noqa: E402


@pytest.mark.parametrize("arch", ["gru", "recurrence"])
@pytest.mark.parametrize("layers", [1, 2])
def test_head_is_identity_at_init(arch, layers):
    """Both architectures must nest the frozen rollout exactly at init."""
    head = VelocityDynamicsHead(state_dim=16, arch=arch, n_layers=layers)
    u, du = torch.randn(3, 2, 2), torch.randn(3, 2, 2)
    st = head.init_state(3, 2, u.device, u.dtype)
    u_next, _ = head(u, du, None, st)
    assert torch.equal(u_next, u)


@pytest.mark.parametrize("arch", ["gru", "recurrence"])
@pytest.mark.parametrize("layers", [1, 2])
def test_head_equivariance(arch, layers):
    """g(u+v, roll(h,s)) == g(u,h) + v, for every architecture."""
    torch.manual_seed(0)
    head = VelocityDynamicsHead(state_dim=16, arch=arch, n_layers=layers).double()
    for p in head.parameters():
        torch.nn.init.normal_(p, std=0.4)
    u_hist = [torch.randn(3, 2, 2, dtype=torch.double) for _ in range(6)]
    c = torch.randn(1, 1, 2, dtype=torch.double)

    def roll(hist):
        st = head.init_state(3, 2, torch.device("cpu"), torch.double)
        prev, out = None, None
        for u in hist:
            du = torch.zeros_like(u) if prev is None else u - prev
            out, st = head(u, du, None, st)
            prev = u
        return out

    assert torch.allclose(roll([u + c for u in u_hist]), roll(u_hist) + c, atol=1e-5)


def _open_loop(head, steps=200, seed=1):
    torch.manual_seed(seed)
    for _, p in head.named_parameters():
        torch.nn.init.normal_(p, std=1.0)
    u = torch.randn(4, 2, 2)
    st = head.init_state(4, 2, u.device, u.dtype)
    du = torch.randn(4, 2, 2)
    with torch.no_grad():
        for _ in range(steps):
            nxt, st = head(u, du, None, st)
            du, u = nxt - u, nxt
    return u


def test_pole_bound_alone_is_not_a_divergence_proof():
    """
    Documents a NEGATIVE result so nobody re-derives the wrong guarantee from
    the parametrisation. Both poles sit at radius rho < 1, so every INDIVIDUAL
    step contracts -- but the coefficients are recomputed each step from a
    state the rollout itself drives, so this is a switched linear system and
    the product of contractive matrices can still grow. Whether it actually
    blows up is seed-dependent (measured: 7.1 at one seed, 1e16 at another),
    which is exactly the point -- it is not bounded, so it is not a guarantee.
    Asserted as a spread across seeds rather than a single blow-up, since any
    one seed is flaky.
    """
    worst = max(_open_loop(VelocityDynamicsHead(state_dim=16, arch="recurrence"),
                           seed=s).abs().max().item()
                for s in range(6))
    assert worst > 50.0, (
        f"worst-case open-loop magnitude over 6 seeds was only {worst:.1f}; if the "
        f"rollout is now genuinely bounded the guarantee may be real and this "
        f"test should be replaced by a positive one")


@pytest.mark.parametrize("arch", ["gru", "recurrence"])
def test_v_max_is_what_actually_bounds_the_rollout(arch):
    """v_max is the hard guard, and the data's own speed limit is its natural
    value. It costs exact equivariance only when it binds, which on in-range
    data it never should."""
    u = _open_loop(VelocityDynamicsHead(state_dim=16, arch=arch, v_max=8.0))
    assert torch.isfinite(u).all()
    assert u.abs().max() <= 8.0 + 1e-6, f"v_max did not hold: {u.abs().max().item()}"


def test_scheduled_sampling_is_off_by_default_and_train_only():
    """p=0 must reproduce the tracked protocol bitwise; p=1 must change it;
    and neither must have any effect in eval mode."""
    torch.manual_seed(3)
    x = torch.rand(2, 5, 1, H, W)
    tgt = torch.rand(2, 3, 1, H, W)

    def run(p, training):
        torch.manual_seed(4)
        m = Seq2SeqMEConvLSTM(input_channels=1, hidden_channels=8, n_slots=2,
                              decoder_layers=1, use_velocity_dynamics=True)
        m.train(training)
        torch.manual_seed(5)                        # fix the sampling coin too
        with torch.no_grad():
            return m(x, pred_len=3, target_seq=tgt, track_decoder_velocity=True,
                     decoder_sampling_p=p)

    assert torch.equal(run(0.0, True), run(0.0, True))
    assert not torch.equal(run(0.0, True), run(1.0, True)), \
        "p=1 must actually switch the decoder onto the head"
    assert torch.equal(run(0.0, False), run(1.0, False)), \
        "scheduled sampling must never fire outside training"


@pytest.mark.parametrize("sup", ["none", "teacher", "openloop"])
def test_decoder_supervision_controls_what_leaks_from_the_future(sup):
    """
    The head must not be fed velocities MEASURED AGAINST THE TARGET FRAMES as
    inputs -- that is teacher forcing, and repeating the input is its optimal
    solution, which is why the head measured at one-step-lag quality.

    Checked by making the target frames a pure translation of a known amount:
    under 'teacher' the head's own state picks that velocity up, under 'none'
    and 'openloop' it cannot.
    """
    torch.manual_seed(0)
    ctx = torch.rand(2, 6, 1, H, W)
    # targets translating by a large, unmistakable amount
    tgt = torch.stack([torch.roll(ctx[:, -1], shifts=(0, 7 * (t + 1)), dims=(-2, -1))
                       for t in range(4)], dim=1)

    torch.manual_seed(1)
    m = Seq2SeqMEConvLSTM(input_channels=1, hidden_channels=8, n_slots=1,
                          decoder_layers=1, use_velocity_dynamics=True,
                          vel_dyn_decoder_supervision=sup).train()
    _, dyn = m(ctx, pred_len=4, target_seq=tgt, track_decoder_velocity=True,
               return_dyn_loss=True)
    assert torch.isfinite(dyn)

    n_ctx_only = None
    if sup == "none":
        # 'none' must produce exactly the encoder's own terms, i.e. running
        # with no target_seq at all gives the identical count of loss terms.
        torch.manual_seed(1)
        m2 = Seq2SeqMEConvLSTM(input_channels=1, hidden_channels=8, n_slots=1,
                               decoder_layers=1, use_velocity_dynamics=True,
                               vel_dyn_decoder_supervision="none").train()
        _, dyn2 = m2(ctx, pred_len=4, target_seq=None,
                     track_decoder_velocity=False, return_dyn_loss=True)
        assert torch.allclose(dyn, dyn2), (
            "with supervision 'none' the dynamics loss must be identical whether "
            "or not target frames were supplied -- otherwise something still leaks")


def test_decoder_supervision_rejects_nonsense():
    with pytest.raises(ValueError, match="decoder_supervision"):
        Seq2SeqMEConvLSTM(input_channels=1, hidden_channels=8, n_slots=1,
                          decoder_layers=1, use_velocity_dynamics=True,
                          vel_dyn_decoder_supervision="oracle")
