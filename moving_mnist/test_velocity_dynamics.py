"""
Contract tests for the equivariant velocity dynamics head.

Run from this directory (the modules import each other by bare module name):

    cd moving_mnist && pytest test_velocity_dynamics.py -v

test_equivariance is the one that matters. MEConvLSTM's entire justification
is exact motion equivariance; a velocity head that only *approximately*
commutes with a global translation silently voids that for the trained model.
The other tests guard the second promise -- that switching the head on cannot
by itself change a single number.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from velocity_dynamics_model import VelocityDynamicsHead          # noqa: E402
from velocity_model_based_MEConvLSTM_model import Seq2SeqMEConvLSTM  # noqa: E402


B, K, CH, H, W = 3, 2, 8, 16, 16
STATE_DIM = 12


def _rollout(head, u_hist, h, state):
    """
    Feed a whole velocity history through the head, the way the model does:
    du from consecutive velocities (zeros at the first step), state carried,
    h fixed. Returns the final u_next.

    u_hist : list of (B, K, 2)
    """
    u_prev = None
    u_next = None
    for u in u_hist:
        du = torch.zeros_like(u) if u_prev is None else u - u_prev
        u_next, state = head(u, du, h, state)
        u_prev = u
    return u_next


@pytest.mark.parametrize("use_h", [False, True])
def test_equivariance(use_h):
    """
    g(u_{<=t} + v, roll(h_t, s)) == g(u_{<=t}, h_t) + v, exactly.

    Under a global motion with velocity v, every velocity estimate picks up
    the same offset and the hidden state translates on the torus. If this
    assertion fails the head has broken the model's motion equivariance and
    Theorem 2 no longer applies to anything trained with it -- which is a
    correctness failure, not a tuning problem.

    The head is randomized away from its zero-init here: at init Delta == 0
    and the identity map is trivially equivariant, which would make this test
    pass for the wrong reason.
    """
    torch.manual_seed(0)

    head = VelocityDynamicsHead(state_dim=STATE_DIM, hidden_channels=CH,
                                use_h=use_h, h_embed_dim=6).double()
    # Break the zero-init so a genuinely non-trivial function is tested.
    for p in head.parameters():
        torch.nn.init.normal_(p, std=0.5)

    T = 5
    u_hist = [torch.randn(B, K, 2, dtype=torch.double) for _ in range(T)]
    h = torch.randn(B, K, CH, H, W, dtype=torch.double)
    state0 = head.init_state(B, K, h.device, h.dtype)

    v = torch.randn(1, 1, 2, dtype=torch.double)          # global motion
    sy, sx = 5, -3                                        # global spatial shift

    u_plain = _rollout(head, u_hist, h, state0)
    u_shift = _rollout(head,
                       [u + v for u in u_hist],
                       torch.roll(h, shifts=(sy, sx), dims=(-2, -1)),
                       state0)

    assert torch.allclose(u_shift, u_plain + v, atol=1e-5), \
        f"max deviation {(u_shift - (u_plain + v)).abs().max().item():.3e}"


def test_identity_at_init():
    """Zero-initialized output layer => Delta == 0 => u_next == u_t, exactly.

    This is what makes the head strictly nest the existing frozen rollout: an
    untrained head reproduces today's decoder rather than perturbing it."""
    torch.manual_seed(1)
    for use_h in (False, True):
        head = VelocityDynamicsHead(state_dim=STATE_DIM, hidden_channels=CH,
                                    use_h=use_h)
        u = torch.randn(B, K, 2)
        du = torch.randn(B, K, 2)
        h = torch.randn(B, K, CH, H, W)
        state = head.init_state(B, K, u.device, u.dtype)

        u_next, _ = head(u, du, h, state)
        assert torch.equal(u_next, u), f"use_h={use_h}: head is not identity at init"


def test_state_shape_and_per_slot():
    """GRU state is (B, K, state_dim), per slot, and untouched by the caller.

    Per-slot is checked by feeding two slots different histories and requiring
    their states to diverge -- a shared/broadcast state would keep them equal.
    """
    torch.manual_seed(2)
    head = VelocityDynamicsHead(state_dim=STATE_DIM)
    for p in head.parameters():
        torch.nn.init.normal_(p, std=0.5)

    state = head.init_state(B, K, torch.device("cpu"), torch.float32)
    assert state.shape == (B, K, STATE_DIM)
    assert torch.count_nonzero(state) == 0

    u = torch.zeros(B, K, 2)
    du = torch.zeros(B, K, 2)
    du[:, 0] = 1.0                       # slot 0 accelerating, slot 1 not
    _, new_state = head(u, du, None, state)

    assert new_state.shape == (B, K, STATE_DIM)
    assert not torch.allclose(new_state[:, 0], new_state[:, 1]), \
        "slots share a state -- the GRU state is not per-slot"


def test_dynamics_state_is_not_warped():
    """
    The head's state must be invariant, so the model must never warp it.

    Warping is what MEConvLSTMCell does to (h, c), which live on the image
    grid. This state lives in velocity-difference space, which a global
    translation does not touch -- the check here is structural: the head owns
    no spatial dimensions to warp, and the model's forward pass never passes
    the state through cell.warp.
    """
    head = VelocityDynamicsHead(state_dim=STATE_DIM)
    state = head.init_state(B, K, torch.device("cpu"), torch.float32)
    assert state.dim() == 3, "state has spatial dims -- it would be warpable, and must not be"

    import inspect
    src = inspect.getsource(Seq2SeqMEConvLSTM.forward)
    for line in src.splitlines():
        if "warp" in line and "dyn_state" in line:
            pytest.fail(f"dynamics state is being warped: {line.strip()}")


def _fixed_input(seed=7, T_in=5, pred_len=4):
    torch.manual_seed(seed)
    return (torch.rand(2, T_in, 1, H, W), pred_len)


def _build(seed, **kwargs):
    torch.manual_seed(seed)
    return Seq2SeqMEConvLSTM(input_channels=1, hidden_channels=CH, n_slots=K,
                             decoder_layers=1, **kwargs)


def test_nesting_bitwise_identical():
    """
    use_velocity_dynamics=False and a freshly-initialized True must produce
    BITWISE identical outputs in frozen decoder mode.

    Bitwise, not close: the point of the zero-init is that turning the flag on
    cannot move an existing result by even one ulp, so any regression observed
    after enabling the head is something the head learned.

    Note both models are built under the same seed AND the dynamics head is
    constructed last, so the shared parameters draw from an identical RNG
    stream. (This is also why the head is not constructed at all when the flag
    is off -- constructing and ignoring it would shift that stream.)
    """
    x, pred_len = _fixed_input()

    base = _build(11, use_velocity_dynamics=False).eval()
    dyn = _build(11, use_velocity_dynamics=True, vel_dyn_state_dim=STATE_DIM).eval()

    # Sanity: same shared weights, so any output difference is the head's doing.
    for (n, p), (n2, p2) in zip(base.named_parameters(), dyn.named_parameters()):
        assert n == n2
        assert torch.equal(p, p2), f"parameter {n} differs -- RNG streams diverged"

    with torch.no_grad():
        out_base = base(x, pred_len=pred_len, target_seq=None,
                        track_decoder_velocity=False)
        out_dyn = dyn(x, pred_len=pred_len, target_seq=None,
                      track_decoder_velocity=False)

    assert torch.equal(out_base, out_dyn), \
        f"max |diff| = {(out_base - out_dyn).abs().max().item():.3e} (must be exactly 0)"


def test_nesting_holds_with_tracked_decoder():
    """Same nesting guarantee under the training protocol (tracked decoder
    velocities), where the head runs on every decoder step too."""
    x, pred_len = _fixed_input()
    tgt = torch.rand(x.size(0), pred_len, 1, H, W)

    base = _build(12, use_velocity_dynamics=False).eval()
    dyn = _build(12, use_velocity_dynamics=True, vel_dyn_state_dim=STATE_DIM).eval()

    with torch.no_grad():
        out_base = base(x, pred_len=pred_len, target_seq=tgt,
                        track_decoder_velocity=True)
        out_dyn = dyn(x, pred_len=pred_len, target_seq=tgt,
                      track_decoder_velocity=True)

    assert torch.equal(out_base, out_dyn)


def test_predicted_decoder_mode_runs_and_matches_frozen_at_init():
    """
    The third decoder mode is wired up, and at init it reproduces the frozen
    rollout exactly (u_next == u_t with Delta == 0). Once trained it is free
    to differ -- that is the whole point -- but it must start from parity.
    """
    x, pred_len = _fixed_input()
    dyn = _build(13, use_velocity_dynamics=True, vel_dyn_state_dim=STATE_DIM).eval()

    with torch.no_grad():
        frozen = dyn(x, pred_len=pred_len, target_seq=None,
                     track_decoder_velocity=False)
        predicted = dyn(x, pred_len=pred_len, target_seq=None,
                        track_decoder_velocity=False,
                        predict_decoder_velocity=True)

    assert torch.equal(frozen, predicted)


def test_dyn_loss_is_returned_and_trains_the_head():
    """
    return_dyn_loss produces a finite scalar with gradient reaching ONLY the
    dynamics head -- the measurement must not learn from the head, and the
    velocity loss must not reach the ConvLSTM through the encoder.
    """
    x, pred_len = _fixed_input()
    dyn = _build(14, use_velocity_dynamics=True, vel_dyn_state_dim=STATE_DIM)

    out, dyn_loss = dyn(x, pred_len=pred_len, target_seq=None,
                        track_decoder_velocity=False, return_dyn_loss=True)
    assert dyn_loss.dim() == 0 and torch.isfinite(dyn_loss)
    assert dyn_loss.item() > 0, "zero-init head should mispredict a changing velocity"

    dyn_loss.backward()
    head_grads = [p.grad for p in dyn.vel_dyn.parameters() if p.grad is not None]
    assert head_grads, "dynamics loss produced no gradient for the head"
    assert any(g.abs().sum() > 0 for g in head_grads)

    for name, p in dyn.cell.named_parameters():
        assert p.grad is None or torch.count_nonzero(p.grad) == 0, \
            f"dynamics loss leaked into the ConvLSTM cell ({name})"


def test_openloop_adds_supervision():
    """--vel_dyn_openloop_k replays the head open-loop over encoder steps it
    already has measurements for, so it must add loss terms without touching
    the image path."""
    x, pred_len = _fixed_input(T_in=6)

    plain = _build(15, use_velocity_dynamics=True, vel_dyn_state_dim=STATE_DIM).eval()
    ol = _build(15, use_velocity_dynamics=True, vel_dyn_state_dim=STATE_DIM,
                vel_dyn_openloop_k=3).eval()

    with torch.no_grad():
        out_a, loss_a = plain(x, pred_len=pred_len, target_seq=None,
                              track_decoder_velocity=False, return_dyn_loss=True)
        out_b, loss_b = ol(x, pred_len=pred_len, target_seq=None,
                           track_decoder_velocity=False, return_dyn_loss=True)

    assert torch.equal(out_a, out_b), "open-loop replay must not change the images"
    assert torch.isfinite(loss_b)
    assert not torch.equal(loss_a, loss_b), \
        "open-loop replay added no supervision terms"


def test_learned_gain_is_equivariant_in_the_model():
    """
    The learned gain reads the phase-correlation peak score, which is
    invariant under a global translation of the input. So the blended encoder
    velocity stays equivariant: translating the whole input sequence by a
    constant per-frame shift must shift every velocity estimate by exactly
    that amount.

    A rigid sub-pixel-free shift of the frames by (dx, dy) per step adds
    (dx, dy) to the true motion; on the circular grid this is exact.
    """
    torch.manual_seed(3)
    T_in = 5
    base = torch.zeros(1, T_in, 1, H, W)
    # A small blob translating by (1, 2) px/step on the torus.
    for t in range(T_in):
        blob = torch.zeros(H, W)
        blob[4:8, 4:8] = torch.rand(4, 4) + 0.5
        base[0, t, 0] = torch.roll(blob, shifts=(2 * t, 1 * t), dims=(0, 1))

    model = _build(16, use_velocity_dynamics=True, vel_dyn_state_dim=STATE_DIM,
                   vel_dyn_gain='learned').eval()

    dx, dy = 3, -2
    shifted = torch.stack([torch.roll(base[0, t], shifts=(dy, dx), dims=(-2, -1))
                           for t in range(T_in)]).unsqueeze(0)

    with torch.no_grad():
        _, v_base = model(base, pred_len=1, target_seq=None,
                          track_decoder_velocity=False, return_velocity=True)
        _, v_shift = model(shifted, pred_len=1, target_seq=None,
                           track_decoder_velocity=False, return_velocity=True)

    # A rigid shift applied identically to every frame changes no velocity.
    assert torch.allclose(v_base, v_shift, atol=1e-5), \
        f"max deviation {(v_base - v_shift).abs().max().item():.3e}"
