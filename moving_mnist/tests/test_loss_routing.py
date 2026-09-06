"""
--loss_routing split: the Fourier shape/location halves and where their
gradients are allowed to land.

    cd moving_mnist && pytest tests/test_loss_routing.py -v

The point of these is less "does the plumbing work" than pinning two facts
that decide whether the plumbing is worth using at all:

  * the split changes only gradient PATHS, never the objective's value;
  * under the current training defaults the velocity head is not on the pixel
    path, so the location term routes to a zero gradient. That is a property
    of the protocol, not of the routing code, and it is the thing to check
    before concluding that routing did or did not help.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fourier_loss import FourierShapePhaseLoss           # noqa: E402
from train import FourierPlusL1Loss                      # noqa: E402
from train_eval_utils import split_parameters            # noqa: E402
from velocity_model_based_MEConvLSTM_model import Seq2SeqMEConvLSTM  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _wandb_offline():
    """train_epoch logs figures; give it a run to log into, writing nowhere."""
    import wandb
    run = wandb.init(mode="disabled")
    yield
    run.finish()


def make_model(**kw):
    torch.manual_seed(0)
    base = dict(input_channels=1, hidden_channels=8, n_slots=1, decoder_layers=1,
                use_velocity_dynamics=True, vel_dyn_state_dim=16,
                vel_dyn_openloop_k=4, vel_dyn_decoder_supervision='none',
                track_corr_alpha=1.0)
    base.update(kw)
    return Seq2SeqMEConvLSTM(**base)


def make_batch(B=2, T_in=8, P=4, HW=24):
    torch.manual_seed(1)
    seq = torch.rand(B, T_in + P, 1, HW, HW)
    return seq[:, :T_in], seq[:, T_in:], P


def pixel_grad_norm(model, loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return float(sum((g ** 2).sum() for g in grads if g is not None) ** 0.5)


# ---------------------------------------------------------------- the split

@pytest.mark.parametrize("sw,lw", [(1.0, 1.0), (3.0, 1.0), (0.5, 2.0)])
def test_routed_terms_sum_to_forward(sw, lw):
    """Routing must not change WHAT is optimised, only where gradient goes."""
    crit = FourierPlusL1Loss(shape_weight=sw, location_weight=lw)
    torch.manual_seed(2)
    pred, tgt = torch.rand(2, 4, 1, 16, 16), torch.rand(2, 4, 1, 16, 16)

    enc, vel, _ = crit.routed_terms(pred, tgt)
    assert torch.allclose(enc + vel, crit(pred, tgt), atol=1e-6)


def test_the_users_two_formulations_are_the_same_thing():
    """'MSE minus shape' and 'explicitly built 2|X||Y|(1-cos dphi)' coincide.

    They are the same quantity by the identity in fourier_loss.py, so there is
    no choice to make between them -- only a numerical one, and decompose()
    takes the better branch (it never extracts an angle).
    """
    torch.manual_seed(3)
    pred, tgt = torch.rand(2, 3, 1, 16, 16), torch.rand(2, 3, 1, 16, 16)

    shape, location = FourierShapePhaseLoss.decompose(pred, tgt)
    mse = torch.nn.functional.mse_loss(pred, tgt)

    assert torch.allclose(shape + location, mse, atol=1e-5)
    assert torch.allclose(mse - shape, location, atol=1e-5)

    # ...and against the literal 2|X||Y|(1 - cos(phi_x - phi_y)) form, angles
    # and all, which is what one would write down from the algebra.
    X, Y = torch.fft.fft2(tgt.float()), torch.fft.fft2(pred.float())
    explicit = 2 * X.abs() * Y.abs() * (1 - torch.cos(X.angle() - Y.angle()))
    hw = X.shape[-2] * X.shape[-1]
    assert torch.allclose(explicit.sum() / (hw * pred.numel()), location, atol=1e-4)


def test_shape_is_zero_for_a_pure_translation():
    """The premise of the routing: displacement is ALL location, no shape."""
    torch.manual_seed(4)
    x = torch.rand(1, 1, 1, 16, 16)
    y = torch.roll(x, shifts=(3, -5), dims=(-2, -1))

    shape, location = FourierShapePhaseLoss.decompose(y, x)
    assert float(shape) < 1e-9
    assert float(location) > 1e-3


# ------------------------------------------------------- parameter partition

def test_split_parameters_partitions_everything_exactly_once():
    model = make_model()
    vel, enc = split_parameters(model)

    assert len(vel) > 0 and len(enc) > 0
    ids = {id(p) for p in vel} | {id(p) for p in enc}
    all_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert ids == all_ids
    assert len(vel) + len(enc) == len(all_ids)

    vel_names = {n for n, _ in model.named_parameters() if n.startswith("vel_dyn")}
    assert len(vel_names) == len(vel)
    # The correlators genuinely hold nothing -- the partition has no third case.
    assert not list(model.phase_corr_track.parameters())
    assert not list(model.phase_corr_bootstrap.parameters())


def test_no_velocity_params_when_head_is_off():
    vel, enc = split_parameters(make_model(use_velocity_dynamics=False))
    assert vel == [] and len(enc) > 0


# ------------------------------------------------- where the gradient lands

def test_pixel_loss_gives_the_head_zero_gradient_under_training_defaults():
    """The measurement that decides whether routing can matter.

    During training the decoder velocity is track_velocities(h, target), a
    phase-correlation argmax; the encoder's is the same or a bootstrap. With
    vel_dyn_gain='fixed' the head's output never reaches the warp, so the
    pixel loss cannot see it at all -- BOTH halves are exactly zero at the
    head, and routing the location half there routes nothing.
    """
    model = make_model()
    model.train()
    inp, tgt, P = make_batch()
    vel, enc = split_parameters(model)

    out = model(inp, pred_len=P, target_seq=tgt, decoder_sampling_p=0.0)
    shape, location = FourierShapePhaseLoss.decompose(out, tgt)

    assert pixel_grad_norm(model, shape, vel) == 0.0
    assert pixel_grad_norm(model, location, vel) == 0.0
    # ...while the renderer is very much in the graph.
    assert pixel_grad_norm(model, shape, enc) > 0.0


def test_scheduled_sampling_puts_the_head_on_the_pixel_path():
    """...and once it is there, the SHAPE term reaches it too.

    Which is the reason to route: the warp is bilinear, so changing u changes
    the interpolation blur and hence the magnitude spectrum. The head can
    lower the shape term by preferring near-integer velocities, correct or
    not. Splitting removes that gradient.
    """
    model = make_model()
    model.train()
    inp, tgt, P = make_batch()
    vel, _ = split_parameters(model)

    out = model(inp, pred_len=P, target_seq=tgt, decoder_sampling_p=1.0)
    shape, location = FourierShapePhaseLoss.decompose(out, tgt)

    assert pixel_grad_norm(model, shape, vel) > 0.0
    assert pixel_grad_norm(model, location, vel) > 0.0


def test_routed_backward_confines_each_half_to_its_own_group():
    """The routed gradients equal the restricted single-term gradients, and
    the encoder/decoder sees no location term at all."""
    model = make_model()
    model.train()
    inp, tgt, P = make_batch()
    vel, enc = split_parameters(model)
    crit = FourierPlusL1Loss(shape_weight=3.0, location_weight=1.0)

    out = model(inp, pred_len=P, target_seq=tgt, decoder_sampling_p=1.0)
    enc_term, vel_term, _ = crit.routed_terms(out, tgt)

    g_enc = torch.autograd.grad(enc_term, enc, retain_graph=True, allow_unused=True)
    g_vel = torch.autograd.grad(vel_term, vel, retain_graph=True, allow_unused=True)
    assert any(g is not None and g.abs().sum() > 0 for g in g_enc)
    assert any(g is not None and g.abs().sum() > 0 for g in g_vel)

    # Shared routing would additionally push the location term into enc/dec;
    # that difference is exactly what split removes, so the two must differ.
    g_enc_shared = torch.autograd.grad(enc_term + vel_term, enc,
                                       retain_graph=True, allow_unused=True)
    assert not all(torch.allclose(a, b)
                   for a, b in zip(g_enc, g_enc_shared)
                   if a is not None and b is not None)


def test_train_epoch_split_matches_manual_routing():
    """End-to-end through train_epoch: one step of --loss_routing split lands
    the same gradients as doing the two restricted backward passes by hand."""
    from torch.utils.data import DataLoader, TensorDataset
    from train_eval_utils import train_epoch

    inp, tgt, P = make_batch(B=2, T_in=8, P=4, HW=24)
    seq = torch.cat([inp, tgt], dim=1)
    loader = DataLoader(TensorDataset(seq, torch.zeros(seq.size(0))), batch_size=2)
    crit = FourierPlusL1Loss(shape_weight=3.0, location_weight=1.0)

    model = make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)   # lr=0: inspect grads only
    train_epoch(model, loader, opt, crit, torch.device("cpu"), input_frames=8,
                grad_clip=None, loss_routing='split', decoder_sampling_p=1.0)
    got = {n: (p.grad.clone() if p.grad is not None else None)
           for n, p in model.named_parameters()}

    ref = make_model()
    ref.train()
    vel, enc = split_parameters(ref)
    out = ref(inp, pred_len=P, target_seq=tgt, decoder_sampling_p=1.0)
    enc_term, vel_term, _ = crit.routed_terms(out, tgt)
    want = {}
    for params, term in ((enc, enc_term), (vel, vel_term)):
        grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        want.update({id(p): g for p, g in zip(params, grads)})
    for (n, p), (rn, rp) in zip(model.named_parameters(), ref.named_parameters()):
        w = want[id(rp)]
        if w is None:
            continue
        assert torch.allclose(got[n], w, atol=1e-5), n


def test_split_requires_a_decomposable_criterion_and_a_head():
    from torch.utils.data import DataLoader, TensorDataset
    from train import MSEPlusL1Loss
    from train_eval_utils import train_epoch

    inp, tgt, _ = make_batch(B=2, T_in=8, P=4, HW=24)
    seq = torch.cat([inp, tgt], dim=1)
    loader = DataLoader(TensorDataset(seq, torch.zeros(seq.size(0))), batch_size=2)
    dev = torch.device("cpu")

    model = make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    with pytest.raises(ValueError, match="--fourier_loss"):
        train_epoch(model, loader, opt, MSEPlusL1Loss(), dev, 8, loss_routing='split')

    headless = make_model(use_velocity_dynamics=False)
    opt2 = torch.optim.SGD(headless.parameters(), lr=0.0)
    with pytest.raises(ValueError, match="no velocity-model"):
        train_epoch(headless, loader, opt2, FourierPlusL1Loss(), dev, 8,
                    loss_routing='split')
