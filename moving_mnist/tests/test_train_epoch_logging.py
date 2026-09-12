"""
train_epoch must report the training loss PER DECODER PROTOCOL.

Scheduled sampling flips a coin inside the model, so a single blended train_loss
mixes an oracle-tracked rollout with a deployable one. Those differ by orders of
magnitude, and reporting only the blend is what makes a steeply falling
train_loss sit next to a flat val_loss and read as overfitting when it is really
a change of protocol. wandb is stubbed here so the payload can be asserted on.

    cd moving_mnist && pytest tests/test_train_epoch_logging.py -v
"""

import os
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_eval_utils  # noqa: E402
import visualization  # noqa: E402
from train_eval_utils import train_epoch  # noqa: E402
from velocity_model_based_MEConvLSTM_model import Seq2SeqMEConvLSTM  # noqa: E402

T_IN, PRED, S = 5, 3, 16


class _StubWandb:
    """Captures log() payloads; everything else is a no-op the callers expect."""

    def __init__(self):
        self.payloads = []

    def log(self, d, *a, **k):
        self.payloads.append(d)

    def Image(self, *a, **k):
        return None

    def Table(self, *a, **k):
        return None

    def __getattr__(self, name):          # wandb.plot.line(...) etc.
        return lambda *a, **k: None


@pytest.fixture
def stub(monkeypatch):
    s = _StubWandb()
    monkeypatch.setattr(train_eval_utils, "wandb", s)
    monkeypatch.setattr(visualization, "wandb", s)
    return s


def _loader(n=4, b=2):
    blob = torch.zeros(1, S, S)
    blob[0, 3:8, 4:9] = 0.8
    seqs = torch.stack([torch.stack([torch.roll(blob, (t, 2 * t), dims=(1, 2))
                                     for t in range(T_IN + PRED)]) for _ in range(n)])
    return DataLoader(TensorDataset(seqs, torch.zeros(n)), batch_size=b)


def _model():
    torch.manual_seed(0)
    m = Seq2SeqMEConvLSTM(
        input_channels=1, hidden_channels=4, n_slots=1, decoder_layers=1,
        decoder_channels=4, use_velocity_dynamics=True, vel_dyn_state_dim=8,
        vel_dyn_openloop_k=PRED, vel_dyn_decoder_supervision="none",
        vel_dyn_loss="position")
    with torch.no_grad():
        m.vel_dyn.out.weight.normal_(0, 0.05)
        m.vel_dyn.out.bias.normal_(0, 0.05)
    return m


def _run(stub, sampling_p):
    m = _model()
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    train_epoch(m, _loader(), opt, torch.nn.MSELoss(), torch.device("cpu"), T_IN,
                grad_clip=1.0, vel_dyn_loss_weight=1.0, decoder_sampling_p=sampling_p)
    merged = {}
    for p in stub.payloads:
        merged.update(p)
    return merged


def test_oracle_protocol_is_labelled_as_such(stub):
    keys = _run(stub, 0.0)
    assert "train_loss_tracked" in keys, sorted(keys)
    assert keys["train_frac_tracked"] == pytest.approx(1.0)
    assert "train_loss_predicted" not in keys


def test_sampled_protocol_is_labelled_as_such(stub):
    keys = _run(stub, 1.0)
    assert "train_loss_predicted" in keys, sorted(keys)
    assert keys["train_frac_predicted"] == pytest.approx(1.0)
    assert "train_loss_tracked" not in keys


def test_the_blended_train_loss_is_still_reported(stub):
    """The headline train_loss stays, so existing runs remain comparable -- the
    per-protocol keys are additions, not a rename."""
    keys = _run(stub, 0.0)
    assert "train_shape_loss" in keys and "train_location_loss" in keys
