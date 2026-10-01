"""Shifts, schedules, metrics, baselines, worker seeding."""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from motion_benchmarks.common.baselines import eulerian_persistence, lagrangian_persistence
from motion_benchmarks.common.metrics import (FSS, Categorical, PerLeadError, ke_spectrum,
                                              nusselt_volume, radial_spectrum, velocity_epe)
from motion_benchmarks.common.schedules import apply_freeze, make_schedule
from motion_benchmarks.common.seeding import worker_init_fn
from motion_benchmarks.common.shifts import (bilinear_shift_np, bilinear_shift_torch,
                                             cumulative_displacement, fourier_shift_np,
                                             fourier_shift_torch, roll_torch)


@pytest.mark.parametrize("d", [(3, -2), (0, 5), (-7, 1)])
def test_integer_shifts_equal_roll(noise_field, d):
    ref = np.roll(noise_field, (d[1], d[0]), (0, 1))
    assert np.abs(fourier_shift_np(noise_field, d) - ref).max() < 1e-5
    assert np.abs(bilinear_shift_np(noise_field, d) - ref).max() < 1e-6
    t = torch.tensor(noise_field)[None, None]
    dt = torch.tensor([d], dtype=torch.float32)
    assert torch.allclose(fourier_shift_torch(t, dt)[0, 0], torch.tensor(ref), atol=1e-4)
    assert torch.allclose(bilinear_shift_torch(t, dt)[0, 0], torch.tensor(ref), atol=1e-5)
    assert torch.equal(roll_torch(t, dt)[0, 0], torch.tensor(ref))


def test_torch_numpy_agree_subpixel(noise_field):
    d = (1.37, -0.62)
    t = torch.tensor(noise_field)[None, None]
    dt = torch.tensor([d])
    assert np.abs(fourier_shift_torch(t, dt)[0, 0].numpy() - fourier_shift_np(noise_field, d)).max() < 1e-4
    assert np.abs(bilinear_shift_torch(t, dt)[0, 0].numpy() - bilinear_shift_np(noise_field, d)).max() < 1e-4


def test_shift_composition(noise_field):
    # exact up to the Nyquist bins, which a band-limited field barely populates
    a = fourier_shift_np(fourier_shift_np(noise_field, (0.4, 1.3)), (1.1, -0.8))
    b = fourier_shift_np(noise_field, (1.5, 0.5))
    assert np.abs(a - b).max() < 5e-3


def test_matches_moving_mnist_warp_convention():
    """The package's bilinear shift == MEConvLSTMCell.warp (the model's transport)."""
    from velocity_model_based_MEConvLSTM_model import MEConvLSTMCell
    cell = MEConvLSTMCell(1, 4)
    cell.integer_shift = False                 # sub-pixel shifts need the exact (padded) warp
    x = torch.randn(3, 1, 4, 20, 24)
    u = torch.tensor([[[1.3, -0.4]], [[-2.0, 1.0]], [[0.25, 0.75]]])
    w = cell.warp(x, u)[:, 0]
    s = bilinear_shift_torch(x[:, 0], u[:, 0])
    assert torch.allclose(w, s, atol=1e-5)
    # Moving MNIST's whole-pixel fast path agrees on whole-pixel shifts
    cell.integer_shift = True
    ui = torch.tensor([[[1.0, -3.0]], [[-2.0, 1.0]], [[0.0, 5.0]]])
    assert torch.allclose(cell.warp(x, ui)[:, 0], bilinear_shift_torch(x[:, 0], ui[:, 0]), atol=1e-5)


def test_motion_benchmark_models_use_the_exact_warp():
    """Sub-pixel velocities: every MEConvLSTM built here must take the padded warp."""
    from motion_benchmarks.models.melstm_plus import MEConvLSTMPlus
    assert MEConvLSTMPlus(1, 4, n_slots=2).cell.integer_shift is False


def test_cumulative_displacement_convention():
    m = np.array([[1, 0], [2, 1], [0, -1]], float)
    D = cumulative_displacement(m)
    assert D.tolist() == [[0, 0], [1, 0], [3, 1]]
    assert torch.equal(cumulative_displacement(torch.tensor(m)), torch.tensor(D))


def test_apply_freeze_matches_td_moving_mnist():
    m = np.arange(20, dtype=float).reshape(10, 2)
    f = apply_freeze(m, 6)
    assert (f[:5] == m[:5]).all() and (f[5:] == m[4]).all()
    assert apply_freeze(m, None) is m


@pytest.mark.parametrize("kind", ["piecewise", "ou", "rotating", "constant"])
def test_schedules_respect_speed(kind, rng):
    v = make_schedule(kind, 200, rng, 2.5, hold=(3, 6))
    assert v.shape == (200, 2)
    assert np.hypot(v[:, 0], v[:, 1]).max() <= 2.5 + 1e-9


def test_piecewise_integer_never_zero(rng):
    v = make_schedule("piecewise", 300, rng, 2, integer=True, hold=(1, 2))
    assert (np.abs(v).sum(1) > 0).all() and np.allclose(v, np.round(v))


def test_per_lead_error_and_categorical():
    tgt = torch.zeros(2, 3, 1, 8, 8)
    tgt[:, :, :, 2:4, 2:4] = 1.0
    pred = tgt.clone()
    pred[:, 2] = 0.0                                    # miss everything at lead 3
    e = PerLeadError()
    e.update(pred, tgt)
    r = e.result()
    assert r["mse"][0] == 0 and r["mse"][2] == pytest.approx(4 / 64)
    c = Categorical([0.5])
    c.update(pred, tgt)
    rc = c.result()
    assert rc["csi@0.5"][:2] == [1.0, 1.0] and rc["csi@0.5"][2] == 0.0
    f = FSS([0.5], scales=[1, 3])
    f.update(pred, tgt)
    rf = f.result()
    assert rf["fss@0.5_n1"][0] == 1.0 and rf["fss@0.5_n3"][2] == 0.0


def test_spectrum_and_nusselt():
    x = torch.zeros(16, 16)
    x[:, :] = torch.cos(2 * np.pi * 3 * torch.arange(16) / 16)[None, :]
    s = radial_spectrum(x)
    assert int(torch.argmax(s)) == 3
    ke = ke_spectrum(x[None], x[None])
    assert ke.shape[-1] == 9
    T = torch.ones(1, 4, 5, 5)
    w = torch.zeros(1, 4, 5, 5)
    assert float(nusselt_volume(T, w, kappa=0.1)) == 1.0
    assert velocity_epe(torch.tensor([[3.0, 4.0]]), torch.zeros(1, 2)) == 5.0


def test_persistence_baselines(noise_field):
    x = torch.tensor(noise_field)
    v = torch.tensor([[1.5, -0.5]])
    seq = torch.stack([fourier_shift_torch(x[None], v * t)[0] for t in range(6)])[None, :, None]
    inp, tgt = seq[:, :4], seq[:, 4:]
    lag = lagrangian_persistence(inp, 2, alpha=1.0)
    eul = eulerian_persistence(inp, 2)
    assert ((lag - tgt) ** 2).mean() < 0.05 * ((eul - tgt) ** 2).mean()
    orc = lagrangian_persistence(inp, 2, velocity=v[:, None].expand(1, 2, 2))
    assert ((orc - tgt) ** 2).mean() < 1e-4          # Nyquist-bin residue only


class _StatefulFixedDataset(Dataset):
    """The Moving MNIST pattern: private RNG, index ignored, random=False."""

    def __init__(self):
        self.seed, self.random = 5, False
        self.rng = np.random.RandomState(self.seed)

    def __len__(self):
        return 16

    def __getitem__(self, i):
        return torch.tensor(self.rng.randint(0, 10 ** 9))


def test_worker_seeding_removes_duplication():
    ds = _StatefulFixedDataset()
    dup = torch.cat(list(DataLoader(ds, batch_size=4, num_workers=2)))
    ds.rng = np.random.RandomState(ds.seed)
    fix = torch.cat(list(DataLoader(ds, batch_size=4, num_workers=2, worker_init_fn=worker_init_fn)))
    ds.rng = np.random.RandomState(ds.seed)
    fix2 = torch.cat(list(DataLoader(ds, batch_size=4, num_workers=2, worker_init_fn=worker_init_fn)))
    assert len(set(dup.tolist())) == 8                  # every value appears twice
    assert len(set(fix.tolist())) == 16                 # all distinct
    assert torch.equal(fix, fix2)                       # and reproducible
