"""Shared fixtures. Everything runs on CPU; the whole suite takes a few minutes."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "moving_mnist")):
    if p not in sys.path:
        sys.path.insert(0, p)

torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def band_limited(H, W, corr, rng):
    z = rng.standard_normal((H, W))
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.fftfreq(W)[None, :]
    z = np.real(np.fft.ifft2(np.fft.fft2(z) * np.exp(-2 * (np.pi * corr) ** 2 * (ky ** 2 + kx ** 2))))
    return ((z - z.mean()) / z.std()).astype(np.float32)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def noise_field():
    return band_limited(48, 48, 1.0, np.random.default_rng(7))
