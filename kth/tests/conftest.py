"""Shared fixtures. CPU only; the KTH-data tests skip when data/kth is missing."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "moving_mnist")):
    if p not in sys.path:
        sys.path.insert(0, p)

torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

DATA = ROOT / "data" / "kth"


@pytest.fixture(scope="session")
def store():
    if not (DATA / "kth_frames.h5").exists():
        pytest.skip("data/kth/kth_frames.h5 not found (copy Keller's cache from the cluster)")
    from kth.kth_dataset import KTHStore
    return KTHStore(DATA)
