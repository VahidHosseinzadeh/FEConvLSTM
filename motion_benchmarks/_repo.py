"""
Make the flat-imported modules in `moving_mnist/` importable from this package.

The existing code base imports its modules flat (`from velocity_predictor_model import
PhaseCorrelation`) and is run from inside `moving_mnist/`. Rather than touch that, this package
puts `moving_mnist/` on sys.path once, here, and every module that needs the original models
imports this first.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MOVING_MNIST_DIR = REPO_ROOT / "moving_mnist"

for p in (str(REPO_ROOT), str(MOVING_MNIST_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
