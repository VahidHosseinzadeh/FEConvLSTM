"""
DataLoader worker seeding for the ORIGINAL Moving MNIST datasets (re-exported from
moving_mnist/worker_seeding.py, where train.py / train_classification.py use it under the opt-in
--distinct_worker_streams flag).

The datasets in motion_benchmarks/ do not need it: a fixed item is a pure function of
(seed, index) (datasets/base.py), reproducible for ANY num_workers and batch order.
"""
from .. import _repo  # noqa: F401
from worker_seeding import worker_init_fn  # noqa: E402,F401
