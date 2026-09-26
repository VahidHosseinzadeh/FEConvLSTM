"""
Distinct, reproducible RNG streams for DataLoader workers (opt-in: --distinct_worker_streams).

The trap: a dataset that keeps a PRIVATE, stateful RandomState and ignores its index --
TDMovingMNISTDataset / CommonFateMovingMNISTDataset with random=False, i.e. every FIXED benchmark
set -- is copied into each DataLoader worker in the SAME state, so worker k regenerates worker 0's
stream. For TDMovingMNISTDataset (digits, positions and motions all come from that RNG) a
"10,000-sequence" test set built with num_workers=4 therefore holds ~2,500 distinct sequences,
each seen four times: the mean is unbiased, the error bars are too small by ~2x. (For
CommonFateMovingMNISTDataset with a digit pool the glyph follows the index, so items are not
duplicated, but textures, positions and motions are shared across workers.)

Training sets built with random=True draw from numpy's GLOBAL RNG, which PyTorch >= 1.9 already
reseeds per worker, so they are not affected.

worker_init_fn gives every worker its own stream, still a deterministic function of
(dataset seed, worker id), so a fixed benchmark stays reproducible for a given num_workers.
Off by default: turning it on changes which sequences the fixed benchmarks contain, so results
are only comparable between runs that agree on the flag (and on num_workers).
"""
import numpy as np
import torch


def _unwrap(ds):
    while not hasattr(ds, "rng") and hasattr(ds, "dataset"):
        ds = ds.dataset          # torch.utils.data.Subset (random_split) and friends
    return ds


def worker_init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    np.random.seed(info.seed % (2 ** 32))
    ds = _unwrap(info.dataset)
    if getattr(ds, "random", True) is False and hasattr(ds, "rng") and hasattr(ds, "seed"):
        ds.rng = np.random.RandomState((int(ds.seed) + 9973 * (worker_id + 1)) % (2 ** 32))
