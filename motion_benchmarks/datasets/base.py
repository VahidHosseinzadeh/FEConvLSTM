"""
Base class for every dataset in this package.

Randomness -- designed so the DataLoader-worker duplication bug cannot happen
----------------------------------------------------------------------------
random=True   (training) each __getitem__ seeds a fresh numpy Generator from torch's RNG. PyTorch
              seeds torch differently in every worker and every epoch, so streams never repeat;
              the whole run is still reproducible under torch.manual_seed.
random=False  (fixed benchmark) item i is a PURE FUNCTION of (seed, i):
              np.random.default_rng([seed, i]). Identical for any num_workers, batch size or
              access order, and no reset_rng() bookkeeping is needed (reset_rng exists only for
              API compatibility with the Moving MNIST datasets).

Item format (the repo's): (seq, label, motion)
    seq    float32 (T, C, H, W)
    label  int (0 when the task has none)
    motion float32 (T, N, 2), (vx, vy) px per step, motion[t] = displacement t -> t+1.
           All-NaN when the data has no ground-truth motion (real radar, satellite): the trainer
           then skips velocity metrics and refuses oracle models.

`meta` (a dict on the instance) tells the trainer what the data is: channels, whether motion is
known, CSI/FSS thresholds, the Galilean mean-flow connection for fluids, etc.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


def nan_motion(T, N=1):
    return torch.full((T, N, 2), float("nan"), dtype=torch.float32)


class GeneratedSequenceDataset(Dataset):

    def __init__(self, length, seed=0, random=True):
        self.length = int(length)
        self.seed = int(seed)
        self.random = bool(random)
        self.meta = {}

    def __len__(self):
        return self.length

    def reset_rng(self):
        """No-op: fixed items are functions of (seed, index). Kept for API compatibility."""

    def rng_for(self, index):
        if self.random:
            return np.random.default_rng(int(torch.randint(0, 2 ** 62, (1,)).item()))
        return np.random.default_rng([self.seed, int(index)])

    def __getitem__(self, index):
        if index < 0 or index >= self.length:
            raise IndexError(index)
        return self.generate(self.rng_for(index), index)

    def generate(self, rng, index):   # pragma: no cover - abstract
        raise NotImplementedError

    @staticmethod
    def pack(seq, motion, label=0):
        seq = torch.as_tensor(np.ascontiguousarray(seq), dtype=torch.float32)
        if seq.dim() == 3:
            seq = seq.unsqueeze(1)
        motion = torch.as_tensor(np.ascontiguousarray(motion), dtype=torch.float32)
        if motion.dim() == 2:
            motion = motion.unsqueeze(1)
        return seq, int(label), motion
