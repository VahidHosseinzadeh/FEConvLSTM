"""
Is the dataset leaking? Train a SINGLE-FRAME classifier and see.

Why this exists
---------------
The obvious control is "lstm should sit at chance". That is WRONG, and believing
it will send you hunting for bugs that are not there.

A plain ConvLSTM is not motion-blind. Its cell convolves the input together with
the previous hidden state, which is enough to build local spatiotemporal
correlations -- Reichardt-style motion detectors. So it can notice that a region
moves differently from its surroundings, segment it roughly, and classify the
shape. What it CANNOT do is transport its hidden state, so it cannot accumulate
the figure coherently in a co-moving frame over many steps. Above chance is
therefore expected for `lstm`; the experimental claim is that melstm and felstm
beat it, not that it fails.

The honest leak test is a model that has NO access to motion at all: one frame,
no recurrence. If that beats chance, a per-frame cue exists and every recurrent
number is contaminated. That is what this probe measures, and it is what caught
the texture-seam leak at corr_len=1.0.

Usage
-----
    python moving_mnist/leak_probe.py                     # the current defaults
    python moving_mnist/leak_probe.py --corr_len 0 0.5 1 2
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common_fate_moving_mnist_dataset import CommonFateMovingMNISTDataset


def single_frame_cnn(n_classes=10):
    """Deliberately a strong per-frame model: if a cue exists, this should find it."""
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, 2, 1, padding_mode="circular"), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 64, 3, 2, 1, padding_mode="circular"), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 64, 3, 2, 1, padding_mode="circular"), nn.BatchNorm2d(64), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, n_classes))


def probe(corr_len, args, device):
    def mk(train, seed, rnd):
        return CommonFateMovingMNISTDataset(
            root=args.root, train=train, seq_len=1, image_size=args.image_size,
            max_speed=args.data_v_range, bg_opposite_at_start=True, min_dv=2,
            motion_mode="piecewise", corr_len=corr_len, variant=args.variant,
            return_motion=False, random=rnd, seed=seed, download=True)

    tr = DataLoader(Subset(mk(True, 1, True), list(range(args.n_train))),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    va_ds = mk(False, 2, False)
    va = DataLoader(Subset(va_ds, list(range(args.n_val))), batch_size=args.batch_size)

    net = single_frame_cnn().to(device)
    opt = torch.optim.Adam(net.parameters(), args.lr)
    crit = nn.CrossEntropyLoss()

    best = 0.0
    for _ in range(args.epochs):
        net.train()
        for x, y in tr:
            x, y = x[:, 0].to(device), y.to(device)
            opt.zero_grad(); crit(net(x), y).backward(); opt.step()
        net.eval(); va_ds.reset_rng(); c = n = 0
        with torch.no_grad():
            for x, y in va:
                x, y = x[:, 0].to(device), y.to(device)
                c += (net(x).argmax(1) == y).sum().item(); n += y.numel()
        best = max(best, c / n)
    return best


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--corr_len', type=float, nargs='+', default=[0.0, 0.5, 1.0, 2.0])
    p.add_argument('--image_size', type=int, default=36)
    p.add_argument('--data_v_range', type=int, default=2)
    p.add_argument('--variant', default='moving_mask')
    p.add_argument('--n_train', type=int, default=6000)
    p.add_argument('--n_val', type=int, default=1000)
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--root', default=str(_HERE.parent / 'data'))
    p.add_argument('--threshold', type=float, default=0.15,
                   help='Above this, treat the setting as leaking.')
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"single-frame CNN, {args.image_size}px, chance = 10%, device = {device}")
    print(f"{'corr_len':>9}{'best val acc':>15}   verdict")
    results = {}
    for cl in args.corr_len:
        t0 = time.time()
        acc = probe(cl, args, device)
        results[cl] = acc
        verdict = ("LEAKS — a per-frame cue exists, recurrent numbers are contaminated"
                   if acc > args.threshold else "clean")
        print(f"{cl:>9}{acc:>14.1%}   {verdict}   ({time.time()-t0:.0f}s)", flush=True)
    return results


if __name__ == "__main__":
    main()
