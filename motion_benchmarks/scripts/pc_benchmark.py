#!/usr/bin/env python
"""
Phase-correlation accuracy on a dataset with ground-truth motion: per-step velocity error for a
grid of whitening exponents (alpha) and, optionally, search windows and Hann tapering.

Reproduces the measurements behind the defaults (claude/fluids_rbc_plan.md section 5,
third_experiment_options.md): alpha = 1 is right for broadband textures, alpha ~ 0.5 for
narrowband physics fields and rain, ~0.25 with a Hann window for noisy non-periodic crops
(calcium). Multi-motion data report the error of each true motion against its best-matching
peak (top-K peaks with suppression, or the residual estimator with --residual).

    python -m motion_benchmarks.scripts.pc_benchmark --dataset swift_hohenberg --alphas 1,0.5,0
    python -m motion_benchmarks.scripts.pc_benchmark --dataset calcium --alphas 1,0.5,0.25 --windows 0,1
"""
import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from motion_benchmarks import train_motion as tm  # noqa: E402
from motion_benchmarks.common.phase_correlation import (hann2d, phase_correlate,  # noqa: E402
                                                        residual_peaks)
from motion_benchmarks.datasets.registry import REGISTRY  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--alphas", type=str, default="1,0.5,0.25,0")
    ap.add_argument("--windows", type=str, default="0", help="0,1 = without / with Hann taper")
    ap.add_argument("--radius", type=int, default=None)
    ap.add_argument("--k", type=int, default=None, help="peaks per pair (default: n_motions)")
    ap.add_argument("--residual", action="store_true")
    ap.add_argument("--n", type=int, default=32, help="sequences")
    own, rest = ap.parse_known_args(argv)
    if "--test_size" not in rest:
        rest += ["--test_size", str(own.n)]
    args = tm.get_args(rest + ["--no_baselines", "--num_workers", "0"])
    data = REGISTRY[args.dataset].build(args)
    meta, ds = data["meta"], data["test"]
    if not meta.get("has_motion"):
        raise SystemExit(f"{args.dataset} has no ground-truth motion")
    radius = own.radius if own.radius is not None else meta.get("pc_search_radius")
    items = [ds[i] for i in range(min(own.n, len(ds)))]
    seq = torch.stack([it[0] for it in items])            # B, T, C, H, W
    mot = torch.stack([it[2] for it in items])            # B, T, N, 2
    B, T, C, H, W = seq.shape
    N = mot.shape[2]
    k = own.k or N
    ch = meta.get("pc_channels")
    a = seq[:, :-1].reshape(B * (T - 1), C, H, W)
    b = seq[:, 1:].reshape(B * (T - 1), C, H, W)
    true = mot[:, :-1].reshape(B * (T - 1), N, 2)
    print(f"{args.dataset}: {B} sequences x {T - 1} steps, {N} true motion(s), radius={radius}")
    print(f"{'alpha':>6} {'hann':>5} " + " ".join(f"{'m' + str(j) + ' med':>9} {'p90':>6}"
                                                  for j in range(N)))
    for w in [int(v) for v in own.windows.split(",")]:
        win = hann2d(H, W) if w else None
        for al in [float(v) for v in own.alphas.split(",")]:
            if own.residual:
                est, _ = residual_peaks(a, b, k=k, alpha=al, radius=radius, channels=ch)
            else:
                est, _, _ = phase_correlate(a, b, k=k, alpha=al, radius=radius, window=win,
                                            channels=ch, suppress=2)
            d = torch.linalg.norm(est[:, :, None, :] - true[:, None, :, :], dim=-1)  # M,k,N
            err = d.min(dim=1).values.numpy()                                       # M,N
            print(f"{al:>6.2f} {w:>5d} " + " ".join(
                f"{np.median(err[:, j]):>9.3f} {np.percentile(err[:, j], 90):>6.2f}" for j in range(N)))


if __name__ == "__main__":
    main()
