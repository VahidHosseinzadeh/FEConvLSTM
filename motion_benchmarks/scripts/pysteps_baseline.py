#!/usr/bin/env python
"""
The OPERATIONAL nowcasting baseline: dense optical flow (Lucas-Kanade, pySTEPS) + semi-Lagrangian
extrapolation, evaluated on the same test set and with the same metrics as train_motion.py.

This is stronger than the global-velocity Lagrangian persistence in the results files (a dense
flow field can advect different storms differently), so it is the number a nowcasting reviewer
will ask for. Needs `pip install pysteps`.

    python -m motion_benchmarks.scripts.pysteps_baseline --dataset radar_real \
        --radar_file sevir_vil96.h5 --input_frames 12 --pred_frames 12 --n 500 \
        --out experiments/motion_benchmarks/pysteps_lk.json
"""
import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from motion_benchmarks import train_motion as tm  # noqa: E402
from motion_benchmarks.datasets.registry import REGISTRY  # noqa: E402


class PystepsExtrapolation(nn.Module):
    """LK optical flow on the last `n_flow` frames, semi-Lagrangian extrapolation."""

    def __init__(self, method="LK", n_flow=3):
        super().__init__()
        from pysteps import extrapolation, motion
        self.oflow = motion.get_method(method)
        self.extrap = extrapolation.get_method("semilagrangian")
        self.n_flow = n_flow
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    @torch.no_grad()
    def forward(self, inp, pred_len, motion=None):
        out = []
        x = inp[:, -self.n_flow:, 0].cpu().double().numpy()
        for b in range(x.shape[0]):
            V = self.oflow(x[b])
            fc = self.extrap(x[b, -1], V, pred_len, outval=0.0)
            out.append(np.nan_to_num(fc, nan=0.0))
        y = torch.as_tensor(np.stack(out), dtype=inp.dtype, device=inp.device)
        return y[:, :, None]


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--method", default="LK")
    ap.add_argument("--out", type=str, default=None)
    own, rest = ap.parse_known_args(argv)
    if "--test_size" not in rest:
        rest += ["--test_size", str(own.n)]
    args = tm.get_args(rest + ["--no_baselines"])
    data = REGISTRY[args.dataset].build(args)
    meta = data["meta"]
    ld = DataLoader(data["test"], batch_size=16, shuffle=False, num_workers=args.num_workers)
    model = PystepsExtrapolation(own.method)
    import motion_benchmarks.models.factory as fac
    orig = fac.run_model

    def run_model(m, inp, pred_len, target=None, motion=None, track=None, want_velocity=False):
        if isinstance(m, PystepsExtrapolation):
            return m(inp, pred_len), None
        return orig(m, inp, pred_len, target, motion, track, want_velocity)
    tm.run_model = run_model
    res = tm.evaluate(model, ld, args, meta, torch.device("cpu"), nn.MSELoss())
    print(f"pySTEPS {own.method} + semi-Lagrangian: MSE per lead {np.round(res['mse'], 5).tolist()}")
    if own.out:
        Path(own.out).parent.mkdir(parents=True, exist_ok=True)
        with open(own.out, "w") as f:
            json.dump(dict(args=vars(args), test=res), f, indent=1, default=str)
        print(f"wrote {own.out}")


if __name__ == "__main__":
    main()
