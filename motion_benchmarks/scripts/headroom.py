#!/usr/bin/env python
"""
How much can TRANSPORT buy on a dataset, before training anything?

headroom(lead) = MSE(Eulerian persistence) / MSE(Lagrangian persistence)

with the Lagrangian velocity estimated by phase correlation from the context (and, when the data
carry ground truth, the oracle velocity too). It is the first-order ceiling on the transport
component of the ConvLSTM -> MEConvLSTM gap: a trained ConvLSTM beats Eulerian persistence, a
trained MEConvLSTM should beat Lagrangian persistence. --sweep varies one dataset option and
prints one row per value -- the lifetime sweep on synthetic radar is the headline figure.

    python -m motion_benchmarks.scripts.headroom --dataset radar_synthetic \
        --sweep radar_lifetime=3,6,12,24,48 --n 96 --input_frames 12 --pred_frames 12
    python -m motion_benchmarks.scripts.headroom --dataset swift_hohenberg --n 64

Every option of train_motion.py is accepted (dataset options, --pc_alpha, --pc_window, ...).
"""
import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from motion_benchmarks import train_motion as tm  # noqa: E402
from motion_benchmarks.datasets.registry import REGISTRY  # noqa: E402
from motion_benchmarks.models.factory import pc_window, phase_corr_kwargs  # noqa: E402
from motion_benchmarks.models.persistence import Persistence  # noqa: E402


def run(args, n, batch=16):
    data = REGISTRY[args.dataset].build(args)
    meta = data["meta"]
    ds = data["test"]
    ld = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=args.num_workers)
    pkw = phase_corr_kwargs(args, meta)
    models = {"eulerian": Persistence("eulerian"),
              "lagrangian": Persistence("lagrangian", "estimated", alpha=pkw["alpha"],
                                        subpixel=pkw["subpixel"], radius=pkw["search_radius"],
                                        channels=meta.get("pc_channels"),
                                        window=pc_window(args, meta))}
    if meta.get("has_motion"):
        models["oracle"] = Persistence("lagrangian", "oracle")
    crit = torch.nn.MSELoss()
    out = {}
    for name, m in models.items():
        out[name] = tm.evaluate(m, ld, args, meta, torch.device("cpu"), crit,
                                max_batches=int(np.ceil(n / batch)))
    e = np.array(out["eulerian"]["mse"])
    res = {"eulerian_mse": e.tolist(),
           "lagrangian_mse": out["lagrangian"]["mse"],
           "headroom": (e / np.maximum(np.array(out["lagrangian"]["mse"]), 1e-12)).tolist()}
    if "oracle" in out:
        res["oracle_mse"] = out["oracle"]["mse"]
        res["headroom_oracle"] = (e / np.maximum(np.array(out["oracle"]["mse"]), 1e-12)).tolist()
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--sweep", type=str, default=None, help="option=v1,v2,... (dataset option)")
    ap.add_argument("--n", type=int, default=64, help="test sequences per setting")
    ap.add_argument("--leads", type=str, default="1,3,6,12")
    ap.add_argument("--out", type=str, default=None)
    own, rest = ap.parse_known_args(argv)
    if "--test_size" not in rest:
        rest += ["--test_size", str(own.n)]
    if "--num_workers" not in rest:
        rest += ["--num_workers", "0"]
    args = tm.get_args(rest + ["--no_baselines"])
    leads = [int(v) for v in own.leads.split(",") if int(v) <= args.pred_frames]
    rows = []
    if own.sweep:
        key, vals = own.sweep.split("=")
        for v in vals.split(","):
            setattr(args, key, v if isinstance(getattr(args, key), str) or getattr(args, key) is None
                    else type(getattr(args, key))(v))
            rows.append((f"{key}={v}", run(args, own.n)))
    else:
        rows.append((args.dataset, run(args, own.n)))

    head = f"{'setting':>24}" + "".join(f"{'lead ' + str(L):>10}" for L in leads)
    print("\nheadroom = MSE(Eulerian) / MSE(Lagrangian, estimated velocity)")
    print(head)
    for name, r in rows:
        print(f"{name:>24}" + "".join(f"{r['headroom'][L - 1]:>10.2f}" for L in leads))
    if "headroom_oracle" in rows[0][1]:
        print("\nwith the ORACLE velocity")
        print(head)
        for name, r in rows:
            print(f"{name:>24}" + "".join(f"{r['headroom_oracle'][L - 1]:>10.2f}" for L in leads))
    if own.out:
        Path(own.out).parent.mkdir(parents=True, exist_ok=True)
        with open(own.out, "w") as f:
            json.dump({name: r for name, r in rows}, f, indent=1)
        print(f"\nwrote {own.out}")
    return rows


if __name__ == "__main__":
    main()
