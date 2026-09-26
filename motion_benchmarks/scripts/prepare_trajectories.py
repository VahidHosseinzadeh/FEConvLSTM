#!/usr/bin/env python
"""
Convert levelX (inD / rounD / exiD / uniD) or highD CSV recordings into one compact .npz per
recording for the `trajectories` dataset (loading 100 MB CSVs every run is slow).

    python -m motion_benchmarks.scripts.prepare_trajectories --dataset round \
        --raw_dir ~/data/rounD-dataset-v1.0/data --out_dir ./data/trajectories/round
    python -m motion_benchmarks.scripts.prepare_trajectories --dataset synthetic \
        --synthetic_kind roundabout --n 20 --out_dir ./data/trajectories/synthetic

Then: train_motion.py --dataset trajectories --traj_source round --traj_dir ./data/trajectories/round

Recommended rendering: rounD / inD --traj_meters_per_px 0.5 --traj_fps 5 (about 2 px per step,
direction rotating on the ring); highD --traj_meters_per_px 1.0 --traj_fps 10.
"""
import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from motion_benchmarks.datasets.trajectories import (find_recordings, load_highd,  # noqa: E402
                                                     load_levelx, synthetic_tracks)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True,
                    choices=["round", "ind", "exid", "unid", "highd", "synthetic"])
    ap.add_argument("--raw_dir", type=str, default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--synthetic_kind", default="roundabout",
                    choices=["roundabout", "intersection", "highway"])
    ap.add_argument("--n", type=int, default=20, help="synthetic recordings")
    ap.add_argument("--duration", type=float, default=600.0, help="synthetic seconds each")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.dataset == "synthetic":
        for i in range(a.n):
            tr = synthetic_tracks(a.synthetic_kind, a.duration, seed=i)
            tr.save(out / f"{a.synthetic_kind}_{i:02d}.npz")
        print(f"[prepare_trajectories] wrote {a.n} synthetic recordings to {out}")
        return
    if not a.raw_dir:
        raise SystemExit("--raw_dir required")
    pres = find_recordings(a.raw_dir, a.dataset)
    if not pres:
        raise SystemExit(f"no *_tracks.csv under {a.raw_dir}")
    for pre in pres:
        tr = load_highd(pre) if a.dataset == "highd" else load_levelx(pre, a.dataset)
        dest = out / f"{a.dataset}_{os.path.basename(pre)}.npz"
        tr.save(dest)
        speed = np.hypot(np.diff(tr.x), np.diff(tr.y))
        print(f"[prepare_trajectories] {pre} -> {dest}: {len(np.unique(tr.track_id))} road users, "
              f"{tr.f1 - tr.f0 + 1} frames at {tr.frame_rate:g} fps")


if __name__ == "__main__":
    main()
