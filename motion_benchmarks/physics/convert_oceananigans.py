#!/usr/bin/env python
"""
Convert Oceananigans JLD2 output (physics/rbc3d_oceananigans.jl) to the per-run HDF5 layout of
rbc3d_dedalus.py: /fields (n_snap, 4, nz, ny, nx), channels T, u, v, w at CELL CENTRES.

JLD2 is HDF5 underneath, so h5py reads it: timeseries/<name>/<iteration> datasets, Julia
column-major (x, y, z) arrays appearing as (z, y, x). Oceananigans staggers the velocities
(u on x-faces, v on y-faces, w on z-faces, w with nz + 1 faces in a Bounded direction); they are
averaged onto cell centres here. Halos are stripped if present.

    python -m motion_benchmarks.physics.convert_oceananigans --inputs rbc_runs/run_*.jld2 \
        --outdir rbc_runs_h5 --t_start 100 --dt_snap 0.5
"""
import argparse
import glob
from pathlib import Path

import numpy as np


def _series(f, name):
    g = f["timeseries"][name]
    its = sorted((k for k in g.keys() if k.lstrip("-").isdigit()), key=int)
    return its, g


def _strip(a, n):
    """Remove symmetric halos so the last three axes are (nz[+1], ny, nx) as expected."""
    out = a
    for ax, target in zip((-3, -2, -1), n):
        extra = out.shape[ax] - target
        if extra > 0:
            h = extra // 2
            sl = [slice(None)] * out.ndim
            sl[ax] = slice(h, h + target)
            out = out[tuple(sl)]
    return out


def convert(path, out, t_start=None, dt_snap=None, Ra=2500.0, Pr=0.7):
    import h5py
    with h5py.File(path, "r") as f:
        its, gT = _series(f, "T")
        _, gu = _series(f, "u")
        _, gv = _series(f, "v")
        _, gw = _series(f, "w")
        _, gt = _series(f, "t")
        times = np.array([float(np.asarray(gt[i])) for i in its])
        keep = np.ones(len(its), bool) if t_start is None else times > t_start + 1e-9
        its = [i for i, k in zip(its, keep) if k]
        times = times[keep]
        T0 = np.asarray(gT[its[0]])
        nz, ny, nx = T0.shape[-3:]
        # a Bounded z adds one w face; infer the true cell count from T (which has no face)
        with h5py.File(out, "w") as g:
            d = g.create_dataset("fields", shape=(len(its), 4, nz, ny, nx), dtype="float32",
                                 chunks=(1, 4, nz, ny, nx), compression="lzf")
            d.attrs["channel_names"] = np.array([b"T", b"u", b"v", b"w"])
            for k, it in enumerate(its):
                T = _strip(np.asarray(gT[it], dtype=np.float64), (nz, ny, nx))
                u = _strip(np.asarray(gu[it], dtype=np.float64), (nz, ny, nx))
                v = _strip(np.asarray(gv[it], dtype=np.float64), (nz, ny, nx))
                w = _strip(np.asarray(gw[it], dtype=np.float64), (nz + 1, ny, nx))
                uc = 0.5 * (u + np.roll(u, -1, axis=-1))       # x-faces -> centres (periodic)
                vc = 0.5 * (v + np.roll(v, -1, axis=-2))       # y-faces -> centres (periodic)
                wc = 0.5 * (w[:-1] + w[1:])                    # z-faces -> centres
                d[k] = np.stack([T, uc, vc, wc]).astype(np.float32)
            g.create_dataset("time", data=times)
            dz = 2.0 / nz
            attrs = dict(Ra=Ra, Pr=Pr, kappa=(Ra * Pr) ** -0.5, nu=(Ra / Pr) ** -0.5,
                         Lx=2 * np.pi, Ly=2 * np.pi, Lz=2.0,
                         dt_snap=float(dt_snap if dt_snap else np.median(np.diff(times))),
                         t_start=float(times[0]), bottom_T=1.0, top_T=0.0,
                         solver="oceananigans", n_written=len(its))
            for kk, vv in attrs.items():
                g.attrs[kk] = vv
            g.attrs["z"] = -1.0 + (np.arange(nz) + 0.5) * dz
    return len(its)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--t_start", type=float, default=None,
                    help="drop snapshots at or before this time (e.g. the initial write)")
    ap.add_argument("--dt_snap", type=float, default=None)
    ap.add_argument("--Ra", type=float, default=2500.0)
    ap.add_argument("--Pr", type=float, default=0.7)
    a = ap.parse_args(argv)
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    for p in sorted({q for pat in a.inputs for q in glob.glob(pat)}):
        out = Path(a.outdir) / (Path(p).stem + ".h5")
        n = convert(p, out, a.t_start, a.dt_snap, a.Ra, a.Pr)
        print(f"[convert_oceananigans] {p} -> {out} ({n} snapshots)")


if __name__ == "__main__":
    main()
