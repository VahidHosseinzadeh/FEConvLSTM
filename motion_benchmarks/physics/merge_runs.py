#!/usr/bin/env python
"""
Merge per-run HDF5 files (rbc3d_dedalus.py / convert_oceananigans.py) into ONE file with
/fields (n_runs, n_snap, 4, nz, ny, nx), the layout RBC3DSource reads.

By default this writes an HDF5 VIRTUAL dataset: no data are copied, the merged file just points
at the run files (keep them next to it; relative paths are stored). --copy materialises a
self-contained file instead.

    python -m motion_benchmarks.physics.merge_runs --inputs rbc_runs/run_*.h5 --out rbc3d_ra2500.h5
"""
import argparse
import glob
import os

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy", action="store_true")
    a = ap.parse_args(argv)
    import h5py

    files = sorted({p for pat in a.inputs for p in glob.glob(pat)})
    if not files:
        raise SystemExit("no input files")
    shapes, good = [], []
    for p in files:
        with h5py.File(p, "r") as f:
            n_written = int(f.attrs.get("n_written", f["fields"].shape[0]))
            if n_written < f["fields"].shape[0]:
                print(f"[merge_runs] skipping incomplete run {p} ({n_written} snapshots)")
                continue
            shapes.append(f["fields"].shape)
            good.append(p)
    if len(set(shapes)) != 1:
        raise SystemExit(f"runs disagree in shape: {sorted(set(shapes))}")
    shape = (len(good),) + shapes[0]
    out_dir = os.path.dirname(os.path.abspath(a.out))
    with h5py.File(good[0], "r") as f0:
        attrs = dict(f0.attrs)
        names = f0["fields"].attrs.get("channel_names", np.array([b"T", b"u", b"v", b"w"]))
    with h5py.File(a.out, "w") as g:
        if a.copy:
            d = g.create_dataset("fields", shape=shape, dtype="float32",
                                 chunks=(1, 1) + shape[2:], compression="lzf")
            for i, p in enumerate(good):
                with h5py.File(p, "r") as f:
                    for t in range(shape[1]):
                        d[i, t] = f["fields"][t]
        else:
            layout = h5py.VirtualLayout(shape=shape, dtype="float32")
            for i, p in enumerate(good):
                rel = os.path.relpath(os.path.abspath(p), out_dir)
                layout[i] = h5py.VirtualSource(rel, "fields", shape=shapes[0])
            d = g.create_virtual_dataset("fields", layout, fillvalue=np.nan)
        d.attrs["channel_names"] = names
        for k, v in attrs.items():
            if k not in ("seed", "n_written"):
                g.attrs[k] = v
        g.attrs["runs"] = np.array([os.path.basename(p).encode() for p in good])
    print(f"[merge_runs] {a.out}: fields {shape} ({'copied' if a.copy else 'virtual'})")


if __name__ == "__main__":
    main()
