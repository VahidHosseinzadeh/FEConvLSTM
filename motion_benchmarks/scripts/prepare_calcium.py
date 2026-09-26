#!/usr/bin/env python
"""
Pack motion-CORRECTED two-photon movies (and optionally their registration traces) into the
archive the `calcium` dataset reads: /movies/<name> (T, H, W) float32, /traces/<name> (T, 2).

Inputs (any mix):
  *.tif / *.tiff   a stack (needs `tifffile`), or a directory of single-frame TIFFs (Neurofinder)
  *.npy            (T, H, W)
  *.h5 / *.nwb     --h5_key names the (T, H, W) dataset (use --inspect to list the file)
  suite2p/plane0   a Suite2p output folder: registered movie data.bin + ops.npy (Ly, Lx,
                   nframes, xoff, yoff); the rigid offsets become /traces/<name>
  --synthetic N    N synthetic movies (cells with calcium transients) for a dry run

Frames are cropped to --max_size around the centre and --max_frames long to bound the archive.
The traces are the offsets the registration APPLIED; re-applying them (train_motion.py
--ca_motion real) injects real brain motion into a registered movie. The sign convention does
not matter for the statistics of the motion.

    python -m motion_benchmarks.scripts.prepare_calcium --inputs suite2p/plane0 --out ca.h5
    python -m motion_benchmarks.scripts.prepare_calcium --synthetic 8 --out ca_synth.h5
"""
import argparse
import glob
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from motion_benchmarks.datasets.calcium import synthetic_calcium_movie  # noqa: E402


def read_suite2p(folder):
    ops = np.load(os.path.join(folder, "ops.npy"), allow_pickle=True).item()
    Ly, Lx, n = int(ops["Ly"]), int(ops["Lx"]), int(ops["nframes"])
    mv = np.memmap(os.path.join(folder, "data.bin"), dtype=np.int16, mode="r", shape=(n, Ly, Lx))
    tr = None
    if "xoff" in ops and "yoff" in ops:
        tr = np.stack([np.asarray(ops["xoff"], float), np.asarray(ops["yoff"], float)], axis=1)
    return mv, tr, float(ops.get("fs", 30.0))


def read_any(path, h5_key=None):
    p = Path(path)
    if p.is_dir() and (p / "ops.npy").exists():
        return read_suite2p(str(p))
    if p.is_dir():
        import tifffile
        files = sorted(glob.glob(str(p / "*.tif*")))
        return np.stack([tifffile.imread(f) for f in files]), None, None
    if p.suffix in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(str(p)), None, None
    if p.suffix == ".npy":
        return np.load(str(p), mmap_mode="r"), None, None
    import h5py
    f = h5py.File(str(p), "r")
    if not h5_key:
        raise SystemExit(f"{p}: pass --h5_key (run with --inspect to list datasets)")
    return f[h5_key], None, None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inputs", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--h5_key", type=str, default=None)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--max_size", type=int, default=256)
    ap.add_argument("--max_frames", type=int, default=6000)
    ap.add_argument("--synthetic", type=int, default=0)
    a = ap.parse_args(argv)
    if a.inspect:
        from motion_benchmarks.datasets.fluids import inspect_h5
        for p in a.inputs:
            inspect_h5(p)
        return
    import h5py
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fps_seen = []
    with h5py.File(a.out, "w") as f:
        gm, gt = f.create_group("movies"), f.create_group("traces")
        k = 0
        for i in range(a.synthetic):
            mv = synthetic_calcium_movie(T=min(a.max_frames, 1500), H=a.max_size // 2 + 48,
                                         W=a.max_size // 2 + 48, seed=i)
            gm.create_dataset(f"synthetic_{i:03d}", data=mv, chunks=(1,) + mv.shape[1:],
                              compression="lzf")
            k += 1
        for p in a.inputs:
            mv, tr, fps = read_any(p, a.h5_key)
            T, H, W = mv.shape
            s = min(a.max_size, H, W)
            y0, x0 = (H - s) // 2, (W - s) // 2
            n = min(T, a.max_frames)
            name = f"{k:03d}_" + Path(p).name.replace(".", "_")
            d = gm.create_dataset(name, shape=(n, s, s), dtype="float32",
                                  chunks=(1, s, s), compression="lzf")
            for t0 in range(0, n, 500):
                d[t0:t0 + 500] = np.asarray(mv[t0:min(n, t0 + 500), y0:y0 + s, x0:x0 + s],
                                            dtype=np.float32)
            if tr is not None:
                gt.create_dataset(name, data=np.asarray(tr[:n], np.float32))
            if fps:
                fps_seen.append(fps)
            print(f"[prepare_calcium] {p} -> movies/{name} ({n}, {s}, {s})"
                  + (" + trace" if tr is not None else ""))
            k += 1
        f.attrs["fps"] = float(a.fps or (np.median(fps_seen) if fps_seen else 30.0))
    print(f"[prepare_calcium] wrote {a.out} ({k} movies)")


if __name__ == "__main__":
    main()
