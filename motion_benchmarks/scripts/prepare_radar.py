#!/usr/bin/env python
"""
Build a radar archive (/frames (events, T, H, W) + attrs) for the radar_real dataset.

Sources
-------
sevir    SEVIR VIL files, dataset 'vil' (N, 384, 384, 49) uint8, 1 km, 5 min. --downsample 4
         gives 96 x 96 at 4 km (block mean). CSI thresholds: the standard SEVIR VIL levels
         16, 74, 133, 160, 181, 219 (pixel units, stored / 255).
             aws s3 cp --no-sign-request \
                 s3://sevir/data/vil/2019/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5 .
             (or https://sevir.s3.amazonaws.com/data/vil/2019/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5)
pysteps  the pySTEPS example cases (<= 24 frames each: a qualitative figure, not a training set)
             --pysteps_case fmi | mch | bom | knmi | mrms | opera
         or your own archive configured in a pystepsrc data source
             --pysteps_source mch --start 201505151600 --n_frames 144
         rain rate in mm/h, cut into --tile x --tile tiles (NaN outside the radar domain -> the
         tile is dropped unless --keep_nan_tiles).
array    .npy / .npz / .h5 arrays of rain rate (T, H, W) or (N, T, H, W) (--key picks the npz/h5
         entry), e.g. KNMI / TAASRAD19 / MRMS converted by you.

    python -m motion_benchmarks.scripts.prepare_radar --source sevir \
        --inputs SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5 --downsample 4 --out sevir_vil96.h5
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np

SEVIR_VIL_THRESHOLDS = [16, 74, 133, 160, 181, 219]


def block_mean(a, k):
    if k == 1:
        return a
    *lead, H, W = a.shape
    a = a[..., :H // k * k, :W // k * k]
    return a.reshape(*lead, H // k, k, W // k, k).mean(axis=(-3, -1))


def tiles(seq, tile, stride, keep_nan=False):
    """(T, H, W) -> list of (T, tile, tile) windows."""
    T, H, W = seq.shape
    out = []
    for y in range(0, H - tile + 1, stride):
        for x in range(0, W - tile + 1, stride):
            w = seq[:, y:y + tile, x:x + tile]
            if not keep_nan and not np.isfinite(w).all():
                continue
            out.append(np.nan_to_num(w, nan=0.0))
    return out


def write(out, frames, **attrs):
    import h5py
    frames = np.asarray(frames)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.create_dataset("frames", data=frames, chunks=(1,) + frames.shape[1:],
                         compression="gzip", compression_opts=4)
        for k, v in attrs.items():
            f.attrs[k] = v
    print(f"[prepare_radar] wrote {out}: frames {frames.shape} {frames.dtype}")


def from_sevir(files, downsample, max_events=None):
    import h5py
    evs = []
    for p in files:
        with h5py.File(p, "r") as f:
            vil = f["vil"]
            n = vil.shape[0] if max_events is None else min(vil.shape[0], max_events - len(evs))
            for i in range(n):
                e = np.transpose(vil[i], (2, 0, 1)).astype(np.float32)        # (49, 384, 384)
                evs.append(np.round(block_mean(e, downsample)).astype(np.uint8))
        if max_events is not None and len(evs) >= max_events:
            break
    return np.stack(evs)


def from_pysteps(case=None, source=None, start=None, n_frames=24, data_dir=None):
    import pysteps
    from pysteps import datasets, io
    if data_dir:
        if not Path(data_dir).exists() or not any(Path(data_dir).iterdir()):
            datasets.download_pysteps_data(data_dir, force=True)       # ~ 200 MB from GitHub
        rc = datasets.create_default_pystepsrc(data_dir, config_dir=data_dir)
        pysteps.load_config_file(rc)
    if case:
        R, meta, dt = datasets.load_dataset(case=case, frames=min(n_frames, 24 if case != "mrms" else 35))
    else:
        from datetime import datetime
        ds = pysteps.rcparams.data_sources[source]
        date = datetime.strptime(start, "%Y%m%d%H%M")
        fns = io.archive.find_by_date(date, ds["root_path"], ds["path_fmt"], ds["fn_pattern"],
                                      ds["fn_ext"], ds["timestep"], num_next_files=n_frames - 1)
        importer = io.get_method(ds["importer"], "importer")
        R, _, meta = io.read_timeseries(fns, importer, **ds["importer_kwargs"])
        from pysteps.utils import conversion
        R, meta = conversion.to_rainrate(R, meta)
        dt = ds["timestep"]
    km = float(meta.get("xpixelsize", 1000.0)) / 1000.0
    return np.asarray(R, dtype=np.float32), float(dt), km


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", required=True, choices=["sevir", "pysteps", "array"])
    ap.add_argument("--inputs", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--max_events", type=int, default=None)
    ap.add_argument("--tile", type=int, default=96)
    ap.add_argument("--tile_stride", type=int, default=96)
    ap.add_argument("--keep_nan_tiles", action="store_true")
    ap.add_argument("--key", type=str, default=None)
    ap.add_argument("--dt_minutes", type=float, default=5.0)
    ap.add_argument("--km_per_px", type=float, default=1.0)
    ap.add_argument("--pysteps_case", type=str, default=None)
    ap.add_argument("--pysteps_source", type=str, default=None)
    ap.add_argument("--pysteps_data_dir", type=str, default=None)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--n_frames", type=int, default=24)
    a = ap.parse_args(argv)
    files = sorted({q for pat in a.inputs for q in glob.glob(pat)})

    if a.source == "sevir":
        if not files:
            raise SystemExit("--inputs SEVIR_VIL_*.h5 required")
        fr = from_sevir(files, a.downsample, a.max_events)
        write(a.out, fr, scale=1.0 / 255.0, offset=0.0, quantity="vil", dt_minutes=5.0,
              km_per_px=1.0 * a.downsample,
              thresholds=np.array(SEVIR_VIL_THRESHOLDS, dtype=np.float64) / 255.0)
        return
    if a.source == "pysteps":
        R, dt, km = from_pysteps(a.pysteps_case, a.pysteps_source, a.start, a.n_frames,
                                 a.pysteps_data_dir)
        R = block_mean(R, a.downsample)
        ev = tiles(R, a.tile, a.tile_stride, a.keep_nan_tiles)
        if not ev:
            raise SystemExit("no complete tiles: lower --tile or pass --keep_nan_tiles")
        write(a.out, np.stack(ev).astype(np.float16), scale=1.0, offset=0.0,
              quantity="rain_rate", dt_minutes=dt, km_per_px=km * a.downsample,
              thresholds=np.array([0.5, 2.0, 8.0]))
        return
    evs = []
    for p in files:
        if p.endswith(".npy"):
            x = np.load(p)
        elif p.endswith(".npz"):
            z = np.load(p)
            x = z[a.key or list(z.keys())[0]]
        else:
            import h5py
            with h5py.File(p, "r") as f:
                x = np.asarray(f[a.key or list(f.keys())[0]])
        x = np.asarray(x, dtype=np.float32)
        seqs = [x] if x.ndim == 3 else list(x)
        for s in seqs:
            s = block_mean(s, a.downsample)
            evs += tiles(s, a.tile, a.tile_stride, a.keep_nan_tiles) if s.shape[-1] > a.tile else [s]
    if not evs:
        raise SystemExit("no input sequences")
    write(a.out, np.stack(evs).astype(np.float16), scale=1.0, offset=0.0, quantity="rain_rate",
          dt_minutes=a.dt_minutes, km_per_px=a.km_per_px * a.downsample,
          thresholds=np.array([0.5, 2.0, 8.0]))


if __name__ == "__main__":
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
