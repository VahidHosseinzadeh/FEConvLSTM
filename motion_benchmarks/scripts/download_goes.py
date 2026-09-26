#!/usr/bin/env python
"""
Download GOES-R ABI imagery from the NOAA open-data buckets on AWS and build a frame archive
(/frames (events, T, H, W)) for the `satellite` dataset.

Why satellite clouds: they persist much longer than rain echoes -- the long-lifetime end of the
headroom table, where transport pays most -- and clouds at different heights move with different
winds, so one scene holds several coherently moving layers: the K-slot, local-motion case, in
real data. Operational atmospheric motion vectors are computed by exactly this kind of template
matching.

Buckets (anonymous HTTPS, no account):  noaa-goes19 (GOES-East since April 2025),
noaa-goes16 (GOES-East before), noaa-goes18 (GOES-West).
Product ABI-L2-CMIPC = Cloud and Moisture Imagery, CONUS sector, every 5 minutes, one file per
band; band 13 (10.3 um "clean" IR) is 2 km, brightness temperature in K, day and night.
Files: <bucket>/ABI-L2-CMIPC/<year>/<day-of-year>/<hour>/OR_ABI-L2-CMIPC-M6C13_G19_s...nc

Each event = one fixed crop location x `--event_frames` consecutive scans. Crops are
block-averaged by --downsample (512 px at 2 km, /4 -> 128 px at 8 km). Stored as uint16 in
0.01 K (scale 0.01) with quantity 'brightness_temperature'; the model sees (320 K - BT) / 120 K.

    python -m motion_benchmarks.scripts.download_goes --satellite 19 --band 13 \
        --start 2025-07-01T12 --hours 12 --crop 512 --downsample 4 --n_crops 6 \
        --event_frames 36 --out goes19_c13_128.h5

A raw file is deleted after it has been cropped, unless --keep_raw DIR is given.
"""
import argparse
import datetime as dt
import os
import re
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

S3 = "https://{bucket}.s3.amazonaws.com"
_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
_T = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)_")


def list_keys(bucket, prefix):
    """All object keys under prefix (S3 ListObjectsV2, paginated)."""
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        url = S3.format(bucket=bucket) + "/?" + urllib.parse.urlencode(q)
        with urllib.request.urlopen(url, timeout=60) as r:
            root = ET.fromstring(r.read())
        keys += [c.find(_NS + "Key").text for c in root.findall(_NS + "Contents")]
        trunc = root.find(_NS + "IsTruncated")
        if trunc is None or trunc.text != "true":
            return keys
        token = root.find(_NS + "NextContinuationToken").text


def scan_start(key):
    m = _T.search(key)
    if not m:
        return None
    y, doy, h, mi, s, _ = (int(v) for v in m.groups())
    return dt.datetime(y, 1, 1) + dt.timedelta(days=doy - 1, hours=h, minutes=mi, seconds=s)


def download(bucket, key, dest, retries=3):
    url = S3.format(bucket=bucket) + "/" + urllib.parse.quote(key)
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            return dest
        except Exception as exc:                  # noqa: BLE001 - network, retry
            if k == retries - 1:
                raise
            print(f"[download_goes] retry {key}: {exc}")
            time.sleep(2 * (k + 1))


def read_cmi(path):
    """CMI (y, x) as float32 with NaN fill. netCDF4 if present, else h5py + manual scaling."""
    try:
        from netCDF4 import Dataset
        with Dataset(path) as nc:
            v = nc.variables["CMI"]
            v.set_auto_maskandscale(True)
            a = v[:]
            return np.ma.filled(a.astype(np.float32), np.nan)
    except ImportError:
        import h5py
        with h5py.File(path, "r") as f:
            v = f["CMI"]
            raw = v[()].astype(np.float32)
            fill = v.attrs.get("_FillValue")
            sf = float(np.ravel(v.attrs.get("scale_factor", [1.0]))[0])
            off = float(np.ravel(v.attrs.get("add_offset", [0.0]))[0])
            out = raw * sf + off
            if fill is not None:
                out[raw == float(np.ravel(fill)[0])] = np.nan
            return out


def block_mean(a, k):
    if k == 1:
        return a
    H, W = a.shape
    a = a[:H // k * k, :W // k * k]
    return np.nanmean(a.reshape(H // k, k, W // k, k), axis=(1, 3))


def choose_crops(shape, crop, n, rng, margin=16):
    H, W = shape
    return [(int(rng.integers(margin, H - crop - margin)), int(rng.integers(margin, W - crop - margin)))
            for _ in range(n)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--satellite", type=int, default=19, choices=[16, 17, 18, 19])
    ap.add_argument("--product", default="ABI-L2-CMIPC")
    ap.add_argument("--band", type=int, default=13)
    ap.add_argument("--start", required=True, help="UTC start, e.g. 2025-07-01T12")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--every", type=int, default=1, help="keep every n-th scan")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--downsample", type=int, default=4)
    ap.add_argument("--n_crops", type=int, default=6)
    ap.add_argument("--crops", type=str, default=None, help="explicit 'r0,c0;r1,c1;...'")
    ap.add_argument("--event_frames", type=int, default=36)
    ap.add_argument("--max_gap_minutes", type=float, default=12.0,
                    help="a larger gap between scans starts a new event")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep_raw", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    bucket = f"noaa-goes{a.satellite}"
    t0 = dt.datetime.fromisoformat(a.start)
    t1 = t0 + dt.timedelta(hours=a.hours)
    tag = f"C{a.band:02d}_G{a.satellite}_"          # any scan mode (M3 / M4 / M6)
    keys, h = [], t0.replace(minute=0, second=0, microsecond=0)
    while h < t1:
        prefix = f"{a.product}/{h.year}/{h.timetuple().tm_yday:03d}/{h.hour:02d}/"
        ks = [k for k in list_keys(bucket, prefix) if tag in k]
        keys += [k for k in ks if scan_start(k) and t0 <= scan_start(k) < t1]
        h += dt.timedelta(hours=1)
    keys = sorted(set(keys), key=scan_start)[::a.every]
    if not keys:
        raise SystemExit(f"no files for {bucket} {a.product} band {a.band} in [{t0}, {t1})")
    print(f"[download_goes] {len(keys)} scans from {bucket}")

    rng = np.random.default_rng(a.seed)
    raw_dir = Path(a.keep_raw) if a.keep_raw else Path(tempfile.mkdtemp(prefix="goes_"))
    raw_dir.mkdir(parents=True, exist_ok=True)
    crops, series, times = None, [], []
    for i, k in enumerate(keys):
        dest = raw_dir / os.path.basename(k)
        if not dest.exists():
            download(bucket, k, dest)
        img = read_cmi(dest)
        if not a.keep_raw:
            dest.unlink()
        if crops is None:
            if a.crops:
                crops = [tuple(int(v) for v in c.split(",")) for c in a.crops.split(";")]
            else:
                crops = choose_crops(img.shape, a.crop, a.n_crops, rng)
        frame = [block_mean(img[r:r + a.crop, c:c + a.crop], a.downsample) for r, c in crops]
        series.append(np.stack(frame))
        times.append(scan_start(k))
        if (i + 1) % 12 == 0:
            print(f"[download_goes] {i + 1}/{len(keys)} {times[-1]}")

    # cut the time series into events at gaps and every event_frames scans
    series = np.stack(series)                                    # (T, n_crops, h, w)
    gaps = [0] + [j for j in range(1, len(times))
                  if (times[j] - times[j - 1]).total_seconds() / 60 > a.max_gap_minutes] + [len(times)]
    events = []
    for s0, s1 in zip(gaps[:-1], gaps[1:]):
        for e0 in range(s0, s1 - a.event_frames + 1, a.event_frames):
            for c in range(series.shape[1]):
                ev = series[e0:e0 + a.event_frames, c]
                if np.isfinite(ev).mean() > 0.99:
                    events.append(np.nan_to_num(ev, nan=float(np.nanmean(ev))))
    if not events:
        raise SystemExit("no complete events: lower --event_frames or extend --hours")
    frames = np.clip(np.round(np.stack(events) / 0.01), 0, 65535).astype(np.uint16)
    dtm = np.median([(times[j] - times[j - 1]).total_seconds() / 60 for j in range(1, len(times))])
    import h5py
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.out, "w") as f:
        f.create_dataset("frames", data=frames, chunks=(1,) + frames.shape[1:],
                         compression="gzip", compression_opts=4)
        f.attrs.update(dict(scale=0.01, offset=0.0, quantity="brightness_temperature",
                            dt_minutes=float(dtm), km_per_px=2.0 * a.downsample,
                            thresholds=np.array([260.0, 235.0, 210.0]),
                            satellite=a.satellite, band=a.band, product=a.product,
                            start=a.start, crops=np.array(crops)))
    print(f"[download_goes] wrote {a.out}: frames {frames.shape} (dt {dtm:.1f} min)")


if __name__ == "__main__":
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
