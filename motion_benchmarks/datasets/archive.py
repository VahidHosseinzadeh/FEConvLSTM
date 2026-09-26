"""
Real image sequences from a prepared HDF5 archive: radar (scripts/prepare_radar.py) and
geostationary satellite imagery (scripts/download_goes.py) share one layout and one loader.

Archive layout
--------------
/frames   (n_events, T_event, H, W)  uint8 / float16 / float32, raw values
attrs     scale, offset        physical = raw * scale + offset
          quantity             'rain_rate' (mm/h) | 'vil' (SEVIR, 0..1 after scaling)
                               | 'brightness_temperature' (K, GOES ABI IR bands)
                               | 'reflectance' (0..1, GOES ABI visible bands)
          dt_minutes, km_per_px
          thresholds           physical thresholds for CSI / FSS
                               (rain rate: 0.5, 2, 8 mm/h; SEVIR VIL: the standard
                               16, 74, 133, 160, 181, 219 pixel levels / 255; IR: cloud-top
                               temperatures 260, 235, 210 K -- colder is higher cloud)

Model input: rain rate -> log1p(R) / 4 (as the synthetic radar); VIL and reflectance -> the
value itself; brightness temperature -> (320 K - BT) / 120 K (cold cloud tops bright).
Crops are random in training and deterministic (by index) for the fixed sets. No ground-truth
motion exists: motion is NaN, velocity metrics are skipped and the oracle models are refused.
The canvas is not periodic, so phase correlation uses a Hann window (meta periodic=False).

`augment` applies the dihedral group D4 (90-degree rotations and flips) to training windows --
a symmetry the physics does not quite have (the mean wind has a climatological direction), so
it is off by default; it is there for the G_mot x| G_glob experiments.
"""
import os

import numpy as np

from .base import GeneratedSequenceDataset, nan_motion


def to_model_units(x, quantity):
    if quantity == "rain_rate":
        return np.log1p(np.maximum(x, 0.0)) / 4.0
    if quantity == "brightness_temperature":
        return (320.0 - np.asarray(x, dtype=np.float64)) / 120.0
    return x


class FrameArchiveDataset(GeneratedSequenceDataset):

    def __init__(self, path, length, seq_len, events=None, crop=None, time_stride=1,
                 min_wet_fraction=0.0, wet_threshold=None, augment=False, seed=0, random=True,
                 max_tries=20):
        super().__init__(length, seed, random)
        import h5py
        self.path = str(path)
        with h5py.File(self.path, "r") as f:
            d = f["frames"]
            self.n_events, self.T_event, self.H0, self.W0 = d.shape
            a = dict(f.attrs)
        self.scale = float(a.get("scale", 1.0))
        self.offset = float(a.get("offset", 0.0))
        self.quantity = str(a.get("quantity", "rain_rate"))
        self.events = list(range(self.n_events)) if events is None else list(events)
        self.seq_len = int(seq_len)
        self.stride = int(time_stride)
        self.crop = int(crop) if crop else None
        self.min_wet = float(min_wet_fraction)
        thr = np.asarray(a.get("thresholds", [0.5, 2.0, 8.0]), dtype=np.float64)
        if wet_threshold is None:
            wet_threshold = thr.max() if self.quantity == "brightness_temperature" else thr.min()
        self.wet_threshold = float(wet_threshold)
        self.augment = bool(augment)
        self.max_tries = int(max_tries)
        span = (self.seq_len - 1) * self.stride + 1
        if span > self.T_event:
            raise ValueError(f"seq_len*stride={span} > {self.T_event} frames per event")
        self._span = span
        S = self.crop or self.H0
        self.meta = dict(name="archive_" + self.quantity, in_channels=1, out_channels=1,
                         channel_names=[self.quantity], has_motion=False, n_motions=1,
                         periodic=False, pc_alpha=0.5, quantity=self.quantity,
                         thresholds=[float(to_model_units(t, self.quantity)) for t in thr],
                         physical_thresholds=thr.tolist(), fss_scales=[1, 5, 9, 17],
                         dt_minutes=float(a.get("dt_minutes", 5.0)) * self.stride,
                         km_per_px=float(a.get("km_per_px", 1.0)), image_size=S)
        self._fh = None
        self._pid = None

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_fh"] = None
        d["_pid"] = None
        return d

    def _file(self):
        import h5py
        if self._pid != os.getpid():
            self._fh = h5py.File(self.path, "r")
            self._pid = os.getpid()
        return self._fh

    def generate(self, rng, index):
        d = self._file()["frames"]
        c = self.crop
        best = None
        for _ in range(self.max_tries):
            e = self.events[int(rng.integers(len(self.events)))]
            t0 = int(rng.integers(self.T_event - self._span + 1))
            if c:
                y0 = int(rng.integers(self.H0 - c + 1))
                x0 = int(rng.integers(self.W0 - c + 1))
                raw = d[e, t0:t0 + self._span:self.stride, y0:y0 + c, x0:x0 + c]
            else:
                raw = d[e, t0:t0 + self._span:self.stride]
            phys = raw.astype(np.float32) * self.scale + self.offset
            if self.quantity == "brightness_temperature":      # "wet" = cloud colder than thr
                wet = float((phys <= self.wet_threshold).mean())
            else:
                wet = float((phys >= self.wet_threshold).mean())
            if best is None or wet > best[0]:
                best = (wet, phys)
            if wet >= self.min_wet:
                break
        x = to_model_units(best[1], self.quantity).astype(np.float32)
        if self.augment:
            k = int(rng.integers(4))
            x = np.rot90(x, k, axes=(1, 2))
            if rng.random() < 0.5:
                x = x[:, :, ::-1]
        return self.pack(np.ascontiguousarray(x), nan_motion(self.seq_len).numpy(), label=0)


RadarH5Dataset = FrameArchiveDataset      # backwards-friendly alias
