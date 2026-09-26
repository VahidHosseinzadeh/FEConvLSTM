"""
Rendered real trajectories -- inD / rounD / exiD / uniD / highD (levelXdata drone datasets).

The cleanest answer to "your motion is artificial": every road user moves along its REAL,
georeferenced trajectory (continuous, smooth, correlated velocities with real braking and
turning), rendered as a sprite. Isotropic sprites keep the symmetry group exactly R^2; oriented
sprites turn the task into an SE(2) one. Optionally the camera (drone) moves too, adding a
global, time-dependent motion on top of the object motions.

Data (free for academic non-commercial use, application form at levelxdata.com; cite, do not
redistribute). A recording XX ships as
    XX_tracks.csv         trackId, frame, xCenter, yCenter [m], heading [deg], length, width, ...
    XX_tracksMeta.csv     trackId, class, ...
    XX_recordingMeta.csv  frameRate, orthoPxToMeter, ...
    XX_background.png     orthophoto downscaled by the dataset's scale_down_factor
highD differs: tracks.csv has id, frame, x, y (top-left of the box, image axes, y down), width
(length along x), height (width along y); the class is in tracksMeta.csv.
scripts/prepare_trajectories.py converts either format to one compact .npz per recording.

At 25 fps real vehicles change velocity very little per frame; subsample (fps_out 2.5-5) so the
per-step velocity change is significant, and prefer rounD / inD over highD for the
time-dependence.

Without the data, `synthetic_tracks` generates roundabout / intersection / highway kinematics in
the same format, so the whole pipeline runs (and it is a reasonable warm-up benchmark itself).
"""
import glob
import math
import os
from pathlib import Path

import numpy as np

from ..common.schedules import make_schedule
from ..common.shifts import cumulative_displacement
from .base import GeneratedSequenceDataset

CLASS_IDS = {"car": 0, "van": 0, "truck": 1, "truck_bus": 1, "bus": 1, "trailer": 1,
             "pedestrian": 2, "bicycle": 3, "motorcycle": 4}
CLASS_INTENSITY = {0: 1.0, 1: 0.85, 2: 0.55, 3: 0.7, 4: 0.9}
SCALE_DOWN = {"ind": 12, "round": 10, "exid": 6, "unid": 2}


class Tracks:
    """
    All road users of one recording, as flat per-row arrays.
    x, y in metres with y UP; heading in radians (counter-clockwise from +x).
    """

    def __init__(self, frame, track_id, x, y, heading, length, width, cls, frame_rate,
                 name="", ortho_px_to_meter=None, background=None, scale_down=1):
        order = np.lexsort((track_id, frame))
        self.frame = np.asarray(frame, np.int64)[order]
        self.track_id = np.asarray(track_id, np.int64)[order]
        self.x = np.asarray(x, np.float64)[order]
        self.y = np.asarray(y, np.float64)[order]
        self.heading = np.asarray(heading, np.float64)[order]
        self.length = np.asarray(length, np.float64)[order]
        self.width = np.asarray(width, np.float64)[order]
        self.cls = np.asarray(cls, np.int64)[order]
        self.frame_rate = float(frame_rate)
        self.name = name
        self.ortho = ortho_px_to_meter
        self.background_path = background
        self.scale_down = scale_down
        self.f0, self.f1 = int(self.frame.min()), int(self.frame.max())
        # row range of every frame
        self._starts = np.searchsorted(self.frame, np.arange(self.f0, self.f1 + 2))

    def rows(self, f):
        i = f - self.f0
        if i < 0 or i > self.f1 - self.f0:
            return slice(0, 0)
        return slice(self._starts[i], self._starts[i + 1])

    def save(self, path):
        np.savez_compressed(path, frame=self.frame, track_id=self.track_id, x=self.x, y=self.y,
                            heading=self.heading, length=self.length, width=self.width,
                            cls=self.cls, frame_rate=self.frame_rate, name=self.name,
                            ortho=np.nan if self.ortho is None else self.ortho,
                            background=str(self.background_path or ""),
                            scale_down=self.scale_down)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        ortho = float(z["ortho"])
        return cls(z["frame"], z["track_id"], z["x"], z["y"], z["heading"], z["length"],
                   z["width"], z["cls"], float(z["frame_rate"]), name=str(z["name"]),
                   ortho_px_to_meter=None if np.isnan(ortho) else ortho,
                   background=str(z["background"]) or None, scale_down=int(z["scale_down"]))


# ============================================================================ loaders
def _read_csv(path):
    import pandas as pd
    return pd.read_csv(path)


def load_levelx(prefix, dataset="round"):
    """inD / rounD / exiD / uniD recording: prefix is '.../data/00' (without '_tracks.csv')."""
    tr = _read_csv(f"{prefix}_tracks.csv")
    meta = _read_csv(f"{prefix}_tracksMeta.csv")
    rec = _read_csv(f"{prefix}_recordingMeta.csv")
    cls_of = {int(t): CLASS_IDS.get(str(c).lower(), 0)
              for t, c in zip(meta["trackId"], meta["class"])}
    cls = np.array([cls_of.get(int(t), 0) for t in tr["trackId"]])
    bg = f"{prefix}_background.png"
    return Tracks(tr["frame"].values, tr["trackId"].values, tr["xCenter"].values,
                  tr["yCenter"].values, np.deg2rad(tr["heading"].values),
                  tr["length"].values, tr["width"].values, cls,
                  float(rec["frameRate"].values[0]), name=os.path.basename(prefix),
                  ortho_px_to_meter=float(rec["orthoPxToMeter"].values[0]),
                  background=bg if os.path.exists(bg) else None,
                  scale_down=SCALE_DOWN.get(dataset.lower(), 1))


def load_highd(prefix):
    """highD recording: prefix '.../data/01' (01_tracks.csv, 01_tracksMeta.csv, ...)."""
    tr = _read_csv(f"{prefix}_tracks.csv")
    meta = _read_csv(f"{prefix}_tracksMeta.csv")
    rec = _read_csv(f"{prefix}_recordingMeta.csv")
    cls_of = {int(t): CLASS_IDS.get(str(c).lower(), 0) for t, c in zip(meta["id"], meta["class"])}
    L = tr["width"].values            # highD: 'width' is the extent along x (vehicle length)
    Wd = tr["height"].values          # 'height' is the extent along y (vehicle width)
    xc = tr["x"].values + L / 2
    yc = -(tr["y"].values + Wd / 2)   # image y (down) -> y up
    heading = np.arctan2(-tr["yVelocity"].values, tr["xVelocity"].values)
    cls = np.array([cls_of.get(int(t), 0) for t in tr["id"]])
    return Tracks(tr["frame"].values, tr["id"].values, xc, yc, heading, L, Wd, cls,
                  float(rec["frameRate"].values[0]), name=os.path.basename(prefix))


def find_recordings(root, dataset):
    """All recording prefixes under root for a raw levelX / highD download."""
    files = sorted(glob.glob(os.path.join(root, "**", "*_tracks.csv"), recursive=True))
    return [f[:-len("_tracks.csv")] for f in files]


def load_recordings(root, dataset, ids=None):
    """Prepared .npz files (scripts/prepare_trajectories.py) or raw CSVs, optionally by id."""
    npz = sorted(glob.glob(os.path.join(root, "*.npz")))
    out = []
    if npz:
        for p in npz:
            if ids is None or Path(p).stem.split("_")[-1] in ids:
                out.append(Tracks.load(p))
        return out
    for pre in find_recordings(root, dataset):
        if ids is None or os.path.basename(pre) in ids:
            out.append(load_highd(pre) if dataset.lower() == "highd"
                       else load_levelx(pre, dataset))
    return out


# ============================================================================ synthetic stand-in
def synthetic_tracks(kind="roundabout", duration_s=120.0, frame_rate=25.0, spawn_rate=0.35,
                     seed=0, name=None):
    """
    Kinematic stand-in in the levelX format (metres, y up, heading in radians).

    roundabout   entries on 4 arms, counter-clockwise circulation (radius ~16 m) for 90-270
                 degrees, exit; decelerate into the ring, accelerate out -- the velocity DIRECTION
                 rotates continuously while on the ring.
    intersection approach, slow down (or stop), turn left / right / go straight on an arc.
    highway      lanes along x at 20-35 m/s, occasional lane changes and braking.
    Pedestrians walk slowly (1-1.6 m/s) across the scene in the first two.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / frame_rate
    n_frames = int(duration_s * frame_rate)
    rows = []
    tid = 0

    def add(track, cls, length, width, t0):
        nonlocal tid
        for k, (x, y, h) in enumerate(track):
            f = t0 + k
            if 0 <= f < n_frames:
                rows.append((f, tid, x, y, h, length, width, cls))
        tid += 1

    t = 0.0
    while t < duration_s:
        t += rng.exponential(1.0 / spawn_rate)
        f0 = int(t * frame_rate)
        is_ped = kind != "highway" and rng.random() < 0.15
        if is_ped:
            ang = rng.uniform(0, 2 * np.pi)
            p = np.array([np.cos(ang), np.sin(ang)]) * 30.0
            d = -p / np.linalg.norm(p) + rng.normal(0, 0.3, 2)
            d /= np.linalg.norm(d)
            spd = rng.uniform(1.0, 1.6)
            n = int(60 / spd * frame_rate)
            heading = np.arctan2(d[1], d[0])
            pts = []
            for k in range(n):
                heading += rng.normal(0, 0.01)
                p = p + spd * dt * np.array([np.cos(heading), np.sin(heading)])
                pts.append((p[0], p[1], heading))
            add(pts, 2, 0.6, 0.6, f0)
            continue
        cls = 1 if rng.random() < 0.1 else 0
        length, width = (rng.uniform(8, 12), 2.6) if cls == 1 else (rng.uniform(4.0, 5.2), 1.9)
        pts = []
        if kind == "roundabout":
            R = 16.0
            arm_in = rng.integers(4)
            turn = rng.choice([1, 2, 3]) * np.pi / 2
            a_in = arm_in * np.pi / 2
            v_cruise = rng.uniform(9, 13)
            v_ring = rng.uniform(5, 7.5)
            # approach along the arm towards the ring (right-hand traffic, offset 2 m)
            ax = np.array([np.cos(a_in), np.sin(a_in)])
            nrm = np.array([-ax[1], ax[0]])
            p = ax * 45.0 - nrm * 2.0
            v = v_cruise
            while np.linalg.norm(p) > R + 0.5:
                v += (v_ring - v) * 0.03
                p = p - ax * v * dt
                pts.append((p[0], p[1], np.arctan2(-ax[1], -ax[0])))
            th = np.arctan2(p[1], p[0])
            th_end = th + turn
            while th < th_end:
                th += v_ring * dt / R
                pts.append((R * np.cos(th), R * np.sin(th), th + np.pi / 2))
            aout = np.array([np.cos(th), np.sin(th)])
            p = aout * R
            v = v_ring
            for _ in range(int(4 * frame_rate)):
                v += (v_cruise - v) * 0.03
                p = p + aout * v * dt
                pts.append((p[0], p[1], np.arctan2(aout[1], aout[0])))
        elif kind == "intersection":
            arm_in = rng.integers(4)
            a_in = arm_in * np.pi / 2
            ax = np.array([np.cos(a_in), np.sin(a_in)])
            nrm = np.array([-ax[1], ax[0]])
            p = ax * 50.0 - nrm * 2.0
            v = rng.uniform(8, 12)
            stop = rng.random() < 0.4
            heading = np.arctan2(-ax[1], -ax[0])
            while np.dot(p, ax) > 8.0:
                target = 0.0 if (stop and np.dot(p, ax) < 20) else 6.0
                v += (target - v) * 0.04
                if stop and v < 0.3:
                    for _ in range(int(rng.uniform(1, 3) * frame_rate)):
                        pts.append((p[0], p[1], heading))
                    stop = False
                p = p - ax * max(v, 0.3) * dt
                pts.append((p[0], p[1], heading))
            turn = rng.choice([-1, 0, 1])            # right, straight, left
            omega = turn * (np.pi / 2) / (2.5 if turn == 1 else 1.6)
            v = max(v, 4.0)
            for _ in range(int((2.5 if turn == 1 else 1.6) * frame_rate) if turn else 0):
                heading += omega * dt
                p = p + v * dt * np.array([np.cos(heading), np.sin(heading)])
                pts.append((p[0], p[1], heading))
            for _ in range(int(5 * frame_rate)):
                v += (11 - v) * 0.03
                p = p + v * dt * np.array([np.cos(heading), np.sin(heading)])
                pts.append((p[0], p[1], heading))
        else:  # highway
            direction = rng.choice([-1, 1])
            lane = rng.integers(3)
            y = direction * (2.0 + 3.75 * lane)
            x = -200.0 * direction
            v = rng.uniform(20, 35)
            n = int(420.0 / v * frame_rate)
            y_target = y
            for k in range(n):
                if rng.random() < 0.002:
                    lane = int(np.clip(lane + rng.choice([-1, 1]), 0, 2))
                    y_target = direction * (2.0 + 3.75 * lane)
                if rng.random() < 0.003:
                    v *= rng.uniform(0.6, 0.85)
                v += (rng.uniform(24, 32) - v) * 0.01
                vy = (y_target - y) * 0.6
                y += vy * dt
                x += direction * v * dt
                pts.append((x, y, np.arctan2(vy, direction * v)))
        add(pts, cls, length, width, f0)
    if not rows:
        raise RuntimeError("no synthetic tracks generated")
    a = np.array(rows, dtype=np.float64)
    return Tracks(a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5], a[:, 6], a[:, 7],
                  frame_rate, name=name or f"synthetic_{kind}_{seed}")


# ============================================================================ rendering
def render_sprites(S, cols, rows, headings, lengths, widths, intens, sprite="box",
                   softness=0.6, supersample=1):
    """
    Render sprites on an S x S canvas at sub-pixel centres (cols, rows) [px], sizes in px.
    'gaussian' is isotropic (keeps the symmetry group R^2); 'box' is an anti-aliased oriented
    rectangle (SE(2)). Intensities are max-composited (occlusion, no brightness pile-up).
    """
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float64)
    img = np.zeros((S, S), np.float64)
    for c, r, h, L, W, a in zip(cols, rows, headings, lengths, widths, intens):
        if sprite == "gaussian":
            sig = max(0.5, 0.35 * max(L, W))
            v = a * np.exp(-((xx - c) ** 2 + (yy - r) ** 2) / (2 * sig * sig))
        else:
            ch, sh = math.cos(h), math.sin(h)
            dx, dy = xx - c, -(yy - r)                 # canvas rows point down, y up
            u = ch * dx + sh * dy
            w = -sh * dx + ch * dy
            s = max(softness, 1e-3)
            v = a / (1 + np.exp(-(0.5 * L - np.abs(u)) / s)) / (1 + np.exp(-(0.5 * W - np.abs(w)) / s))
        np.maximum(img, v, out=img)
    return img


class TrajectoryVideoDataset(GeneratedSequenceDataset):
    """
    Windows of real traffic rendered as video.

    recordings     list of Tracks
    image_size     canvas S (px); meters_per_px sets the field of view (S * mpp metres)
    fps_out        output frame rate (subsampled from the recording's)
    sprite         'box' (oriented) | 'gaussian' (isotropic)
    n_motions      how many per-object motions to return (ranked by visible frames)
    min_objects    reject windows with fewer road users in the first frame
    min_speed      reject windows whose fastest object moves slower than this (px/step)
    camera         None | a schedule name ('piecewise', 'ou', 'rotating'): the viewpoint moves
                   (a drifting drone), adding a global time-dependent motion; camera_max_speed
                   in px/step. motion[:, 0] is then the CAMERA-induced background motion and the
                   object motions follow.
    background     'none' | 'photo' (the recording's orthophoto, needs its background png)
    """

    def __init__(self, recordings, length, seq_len, image_size=64, meters_per_px=0.5,
                 fps_out=5.0, sprite="box", n_motions=2, min_objects=1, max_objects=None,
                 min_speed=0.3, camera=None, camera_max_speed=1.0, background="none",
                 noise_std=0.0, seed=0, random=True, max_tries=50):
        super().__init__(length, seed, random)
        if not recordings:
            raise ValueError("no recordings")
        self.recs = recordings
        self.seq_len = int(seq_len)
        self.S = int(image_size)
        self.mpp = float(meters_per_px)
        self.fps_out = float(fps_out)
        self.sprite = sprite
        self.n_motions = int(n_motions)
        self.min_objects = int(min_objects)
        self.max_objects = max_objects
        self.min_speed = float(min_speed)
        self.camera = camera
        self.camera_max_speed = float(camera_max_speed)
        self.background = background
        self.noise_std = float(noise_std)
        self.max_tries = int(max_tries)
        self._bg_cache = {}
        n_m = self.n_motions + (1 if camera else 0)
        self.meta = dict(name="trajectories", in_channels=1, out_channels=1,
                         channel_names=["occupancy"], has_motion=True, n_motions=n_m,
                         periodic=False, pc_window=background != "none", pc_alpha=1.0,
                         thresholds=[0.25, 0.5],
                         fss_scales=[1, 3, 7], fps_out=self.fps_out, meters_per_px=self.mpp)

    def _photo(self, rec):
        if rec.background_path is None or rec.ortho is None:
            return None
        if rec.name not in self._bg_cache:
            import matplotlib.image as mpimg
            im = mpimg.imread(rec.background_path)
            if im.ndim == 3:
                im = im[..., :3].mean(axis=-1)
            self._bg_cache[rec.name] = im.astype(np.float32)
        return self._bg_cache[rec.name]

    def _sample_photo(self, rec, cx, cy):
        from scipy.ndimage import map_coordinates
        im = self._photo(rec)
        if im is None:
            return np.zeros((self.S, self.S))
        k = rec.ortho * rec.scale_down                 # metres per background pixel
        S = self.S
        yy, xx = np.mgrid[0:S, 0:S].astype(np.float64)
        X = cx + (xx - S / 2 + 0.5) * self.mpp
        Y = cy - (yy - S / 2 + 0.5) * self.mpp
        return map_coordinates(im, [(-Y / k).ravel(), (X / k).ravel()], order=1,
                               mode="nearest").reshape(S, S)

    def generate(self, rng, index):
        S, T = self.S, self.seq_len
        for _ in range(self.max_tries):
            rec = self.recs[int(rng.integers(len(self.recs)))]
            step = max(1, int(round(rec.frame_rate / self.fps_out)))
            span = (T - 1) * step
            if rec.f1 - rec.f0 <= span:
                continue
            f_start = int(rng.integers(rec.f0, rec.f1 - span))
            r0 = rec.rows(f_start)
            if r0.stop - r0.start == 0:
                continue
            anchor = int(rng.integers(r0.start, r0.stop))
            half = 0.5 * S * self.mpp
            cx = rec.x[anchor] + rng.uniform(-0.3, 0.3) * half
            cy = rec.y[anchor] + rng.uniform(-0.3, 0.3) * half
            out = self._render(rec, f_start, step, cx, cy, rng)
            if out is not None:
                return out
        raise RuntimeError("could not find a window satisfying min_objects/min_speed; "
                           "relax them or check the recordings")

    def _render(self, rec, f_start, step, cx, cy, rng):
        S, T, mpp = self.S, self.seq_len, self.mpp
        cam = np.zeros((T, 2))
        if self.camera:
            cam = make_schedule(self.camera, T, rng, self.camera_max_speed)
        Dcam = cumulative_displacement(cam)                   # px the CONTENT moves
        frames = np.zeros((T, S, S), np.float32)
        pos = {}                                             # track -> list of (t, col, row)
        for t in range(T):
            f = f_start + t * step
            sl = rec.rows(f)
            # a camera moving by -D makes the content move by +D
            ccx = cx - Dcam[t, 0] * mpp
            ccy = cy + Dcam[t, 1] * mpp
            cols = (rec.x[sl] - ccx) / mpp + S / 2 - 0.5
            rws = -(rec.y[sl] - ccy) / mpp + S / 2 - 0.5
            L = rec.length[sl] / mpp
            W = rec.width[sl] / mpp
            margin = np.maximum(L, W)
            vis = (cols > -margin) & (cols < S - 1 + margin) & (rws > -margin) & (rws < S - 1 + margin)
            ids = rec.track_id[sl][vis]
            if t == 0:
                n0 = int(vis.sum())
                if n0 < self.min_objects or (self.max_objects and n0 > self.max_objects):
                    return None
            inten = np.array([CLASS_INTENSITY.get(int(c), 1.0) for c in rec.cls[sl][vis]])
            img = render_sprites(S, cols[vis], rws[vis], rec.heading[sl][vis], L[vis], W[vis],
                                 inten, sprite=self.sprite)
            if self.background == "photo":
                bg = self._sample_photo(rec, ccx, ccy)
                bg = (bg - bg.mean()) / (bg.std() + 1e-6) * 0.15 + 0.3
                img = np.maximum(img, bg)
            frames[t] = img
            for i, c, r in zip(ids, cols[vis], rws[vis]):
                pos.setdefault(int(i), []).append((t, c, r))
        # per-object motions: displacement t -> t+1 wherever the object is visible in both
        tracks = sorted(pos.values(), key=len, reverse=True)
        motions = []
        for tr in tracks[:self.n_motions]:
            m = np.full((T, 2), np.nan)
            d = {t: (c, r) for t, c, r in tr}
            for t in range(T - 1):
                if t in d and t + 1 in d:
                    m[t] = (d[t + 1][0] - d[t][0], d[t + 1][1] - d[t][1])
            m = _fill_nan(m)
            motions.append(m)
        if not motions:
            return None
        speed = max(np.nanmax(np.hypot(m[:, 0], m[:, 1])) for m in motions)
        if speed < self.min_speed:
            return None
        while len(motions) < self.n_motions:
            motions.append(motions[0])
        motion = np.stack(motions, axis=1)                    # (T, n, 2)
        if self.camera:
            motion = np.concatenate([cam[:, None, :], motion], axis=1)
        if self.noise_std > 0:
            frames = frames + self.noise_std * rng.standard_normal(frames.shape).astype(np.float32)
        return self.pack(frames, motion, label=0)


def _fill_nan(m):
    """Forward/backward fill NaN rows of a (T, 2) motion (objects entering/leaving)."""
    ok = np.isfinite(m[:, 0])
    if not ok.any():
        return np.zeros_like(m)
    idx = np.where(ok, np.arange(len(m)), 0)
    np.maximum.accumulate(idx, out=idx)
    m = m[idx]
    first = np.argmax(ok)
    m[:first] = m[first]
    return m
