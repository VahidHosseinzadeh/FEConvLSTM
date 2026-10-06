"""
KTH clips under Keller's FERNN protocol, with an injected camera motion.

What is Keller's, exactly (akandykeller/FERNN, kth/train_kth_classification.py)
------------------------------------------------------------------------------
* Frames: his cache `kth_frames.h5` -- every frame of the 599 videos, grayscale, resized from
  160x120 STRAIGHT to 32x32 with cv2 INTER_AREA (the paper says "x2 in space"; the code squashes
  the 4:3 frame). The cache already exists on the cluster; this module only reads it.
* Clips: the 00sequences.txt subsequences with at least seq_len * step frames, in file order. The
  annotation is 1-based and his code indexes 0-based with it; kept as he does. A clip is seq_len
  frames at `step` (16 at step 2 = 1.28 s), start uniform in [s, e - seq_len*step + 1].
* Split by person. 'keller' = his code: persons 1-16 / 17-20 / 21-25 -> 1517 / 380 / 473 clips.
  'official' = Schuldt et al. 2004, as written in 00sequences.txt: 11-18 / 19-21,23-25,1,4 /
  22,2,3,5-10 -> 754 / 758 / 858 clips. (His paper says "traditional"; his code is 'keller'.)
* Camera: circular whole-pixel translation of each 32x32 frame, applied before the flip. One
  velocity per clip, never re-sampled ("a more realistic restricted dataset").
* Train augmentation: random window + horizontal flip (p = 0.5). Pixel values / 255.
* Labels: walking, jogging, running, boxing, handwaving, handclapping = 0..5 (his order).

Where this differs from his code, on purpose
--------------------------------------------
* Val and test are FIXED benchmarks. His val/test draw a fresh random window every epoch, so the
  best-val epoch is partly selected on window noise. Here every eval item's window and camera
  trajectory are drawn once, at construction, from (seed, item) -- identical across epochs, runs
  and DataLoader workers. All test sets of one split share the same windows, so test conditions
  are PAIRED: they differ only in the camera.
* Camera trajectories can be any CameraMotion law (camera_motion.py), not only his constant one.
  His constant draws are reproduced exactly (keller_draws).

Each item is (clip (T, 1, 32, 32) float32, label, motion (T, 1, 2) float32, person_v (T, 2)):
motion[t] is the camera step from frame t to t+1 (the repo's convention; the last row is the
unused step past the clip). person_v[t] is a PROXY for the person's apparent velocity over the
same step -- the step of the foreground centroid against the video's median background, plus the
camera's step -- NaN where the person is not reliably segmented. It is a diagnostic only (does a
MEConvLSTM slot follow the person?), never an input. It is meaningless in scenario d2 (zoom).
"""
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .camera_motion import CameraMotion, displacements, keller_constant_velocities

KTH_ACTIONS = ("walking", "jogging", "running", "boxing", "handwaving", "handclapping")
SPLITS = {
    "keller": {"train": tuple(range(1, 17)), "val": tuple(range(17, 21)),
               "test": tuple(range(21, 26))},
    "official": {"train": (11, 12, 13, 14, 15, 16, 17, 18),
                 "val": (19, 20, 21, 23, 24, 25, 1, 4),
                 "test": (22, 2, 3, 5, 6, 7, 8, 9, 10)},
}
_LINE_RE = re.compile(r"^(?P<fname>\S+)\s+frames\s+(?P<ranges>[\d,\s\-]+)$")


def person_id(video):
    return int(video.split("_")[0][6:8])


def action_of(video):
    return video.split("_")[1]


def scenario_of(video):
    return video.split("_")[2]


class KTHStore:
    """
    Everything shared by the splits, loaded once: the frames (uint8, in memory, ~155 MB), the
    annotated subsequences, and the per-frame person proxy.

    The person proxy, per video: background = the temporal median of all its frames (the camera
    is static in KTH, except the zoom of scenario d2); foreground = |frame - background| > fg_thr;
    centroid of the foreground where it covers at least min_area pixels, NaN elsewhere. At 32x32
    and step 2 this measures, per model step, ~1 px for walkers, ~1.8 for joggers and 2-2.7 for
    runners (horizontal), against ~0.2 px for the in-place actions.
    """

    def __init__(self, root, fg_thr=0.08, min_area=8):
        import h5py
        root = Path(root)
        h5 = root / "kth_frames.h5"
        txt = root / "00sequences.txt"
        if not h5.exists() or not txt.exists():
            raise FileNotFoundError(
                f"KTH not found under {root}: need kth_frames.h5 (Keller's 32x32 cache) and "
                f"00sequences.txt. On the cluster both are in ~/FERNN-master new start/data/kth/.")
        with h5py.File(h5, "r") as f:
            self.frames = {k: f[k][:] for k in f.keys()}
        self.height, self.width = next(iter(self.frames.values())).shape[1:]
        self.annotations = self._parse(txt)
        self.fg_thr, self.min_area = fg_thr, min_area
        self.centroids = {k: self._track(v) for k, v in self.frames.items()}

    def _parse(self, txt):
        """(video, s, e) for every annotated range, in file order (Keller's parse)."""
        out = []
        for line in open(txt):
            m = _LINE_RE.match(line.strip())
            if not m:
                continue
            video = m.group("fname")
            if video not in self.frames:
                continue
            n = len(self.frames[video])
            for rng in m.group("ranges").split(","):
                rng = rng.strip()
                if not rng:
                    continue
                s, e = (map(int, rng.split("-")) if "-" in rng else (int(rng), int(rng)))
                out.append((video, s, min(e, n - 1)))
        return out

    def _track(self, frames):
        x = frames.astype(np.float32) / 255.0
        fg = np.abs(x - np.median(x, axis=0)) > self.fg_thr
        area = fg.sum(axis=(1, 2)).astype(np.float32)
        yy, xx = np.mgrid[0:x.shape[1], 0:x.shape[2]]
        cx = (fg * xx).sum(axis=(1, 2)) / np.maximum(area, 1)
        cy = (fg * yy).sum(axis=(1, 2)) / np.maximum(area, 1)
        c = np.stack([cx, cy], axis=-1).astype(np.float32)
        c[area < self.min_area] = np.nan
        return c

    def clips(self, split, scheme, min_frames):
        persons = set(SPLITS[scheme][split])
        return [(v, s, e) for v, s, e in self.annotations
                if e - s + 1 >= min_frames and person_id(v) in persons]


class KTHClips(Dataset):
    """
    One split under one camera law.

    train=True : a fresh random window and a 50% horizontal flip on every access (Keller's
                 augmentation); the camera trajectory is fixed per clip unless resample_camera.
    train=False: eval_windows fixed windows per clip, no flip, fixed trajectories.

    seed fixes the eval windows (from seed and the split only, so every camera law sees the same
    windows) and the trajectories (from seed, the split and the camera law). Keller's constant
    draws ignore the seed, as his code does: they depend on the clip order alone.
    """

    def __init__(self, store, split="train", scheme="keller", camera=None, seq_len=16, step=2,
                 train=False, seed=0, eval_windows=1, resample_camera=False):
        self.store = store
        self.split, self.scheme = split, scheme
        self.seq_len, self.step = seq_len, step
        self.train = train
        self.camera = camera or CameraMotion("none", seq_len=seq_len)
        if self.camera.seq_len != seq_len:
            raise ValueError("camera.seq_len must equal seq_len")
        self.resample_camera = bool(resample_camera and train)
        self.clip_list = store.clips(split, scheme, seq_len * step)
        self.labels = [KTH_ACTIONS.index(action_of(v)) for v, _, _ in self.clip_list]
        n_clips = len(self.clip_list)
        self.windows = 1 if train else int(eval_windows)
        n_items = n_clips * self.windows
        split_salt = {"train": 1, "val": 2, "test": 3}[split]

        # Fixed eval windows: a function of (seed, split) only -> paired test conditions.
        self.starts = None
        if not train:
            rng = np.random.RandomState(seed * 1009 + split_salt)
            self.starts = np.array([rng.randint(s, e - seq_len * step + 2)
                                    for v, s, e in self.clip_list
                                    for _ in range(self.windows)], dtype=np.int64)

        # Fixed camera trajectories, one per clip (shared by that clip's windows).
        if self.camera.uses_keller_draws:
            per_clip = keller_constant_velocities(n_clips, self.camera.v_range)
            traj = np.repeat(per_clip[:, None, :], seq_len, axis=1)
        else:
            rng = np.random.RandomState(seed * 7919 + split_salt * 101 + 17)
            traj = np.stack([self.camera.draw(rng) for _ in range(n_clips)]) if n_clips else \
                np.zeros((0, seq_len, 2), dtype=np.int64)
        self.trajectories = np.repeat(traj, self.windows, axis=0) if self.windows > 1 else traj
        assert len(self.trajectories) == n_items

    def __len__(self):
        return len(self.clip_list) * self.windows

    def clip_index(self, i):
        return i // self.windows

    def video(self, i):
        return self.clip_list[self.clip_index(i)][0]

    def __getitem__(self, i):
        video, s, e = self.clip_list[self.clip_index(i)]
        T, step = self.seq_len, self.step
        if self.train:
            start = np.random.randint(s, e - T * step + 2)
        else:
            start = int(self.starts[i])
        idx = start + step * np.arange(T)
        frames = self.store.frames[video][idx].astype(np.float32) / 255.0      # (T, H, W)

        v = self.camera.draw(np.random) if self.resample_camera else self.trajectories[i]
        d = displacements(v)
        clip = np.stack([np.roll(frames[t], (int(d[t, 1]), int(d[t, 0])), axis=(0, 1))
                         for t in range(T)])

        c = self.store.centroids[video][idx]                                    # (T, 2)
        person = np.full((T, 2), np.nan, dtype=np.float32)
        person[:-1] = c[1:] - c[:-1] + v[:-1]
        motion = v.astype(np.float32).copy()

        if self.train and np.random.rand() < 0.5:
            # flip(roll(x, d)) == roll(flip(x), -d) exactly on the torus, so negating vx is
            # the whole story for the velocities.
            clip = clip[:, :, ::-1]
            motion[:, 0] *= -1
            person[:, 0] *= -1

        return (torch.from_numpy(np.ascontiguousarray(clip))[:, None],
                self.labels[self.clip_index(i)],
                torch.from_numpy(motion)[:, None, :],
                torch.from_numpy(person))
