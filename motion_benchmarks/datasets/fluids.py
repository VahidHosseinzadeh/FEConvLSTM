"""
Fluids seen from a moving frame -- Rayleigh-Benard (3D, height as channels), Swift-Hohenberg, and
The Well -- under the REGULAR or the GALILEAN action.

The experiment (claude/fluids_rbc_plan.md section 6)
----------------------------------------------------
Take a physical simulation and observe it from a platform whose velocity changes in time:

  action='regular'   translate every field horizontally by d(t) (exact Fourier shift on the
                     periodic axes), values untouched. Exactly the paper's group action; the
                     velocity must come from phase correlation (alpha = 0.5 on these fields).
  action='galilean'  translate the fields AND add d'(t) to the horizontal velocity channels --
                     what a non-inertial observer actually measures. The spatial mean flow then
                     transforms as <u_h>[psi > X] = <u_h>[X] + v_t, the abelian connection law, so
                     MEConvLSTM can take its velocity from the fluid itself
                     (--velocity_source mean_flow): parameter-free and exactly equivariant.
  action='none'      the lab frame (the Fromme et al. setting).

Galilean invariance is flow equivariance; a time-dependent boost is a motion. The motion schedule
is the same continuous-valued piecewise / OU / rotating family as everywhere else, and the
rollout protocol (freeze_after) is the Moving MNIST one, so length and velocity generalisation
carry over unchanged.

Conventions: motion[t] (px per snapshot) is the frame displacement from snapshot t to t+1; the
Galilean boost added at snapshot t is the observer velocity during [t, t+1) (right-continuous),
V = motion[t] * spacing / dt. With that convention the connection read from snapshot t is exactly
the displacement to snapshot t+1 (models: --mean_flow_lag previous).

Sources (all return physical, un-normalised windows (T, C, H, W); normalisation happens here)
------------------------------------------------------------------------------------------
ArraySource      in-memory array, e.g. a Swift-Hohenberg bank (physics/swift_hohenberg.py)
RBC3DSource      HDF5 from physics/rbc3d_dedalus.py or physics/convert_oceananigans.py:
                 /fields (runs, T, 4, nz, ny, nx), channels T,u,v,w -> height as channels
WellSource       a The Well HDF5 file (e.g. rayleigh_benard): t0_fields/*, t1_fields/velocity,
                 axis order (traj, time, x, y[, comp]); transposed to rows = y (walls),
                 cols = x (periodic). Motion is horizontal only.
"""
import os
from pathlib import Path

import numpy as np

from ..common.schedules import apply_freeze, make_schedule
from ..common.shifts import cumulative_displacement
from .base import GeneratedSequenceDataset


# ============================================================================ sources
class _Source:
    """Common metadata. Subclasses set: n_traj, n_steps, C, H, W, channel_names, ux, uy, uz,
    T_ch, periodic (rows, cols), spacing (dx cols, dy rows), dt, physics, and window()."""

    ux, uy, uz, T_ch = [], [], [], []
    periodic = (True, True)
    spacing = (1.0, 1.0)
    dt = 1.0
    physics = None
    _stats = None

    def window(self, j, s, T, stride=1):   # pragma: no cover - abstract
        raise NotImplementedError

    def stats(self, n_samples=64, seed=0):
        """Per-channel (mean, std) from a subsample of lab-frame windows (cached)."""
        if self._stats is None:
            rng = np.random.default_rng(seed)
            acc, acc2, n = 0.0, 0.0, 0
            for _ in range(n_samples):
                j = int(rng.integers(self.n_traj))
                s = int(rng.integers(self.n_steps))
                x = self.window(j, s, 1)[0].astype(np.float64)      # (C, H, W)
                acc = acc + x.mean(axis=(1, 2))
                acc2 = acc2 + (x ** 2).mean(axis=(1, 2))
                n += 1
            mean = acc / n
            std = np.sqrt(np.maximum(acc2 / n - mean ** 2, 1e-12))
            self._stats = (mean.astype(np.float32), std.astype(np.float32))
        return self._stats


class ArraySource(_Source):

    def __init__(self, array, channel_names=None, periodic=(True, True), spacing=(1.0, 1.0),
                 dt=1.0, ux=(), uy=(), uz=(), T_ch=(), physics=None):
        a = np.asarray(array, dtype=np.float32)
        if a.ndim == 4:
            a = a[:, :, None]
        self.a = a
        self.n_traj, self.n_steps, self.C, self.H, self.W = a.shape
        self.channel_names = list(channel_names or [f"c{i}" for i in range(self.C)])
        self.periodic, self.spacing, self.dt = tuple(periodic), tuple(spacing), float(dt)
        self.ux, self.uy, self.uz, self.T_ch = list(ux), list(uy), list(uz), list(T_ch)
        self.physics = physics

    def window(self, j, s, T, stride=1):
        return self.a[j, s:s + (T - 1) * stride + 1:stride]


def swift_hohenberg_source(n_traj, n_snap, N=48, n_rolls=(4.0, 8.0), noise=0.02, r=0.3, g2=0.0,
                           steps_between=5, dt=0.5, seed=0, cache_dir=None):
    """
    Swift-Hohenberg bank as an ArraySource; cached as .npy under cache_dir when given.

    Rolls (g2 = 0) have an APERTURE problem: motion along a straight roll is unobservable (and
    harmless for prediction, the pattern is invariant along it), so frame-velocity error is
    inflated on stripes; hexagons (g2 ~ 1) are fully observable (0.02 px median error at
    alpha = 0.5 with a search window). Perfectly periodic patterns are also ambiguous up to a
    lattice vector -- the reason the dataset recommends a search window (meta pc_search_radius).
    """
    from ..physics.swift_hohenberg import simulate
    key = (f"sh_N{N}_n{n_traj}_t{n_snap}_r{n_rolls}_z{noise}_r{r}_g{g2}_sb{steps_between}"
           f"_dt{dt}_s{seed}").replace(" ", "")
    path = None
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        path = Path(cache_dir) / (key.replace("(", "").replace(")", "").replace(",", "-") + ".npy")
        if path.exists():
            return ArraySource(np.load(path), channel_names=["psi"], dt=dt * steps_between)
    bank = simulate(n_traj, n_snap, N=N, n_rolls=n_rolls, noise=noise, r=r, g2=g2, seed=seed,
                    steps_between=steps_between, dt=dt)
    if path is not None:
        tmp = str(path) + f".tmp{os.getpid()}.npy"
        np.save(tmp, bank)
        os.replace(tmp, path)
    return ArraySource(bank, channel_names=["psi"], dt=dt * steps_between)


class _H5Mixin:
    """Lazy, fork-safe h5py handle (one per process)."""

    def _h5(self):
        import h5py
        pid = os.getpid()
        if getattr(self, "_fh_pid", None) != pid:
            self._fh = h5py.File(self.path, "r")
            self._fh_pid = pid
        return self._fh

    def __getstate__(self):
        d = dict(self.__dict__)
        d.pop("_fh", None)
        d.pop("_fh_pid", None)
        return d


class RBC3DSource(_H5Mixin, _Source):
    """
    3D Rayleigh-Benard with HEIGHT AS CHANNELS: (T, 4, nz, ny, nx) -> (T, 4 * nz_sel, ny, nx),
    channel order field-major (T at every selected height, then u, then v, then w). This is the
    2D problem the recurrent models already handle, and what height-dependent kernels do.

    z_levels: explicit height indices, or None with z_stride.
    """

    def __init__(self, path, runs=None, z_levels=None, z_stride=1, fields=("T", "u", "v", "w")):
        self.path = str(path)
        f = self._h5()
        d = f["fields"]
        n_runs, self.n_steps, nf, nz, ny, nx = d.shape
        names = [n.decode() if isinstance(n, bytes) else str(n)
                 for n in d.attrs.get("channel_names", f.attrs.get("channel_names",
                                                                  ["T", "u", "v", "w"]))]
        self._field_idx = [names.index(n) for n in fields]
        self.runs = list(range(n_runs)) if runs is None else list(runs)
        self.n_traj = len(self.runs)
        self.z = list(z_levels) if z_levels is not None else list(range(0, nz, z_stride))
        nzs = len(self.z)
        self.C, self.H, self.W = len(fields) * nzs, ny, nx
        self.channel_names = [f"{n}@z{k}" for n in fields for k in self.z]
        pos = {n: i for i, n in enumerate(fields)}
        rng_c = lambda n: list(range(pos[n] * nzs, (pos[n] + 1) * nzs)) if n in pos else []  # noqa: E731
        self.T_ch, self.ux, self.uy, self.uz = rng_c("T"), rng_c("u"), rng_c("v"), rng_c("w")
        a = dict(f.attrs)
        Lx = float(a.get("Lx", 2 * np.pi))
        Ly = float(a.get("Ly", 2 * np.pi))
        self.spacing = (Lx / nx, Ly / ny)
        self.dt = float(a.get("dt_snap", 0.5))
        self.periodic = (True, True)
        Ra, Pr = float(a.get("Ra", 2500.0)), float(a.get("Pr", 0.7))
        self.physics = dict(kappa=float(a.get("kappa", (Ra * Pr) ** -0.5)),
                            Lz=float(a.get("Lz", 2.0)),
                            delta_T=float(a.get("bottom_T", 1.0)) - float(a.get("top_T", 0.0)),
                            Ra=Ra, Pr=Pr)
        self.mid_T = self.T_ch[len(self.T_ch) // 2] if self.T_ch else 0

    def window(self, j, s, T, stride=1):
        d = self._h5()["fields"]
        x = d[self.runs[j], s:s + (T - 1) * stride + 1:stride]           # (T, nf, nz, ny, nx)
        x = x[:, self._field_idx][:, :, self.z]
        return x.reshape(x.shape[0], -1, self.H, self.W).astype(np.float32)


class WellSource(_H5Mixin, _Source):
    """
    One The Well HDF5 file. Scalars from t0_fields (buoyancy, pressure, ...) and the velocity
    from t1_fields/velocity. The Well stores (traj, time, x, y[, comp]); frames here are
    (rows = y, cols = x). `downsample` block-averages (ry, rx).
    """

    def __init__(self, path, scalar_fields=("buoyancy",), use_velocity=True, downsample=(1, 1),
                 x_axis_periodic=True, y_axis_periodic=False, trajectories=None):
        self.path = str(path)
        f = self._h5()
        self.scalar_fields = [s for s in scalar_fields if s in f["t0_fields"]]
        self.use_velocity = use_velocity and "velocity" in f.get("t1_fields", {})
        ref = f["t0_fields"][self.scalar_fields[0]] if self.scalar_fields else f["t1_fields"]["velocity"]
        n_traj, self.n_steps, nx, ny = ref.shape[:4]
        self.trajs = list(range(n_traj)) if trajectories is None else list(trajectories)
        self.n_traj = len(self.trajs)
        self.ry, self.rx = int(downsample[0]), int(downsample[1])
        self.H, self.W = ny // self.ry, nx // self.rx
        names = list(self.scalar_fields)
        if self.use_velocity:
            names += ["u_x", "u_y"]
        self.channel_names = names
        self.C = len(names)
        ns = len(self.scalar_fields)
        self.ux, self.uy = ([ns], [ns + 1]) if self.use_velocity else ([], [])
        self.uz = []
        self.T_ch = [self.scalar_fields.index("buoyancy")] if "buoyancy" in self.scalar_fields else []
        self.periodic = (bool(y_axis_periodic), bool(x_axis_periodic))
        dims = f.get("dimensions", {})
        x = np.asarray(dims["x"]) if "x" in dims else np.linspace(0, 4, nx, endpoint=False)
        t = np.asarray(dims["time"]) if "time" in dims else np.arange(self.n_steps) * 0.25
        Lx = float(x[-1] - x[0] + (x[1] - x[0])) if len(x) > 1 else 1.0
        self.spacing = (Lx / nx * self.rx, 1.0 / ny * self.ry)
        self.dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
        Ra = self._scalar(f, "Rayleigh")
        Pr = self._scalar(f, "Prandtl")
        self.physics = None
        if Ra and Pr and self.T_ch and self.use_velocity:
            self.physics = dict(kappa=(Ra * Pr) ** -0.5, Lz=1.0, delta_T=1.0, Ra=Ra, Pr=Pr,
                                w_from="u_y")

    @staticmethod
    def _scalar(f, name):
        if name in f.attrs:
            return float(f.attrs[name])
        sc = f.get("scalars", {})
        if name in sc:
            return float(np.asarray(sc[name]).ravel()[0])
        return None

    def _block(self, a):
        """(T, X, Y) -> (T, Y//ry, X//rx) block mean, transposed to rows = y."""
        T, X, Y = a.shape
        a = a[:, :X // self.rx * self.rx, :Y // self.ry * self.ry]
        a = a.reshape(T, X // self.rx, self.rx, Y // self.ry, self.ry).mean(axis=(2, 4))
        return np.transpose(a, (0, 2, 1))

    def window(self, j, s, T, stride=1):
        f = self._h5()
        jj = self.trajs[j]
        sl = slice(s, s + (T - 1) * stride + 1, stride)
        chans = [self._block(np.asarray(f["t0_fields"][n][jj, sl])) for n in self.scalar_fields]
        if self.use_velocity:
            v = np.asarray(f["t1_fields"]["velocity"][jj, sl])            # (T, X, Y, 2)
            chans += [self._block(v[..., 0]), self._block(v[..., 1])]
        return np.stack(chans, axis=1).astype(np.float32)


def inspect_h5(path, max_items=60):
    """Print the structure of an HDF5 file (use on a The Well file before training)."""
    import h5py
    lines = []

    def visit(name, obj):
        if len(lines) < max_items:
            shape = getattr(obj, "shape", None)
            lines.append(f"{name:50s} {shape if shape is not None else '(group)'} "
                         f"{dict(obj.attrs) if len(obj.attrs) < 6 else list(obj.attrs)}")
    with h5py.File(path, "r") as f:
        print("root attrs:", dict(f.attrs))
        f.visititems(visit)
    print("\n".join(lines))


# ============================================================================ dataset
class MovingFrameDataset(GeneratedSequenceDataset):

    def __init__(self, source, length, seq_len, action="regular", schedule="piecewise",
                 max_speed=1.5, min_speed=0.0, hold=(3, 6), smooth_prob=0.0, ou_tau=6.0,
                 freeze_after=None, time_stride=1, stats=None, seed=0, random=True,
                 pc_channels=None):
        super().__init__(length, seed, random)
        if action not in ("none", "regular", "galilean"):
            raise ValueError("action must be none|regular|galilean")
        self.src = source
        self.seq_len = int(seq_len)
        self.action = action
        self.schedule = schedule
        self.max_speed, self.min_speed = float(max_speed), float(min_speed)
        self.hold, self.smooth_prob, self.ou_tau = tuple(hold), float(smooth_prob), float(ou_tau)
        self.freeze_after = freeze_after
        self.stride = int(time_stride)
        span = (self.seq_len - 1) * self.stride + 1
        if span > source.n_steps:
            raise ValueError(f"seq_len*stride = {span} exceeds the {source.n_steps} snapshots "
                             f"per trajectory")
        self._span = span
        self.mean, self.std = stats if stats is not None else source.stats()
        self.dt_eff = source.dt * self.stride
        s = source
        H, W = s.H, s.W
        self._ky = np.fft.fftfreq(H)[:, None]
        self._kx = np.fft.fftfreq(W)[None, :]
        both = all(s.periodic)
        pc = pc_channels if pc_channels is not None else (
            [getattr(s, "mid_T", 0)] if s.T_ch else [0])
        meta = dict(name="fluid_moving_frame", in_channels=s.C, out_channels=s.C,
                    channel_names=s.channel_names, has_motion=action != "none", n_motions=1,
                    periodic=both, pc_alpha=0.5, pc_channels=pc, display_channel=pc[0],
                    pc_search_radius=int(np.ceil(max(self.max_speed, 1.0))) + 1,
                    action=action, spacing=list(s.spacing), dt=self.dt_eff)
        groups = {}
        for g, idx in (("T", s.T_ch), ("u", s.ux), ("v", s.uy), ("w", s.uz)):
            if idx:
                groups[g] = list(idx)
        if groups:
            meta["channel_groups"] = groups
        if s.ux or s.uy:
            meta["mean_flow"] = dict(ux_channels=list(s.ux), uy_channels=list(s.uy),
                                     channel_mean=self.mean.tolist(),
                                     channel_std=self.std.tolist(),
                                     px_per_unit=[self.dt_eff / s.spacing[0],
                                                  self.dt_eff / s.spacing[1]])
        if s.physics is not None:
            ph = dict(s.physics)
            ph.update(T_channels=list(s.T_ch), channel_mean=self.mean.tolist(),
                      channel_std=self.std.tolist())
            ph["w_channels"] = list(s.uz) if s.uz else (list(s.uy) if ph.get("w_from") == "u_y" else [])
            if both and s.ux and s.uy and s.uz:
                ph.update(u_channels=list(s.ux), v_channels=list(s.uy))
            meta["physics"] = ph
        self.meta = meta

    def _motion(self, rng, T):
        m = make_schedule(self.schedule, T, rng, self.max_speed, min_speed=self.min_speed,
                          hold=self.hold, smooth_prob=self.smooth_prob, tau=self.ou_tau)
        py, px = self.src.periodic
        if not py:
            m[:, 1] = 0.0
        if not px:
            m[:, 0] = 0.0
        return apply_freeze(m, self.freeze_after)

    def generate(self, rng, index):
        s = self.src
        T = self.seq_len
        j = int(rng.integers(s.n_traj))
        t0 = int(rng.integers(s.n_steps - self._span + 1))
        x = np.array(s.window(j, t0, T, self.stride), dtype=np.float64)     # (T, C, H, W)
        if self.action == "none":
            motion = np.zeros((T, 2))
        else:
            motion = self._motion(rng, T)
            D = cumulative_displacement(motion)
            X = np.fft.fft2(x, axes=(-2, -1))
            ph = np.exp(-2j * np.pi * (self._ky[None] * D[:, 1, None, None]
                                       + self._kx[None] * D[:, 0, None, None]))
            x = np.real(np.fft.ifft2(X * ph[:, None], axes=(-2, -1)))
            if self.action == "galilean":
                Vx = motion[:, 0] * s.spacing[0] / self.dt_eff
                Vy = motion[:, 1] * s.spacing[1] / self.dt_eff
                if s.ux:
                    x[:, s.ux] += Vx[:, None, None, None]
                if s.uy:
                    x[:, s.uy] += Vy[:, None, None, None]
        x = (x - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        return self.pack(x.astype(np.float32), motion[:, None, :], label=0)
