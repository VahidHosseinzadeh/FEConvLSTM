"""
Command-line specs for every dataset: add_args(parser) + build(args). Importing this module
registers them all (train_motion.py does).
"""
from .registry import _floats, _ints, freeze, register, split_seeds


# ============================================================================ synthetic radar
@register("radar_synthetic")
class RadarSyntheticSpec:

    @staticmethod
    def add_args(p):
        g = p.add_argument_group("radar_synthetic")
        g.add_argument("--radar_lifetime", type=str, default="6,48",
                       help="Lagrangian lifetime in frames at the 24-px scale: a value, or "
                            "'lo,hi' for a log-uniform draw per sequence")
        g.add_argument("--radar_beta", type=float, default=2.6)
        g.add_argument("--radar_wet_fraction", type=float, default=0.35)
        g.add_argument("--radar_max_speed", type=float, default=2.5)
        g.add_argument("--radar_min_speed", type=float, default=0.0)
        g.add_argument("--radar_schedule", default="piecewise",
                       choices=["piecewise", "ou", "rotating", "constant"])
        g.add_argument("--radar_hold", type=str, default="3,6")
        g.add_argument("--radar_smooth_prob", type=float, default=0.0)
        g.add_argument("--radar_layers", type=int, default=1,
                       help=">1: independent layers with independent winds (wind shear)")
        g.add_argument("--radar_transform", default="log", choices=["log", "raw"])
        g.add_argument("--radar_test_lifetimes", type=str, default="3,6,12,24,48",
                       help="extra fixed test sets, one per lifetime (headroom curve)")
        g.add_argument("--radar_test_fast_speed", type=float, default=0.0,
                       help=">0: extra test set with winds up to this speed (velocity gen.)")

    @staticmethod
    def build(args):
        from .radar_synthetic import SyntheticRadarDataset
        sd = split_seeds(args)
        T = args.input_frames + args.pred_frames
        common = dict(image_size=args.image_size or 96, beta=args.radar_beta,
                      wet_fraction=args.radar_wet_fraction, max_speed=args.radar_max_speed,
                      min_speed=args.radar_min_speed, schedule=args.radar_schedule,
                      hold=_ints(args.radar_hold), smooth_prob=args.radar_smooth_prob,
                      n_layers=args.radar_layers, transform=args.radar_transform)
        life = _floats(args.radar_lifetime)
        fa = freeze(args)
        train = SyntheticRadarDataset(length=args.train_size, seq_len=T, lifetime=life,
                                      freeze_after=fa, seed=sd["train"], random=True, **common)
        val = SyntheticRadarDataset(length=args.val_size, seq_len=T, lifetime=life,
                                    freeze_after=fa, seed=sd["val"], random=False, **common)
        test = SyntheticRadarDataset(length=args.test_size, seq_len=T, lifetime=life,
                                     freeze_after=fa, seed=sd["test"], random=False, **common)
        gen = None
        if args.gen_pred_frames and args.gen_pred_frames > args.pred_frames:
            gen = SyntheticRadarDataset(length=args.gen_test_size,
                                        seq_len=args.input_frames + args.gen_pred_frames,
                                        lifetime=life, freeze_after=fa, seed=sd["gen"],
                                        random=False, **common)
        extra = {}
        lts = _floats(args.radar_test_lifetimes)
        if lts is not None:
            for lt in (lts if isinstance(lts, tuple) else (lts,)):
                extra[f"lifetime_{lt:g}"] = SyntheticRadarDataset(
                    length=args.extra_test_size, seq_len=T, lifetime=float(lt), freeze_after=fa,
                    seed=sd["extra"], random=False, **common)
        if args.radar_test_fast_speed > 0:
            c2 = dict(common, max_speed=args.radar_test_fast_speed,
                      min_speed=args.radar_max_speed)
            extra["fast_wind"] = SyntheticRadarDataset(
                length=args.extra_test_size, seq_len=T, lifetime=life, freeze_after=fa,
                seed=sd["extra"] + 1, random=False, **c2)
        return dict(train=train, val=val, test=test, gen_test=gen, extra_tests=extra,
                    meta=train.meta)


# ============================================================================ fluids (shared)
def _fluid_args(p):
    if "--fluid_action" in p._option_string_actions:      # shared by three specs: add once
        return
    g = p.add_argument_group("fluids (swift_hohenberg / rbc3d / the_well)")
    g.add_argument("--fluid_action", default="regular", choices=["none", "regular", "galilean"])
    g.add_argument("--fluid_schedule", default="piecewise",
                   choices=["piecewise", "ou", "rotating", "constant"])
    g.add_argument("--fluid_max_speed", type=float, default=1.5, help="px per snapshot")
    g.add_argument("--fluid_min_speed", type=float, default=0.0)
    g.add_argument("--fluid_hold", type=str, default="3,6")
    g.add_argument("--fluid_smooth_prob", type=float, default=0.0)
    g.add_argument("--fluid_time_stride", type=int, default=1,
                   help="use every n-th snapshot (raises intrinsic change per step)")
    g.add_argument("--fluid_test_fast_speed", type=float, default=0.0,
                   help=">0: extra test set with frame speeds in (max_speed, this]")
    g.add_argument("--fluid_extra_actions", type=str, default="",
                   help="comma list of extra test actions, e.g. 'none' = lab frame")


def _fluid_splits(args, sources, pc_channels=None):
    """sources: dict(train=, val=, test=) of _Source; returns the build() dict."""
    from .fluids import MovingFrameDataset
    sd = split_seeds(args)
    T = args.input_frames + args.pred_frames
    stats = sources["train"].stats()
    common = dict(action=args.fluid_action, schedule=args.fluid_schedule,
                  max_speed=args.fluid_max_speed, min_speed=args.fluid_min_speed,
                  hold=_ints(args.fluid_hold), smooth_prob=args.fluid_smooth_prob,
                  time_stride=args.fluid_time_stride, stats=stats, pc_channels=pc_channels)
    fa = freeze(args)
    train = MovingFrameDataset(sources["train"], args.train_size, T, freeze_after=fa,
                               seed=sd["train"], random=True, **common)
    val = MovingFrameDataset(sources["val"], args.val_size, T, freeze_after=fa, seed=sd["val"],
                             random=False, **common)
    test = MovingFrameDataset(sources["test"], args.test_size, T, freeze_after=fa,
                              seed=sd["test"], random=False, **common)
    gen = None
    if args.gen_pred_frames and args.gen_pred_frames > args.pred_frames:
        gen = MovingFrameDataset(sources["test"], args.gen_test_size,
                                 args.input_frames + args.gen_pred_frames, freeze_after=fa,
                                 seed=sd["gen"], random=False, **common)
    extra = {}
    if args.fluid_test_fast_speed > 0:
        c2 = dict(common, max_speed=args.fluid_test_fast_speed, min_speed=args.fluid_max_speed)
        extra["fast_frame"] = MovingFrameDataset(sources["test"], args.extra_test_size, T,
                                                 freeze_after=fa, seed=sd["extra"],
                                                 random=False, **c2)
    for act in [a for a in args.fluid_extra_actions.split(",") if a]:
        c3 = dict(common, action=act)
        extra[f"action_{act}"] = MovingFrameDataset(sources["test"], args.extra_test_size, T,
                                                    freeze_after=fa, seed=sd["extra"] + 7,
                                                    random=False, **c3)
    return dict(train=train, val=val, test=test, gen_test=gen, extra_tests=extra,
                meta=train.meta)


# ============================================================================ swift-hohenberg
@register("swift_hohenberg")
class SwiftHohenbergSpec:

    @staticmethod
    def add_args(p):
        _fluid_args(p)
        g = p.add_argument_group("swift_hohenberg")
        g.add_argument("--sh_n_rolls", type=str, default="4,8",
                       help="roll wavelengths per box (range -> drawn per trajectory)")
        g.add_argument("--sh_noise", type=float, default=0.02)
        g.add_argument("--sh_r", type=float, default=0.3)
        g.add_argument("--sh_g2", type=float, default=1.0,
                       help="~1 = hexagons (frame motion fully observable: 0.02 px phase-"
                            "correlation error); 0 = rolls, RBC-like but with an aperture "
                            "problem along the rolls (motion along a roll is unobservable)")
        g.add_argument("--sh_steps_between", type=int, default=5,
                       help="solver steps (dt 0.5) between snapshots")
        g.add_argument("--sh_snapshots", type=int, default=100)
        g.add_argument("--sh_train_traj", type=int, default=384)
        g.add_argument("--sh_eval_traj", type=int, default=64)
        g.add_argument("--sh_cache", type=str, default="./data/sh_cache")

    @staticmethod
    def build(args):
        from .fluids import swift_hohenberg_source
        N = args.image_size or 48
        nr = _floats(args.sh_n_rolls)
        kw = dict(N=N, n_rolls=nr, noise=args.sh_noise, r=args.sh_r, g2=args.sh_g2,
                  steps_between=args.sh_steps_between, cache_dir=args.sh_cache or None)
        n_snap = max(args.sh_snapshots,
                     (args.input_frames + max(args.pred_frames, args.gen_pred_frames or 0))
                     * args.fluid_time_stride + 1)
        if getattr(args, "smoke_test", False):
            n_train, n_eval = 4, 2
        else:
            n_train, n_eval = args.sh_train_traj, args.sh_eval_traj
        src = dict(train=swift_hohenberg_source(n_train, n_snap, seed=args.data_seed, **kw),
                   val=swift_hohenberg_source(n_eval, n_snap, seed=args.data_seed + 1000, **kw),
                   test=swift_hohenberg_source(n_eval, n_snap, seed=args.data_seed + 2000, **kw))
        return _fluid_splits(args, src)


# ============================================================================ rbc3d
@register("rbc3d")
class RBC3DSpec:

    @staticmethod
    def add_args(p):
        _fluid_args(p)
        g = p.add_argument_group("rbc3d (physics/rbc3d_dedalus.py or convert_oceananigans.py)")
        g.add_argument("--rbc_file", type=str, default=None,
                       help="merged HDF5 with /fields (runs, T, 4, nz, ny, nx)")
        g.add_argument("--rbc_split", type=str, default="60,20,20",
                       help="run counts for train,val,test (Fromme et al.: 60/20/20)")
        g.add_argument("--rbc_z_stride", type=int, default=4)
        g.add_argument("--rbc_z_levels", type=str, default=None)

    @staticmethod
    def build(args):
        from .fluids import RBC3DSource
        if not args.rbc_file:
            raise SystemExit("--rbc_file is required (see motion_benchmarks/README.md, RBC)")
        import h5py
        with h5py.File(args.rbc_file, "r") as f:
            n_runs = f["fields"].shape[0]
        a, b, c = _ints(args.rbc_split)
        if a + b + c > n_runs:
            raise SystemExit(f"--rbc_split asks for {a + b + c} runs, file has {n_runs}")
        zl = list(_ints(args.rbc_z_levels)) if args.rbc_z_levels else None
        mk = lambda runs: RBC3DSource(args.rbc_file, runs=runs, z_levels=zl,  # noqa: E731
                                      z_stride=args.rbc_z_stride)
        src = dict(train=mk(range(a)), val=mk(range(a, a + b)), test=mk(range(a + b, a + b + c)))
        return _fluid_splits(args, src)


# ============================================================================ the well
@register("the_well")
class TheWellSpec:

    @staticmethod
    def add_args(p):
        _fluid_args(p)
        g = p.add_argument_group("the_well (e.g. rayleigh_benard, shear_flow)")
        g.add_argument("--well_train", type=str, default=None, help="HDF5 file (train split)")
        g.add_argument("--well_val", type=str, default=None)
        g.add_argument("--well_test", type=str, default=None)
        g.add_argument("--well_fields", type=str, default="buoyancy",
                       help="scalar fields from t0_fields (comma list)")
        g.add_argument("--well_no_velocity", action="store_true")
        g.add_argument("--well_downsample", type=str, default="4,4", help="rows(y),cols(x)")
        g.add_argument("--well_y_periodic", action="store_true",
                       help="set for fully periodic sets such as shear_flow")

    @staticmethod
    def build(args):
        from .fluids import WellSource
        if not args.well_train:
            raise SystemExit("--well_train is required")
        ds = _ints(args.well_downsample)
        mk = lambda path: WellSource(path, scalar_fields=tuple(args.well_fields.split(",")),  # noqa: E731
                                     use_velocity=not args.well_no_velocity, downsample=ds,
                                     y_axis_periodic=args.well_y_periodic)
        tr = mk(args.well_train)
        src = dict(train=tr, val=mk(args.well_val) if args.well_val else tr,
                   test=mk(args.well_test) if args.well_test else tr)
        if not args.well_val or not args.well_test:
            print("[the_well] WARNING: val/test not given -- evaluating on the TRAIN file")
        return _fluid_splits(args, src)


# ============================================================================ trajectories
@register("trajectories")
class TrajectoriesSpec:

    @staticmethod
    def add_args(p):
        g = p.add_argument_group("trajectories (inD / rounD / exiD / uniD / highD, or synthetic)")
        g.add_argument("--traj_source", default="synthetic",
                       choices=["synthetic", "round", "ind", "exid", "unid", "highd"])
        g.add_argument("--traj_dir", type=str, default=None,
                       help="prepared .npz dir (scripts/prepare_trajectories.py) or raw data dir")
        g.add_argument("--traj_split", type=str, default=None,
                       help="recording ids 'train;val;test', e.g. '00-17;18-20;21-23' "
                            "(default: 75/10/15 percent by recording order)")
        g.add_argument("--traj_synthetic_kind", default="roundabout",
                       choices=["roundabout", "intersection", "highway"])
        g.add_argument("--traj_meters_per_px", type=float, default=0.5)
        g.add_argument("--traj_fps", type=float, default=5.0)
        g.add_argument("--traj_sprite", default="box", choices=["box", "gaussian"])
        g.add_argument("--traj_n_motions", type=int, default=2)
        g.add_argument("--traj_min_objects", type=int, default=1)
        g.add_argument("--traj_min_speed", type=float, default=0.3)
        g.add_argument("--traj_camera", default="none",
                       choices=["none", "piecewise", "ou", "rotating"])
        g.add_argument("--traj_camera_speed", type=float, default=1.0)
        g.add_argument("--traj_background", default="none", choices=["none", "photo"])

    @staticmethod
    def _ids(spec):
        out = []
        for part in spec.split(","):
            if "-" in part:
                a, b = part.split("-")
                w = len(a)
                out += [str(i).zfill(w) for i in range(int(a), int(b) + 1)]
            elif part:
                out.append(part)
        return out

    @staticmethod
    def build(args):
        from .trajectories import TrajectoryVideoDataset, load_recordings, synthetic_tracks
        sd = split_seeds(args)
        if args.traj_source == "synthetic":
            kind = args.traj_synthetic_kind
            dur = 60.0 if getattr(args, "smoke_test", False) else 600.0
            recs = dict(train=[synthetic_tracks(kind, dur, seed=sd["train"] + i) for i in range(3)],
                        val=[synthetic_tracks(kind, dur, seed=sd["val"] + 100)],
                        test=[synthetic_tracks(kind, dur, seed=sd["test"] + 200)])
        else:
            if not args.traj_dir:
                raise SystemExit("--traj_dir is required for real trajectories")
            if args.traj_split:
                tr, va, te = [TrajectoriesSpec._ids(s) for s in args.traj_split.split(";")]
                recs = dict(train=load_recordings(args.traj_dir, args.traj_source, tr),
                            val=load_recordings(args.traj_dir, args.traj_source, va),
                            test=load_recordings(args.traj_dir, args.traj_source, te))
            else:
                allr = load_recordings(args.traj_dir, args.traj_source)
                n = len(allr)
                a, b = max(1, int(0.75 * n)), max(1, int(0.10 * n))
                recs = dict(train=allr[:a], val=allr[a:a + b] or allr[-1:],
                            test=allr[a + b:] or allr[-1:])
        T = args.input_frames + args.pred_frames
        common = dict(image_size=args.image_size or 64, meters_per_px=args.traj_meters_per_px,
                      fps_out=args.traj_fps, sprite=args.traj_sprite,
                      n_motions=args.traj_n_motions, min_objects=args.traj_min_objects,
                      min_speed=args.traj_min_speed,
                      camera=None if args.traj_camera == "none" else args.traj_camera,
                      camera_max_speed=args.traj_camera_speed, background=args.traj_background)
        train = TrajectoryVideoDataset(recs["train"], args.train_size, T, seed=sd["train"],
                                       random=True, **common)
        val = TrajectoryVideoDataset(recs["val"], args.val_size, T, seed=sd["val"],
                                     random=False, **common)
        test = TrajectoryVideoDataset(recs["test"], args.test_size, T, seed=sd["test"],
                                      random=False, **common)
        gen = None
        if args.gen_pred_frames and args.gen_pred_frames > args.pred_frames:
            gen = TrajectoryVideoDataset(recs["test"], args.gen_test_size,
                                         args.input_frames + args.gen_pred_frames,
                                         seed=sd["gen"], random=False, **common)
        return dict(train=train, val=val, test=test, gen_test=gen, extra_tests={},
                    meta=train.meta)


# ============================================================================ archives: radar / satellite
def _archive_args(p, prefix, title, default_crop):
    g = p.add_argument_group(title)
    g.add_argument(f"--{prefix}_file", type=str, default=None,
                   help="prepared HDF5 archive with /frames (events, T, H, W)")
    g.add_argument(f"--{prefix}_split", type=str, default="0.8,0.1,0.1",
                   help="event fractions train,val,test (split by EVENT, never within one)")
    g.add_argument(f"--{prefix}_crop", type=int, default=default_crop)
    g.add_argument(f"--{prefix}_time_stride", type=int, default=1)
    g.add_argument(f"--{prefix}_min_wet", type=float, default=0.05,
                   help="reject windows with less than this fraction above the lowest threshold")
    g.add_argument(f"--{prefix}_augment", action="store_true",
                   help="D4 (rotations / flips) augmentation of training windows")


def _archive_build(args, prefix):
    from .archive import FrameArchiveDataset
    import h5py
    path = getattr(args, f"{prefix}_file")
    if not path:
        raise SystemExit(f"--{prefix}_file is required (see motion_benchmarks/README.md)")
    with h5py.File(path, "r") as f:
        n = f["frames"].shape[0]
    fr = [float(v) for v in getattr(args, f"{prefix}_split").split(",")]
    a, b = int(round(fr[0] * n)), int(round(fr[1] * n))
    ev = dict(train=list(range(0, a)), val=list(range(a, a + b)), test=list(range(a + b, n)))
    for k in ev:
        if not ev[k]:
            ev[k] = [n - 1]
            print(f"[{prefix}] WARNING: empty {k} split, reusing the last event")
    sd = split_seeds(args)
    T = args.input_frames + args.pred_frames
    kw = dict(crop=getattr(args, f"{prefix}_crop"),
              time_stride=getattr(args, f"{prefix}_time_stride"),
              min_wet_fraction=getattr(args, f"{prefix}_min_wet"))
    train = FrameArchiveDataset(path, args.train_size, T, events=ev["train"], seed=sd["train"],
                                random=True, augment=getattr(args, f"{prefix}_augment"), **kw)
    val = FrameArchiveDataset(path, args.val_size, T, events=ev["val"], seed=sd["val"],
                              random=False, **kw)
    test = FrameArchiveDataset(path, args.test_size, T, events=ev["test"], seed=sd["test"],
                               random=False, **kw)
    gen = None
    if args.gen_pred_frames and args.gen_pred_frames > args.pred_frames:
        gen = FrameArchiveDataset(path, args.gen_test_size, args.input_frames + args.gen_pred_frames,
                                  events=ev["test"], seed=sd["gen"], random=False, **kw)
    return dict(train=train, val=val, test=test, gen_test=gen, extra_tests={}, meta=train.meta)


@register("radar_real")
class RadarRealSpec:

    @staticmethod
    def add_args(p):
        _archive_args(p, "radar", "radar_real (scripts/prepare_radar.py: SEVIR, pySTEPS, arrays)", 96)

    @staticmethod
    def build(args):
        return _archive_build(args, "radar")


@register("satellite")
class SatelliteSpec:

    @staticmethod
    def add_args(p):
        _archive_args(p, "sat", "satellite (scripts/download_goes.py: GOES ABI)", 96)

    @staticmethod
    def build(args):
        return _archive_build(args, "sat")


# ============================================================================ calcium imaging
@register("calcium")
class CalciumSpec:

    @staticmethod
    def add_args(p):
        g = p.add_argument_group("calcium (two-photon imaging + injected brain motion)")
        g.add_argument("--ca_file", type=str, default=None,
                       help="archive from scripts/prepare_calcium.py; omit for synthetic movies")
        g.add_argument("--ca_split", type=str, default="0.7,0.15,0.15", help="movie fractions")
        g.add_argument("--ca_time_stride", type=int, default=3)
        g.add_argument("--ca_motion", default="physiological", choices=["physiological", "real"],
                       help="'real' re-applies the archive's registration traces")
        g.add_argument("--ca_motion_scale", type=float, default=1.0)
        g.add_argument("--ca_test_motion_scale", type=float, default=0.0,
                       help=">0: extra test set with motion scaled by this (velocity gen.)")
        g.add_argument("--ca_synthetic_movies", type=int, default=8)

    @staticmethod
    def build(args):
        from .calcium import CalciumMotionDataset, load_calcium_archive, synthetic_calcium_movie
        sd = split_seeds(args)
        T = args.input_frames + args.pred_frames
        S = args.image_size or 64
        if args.ca_file:
            movies, traces, fps = load_calcium_archive(args.ca_file)
            n = len(movies)
            fr = [float(v) for v in args.ca_split.split(",")]
            a, b = max(1, int(round(fr[0] * n))), max(1, int(round(fr[1] * n)))
            idx = dict(train=list(range(a)), val=list(range(a, min(n, a + b))) or [n - 1],
                       test=list(range(min(n, a + b), n)) or [n - 1])
            pick = lambda ids: [movies[i] for i in ids]  # noqa: E731
            mv = {k: pick(v) for k, v in idx.items()}
            tr = traces if args.ca_motion == "real" else None
        else:
            n_mov = 2 if getattr(args, "smoke_test", False) else args.ca_synthetic_movies
            Tm = 400 if getattr(args, "smoke_test", False) else 1500
            gen = lambda s0, k: [synthetic_calcium_movie(T=Tm, H=S + 48, W=S + 48, seed=s0 + j)  # noqa: E731
                                 for j in range(k)]
            mv = dict(train=gen(sd["train"] * 100, n_mov), val=gen(sd["val"] * 100, 1),
                      test=gen(sd["test"] * 100, 2))
            fps, tr = 30.0, None
        kw = dict(image_size=S, fps=fps, time_stride=args.ca_time_stride, traces=tr,
                  motion_scale=args.ca_motion_scale)
        train = CalciumMotionDataset(mv["train"], args.train_size, T, seed=sd["train"],
                                     random=True, **kw)
        val = CalciumMotionDataset(mv["val"], args.val_size, T, seed=sd["val"], random=False, **kw)
        test = CalciumMotionDataset(mv["test"], args.test_size, T, seed=sd["test"], random=False,
                                    **kw)
        gen_t = None
        if args.gen_pred_frames and args.gen_pred_frames > args.pred_frames:
            gen_t = CalciumMotionDataset(mv["test"], args.gen_test_size,
                                         args.input_frames + args.gen_pred_frames,
                                         seed=sd["gen"], random=False, **kw)
        extra = {}
        if args.ca_test_motion_scale > 0:
            kw2 = dict(kw, motion_scale=args.ca_test_motion_scale)
            extra["motion_x%g" % args.ca_test_motion_scale] = CalciumMotionDataset(
                mv["test"], args.extra_test_size, T, seed=sd["extra"], random=False, **kw2)
        return dict(train=train, val=val, test=test, gen_test=gen_t, extra_tests=extra,
                    meta=train.meta)
