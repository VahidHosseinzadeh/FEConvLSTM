"""
Build any model in the comparison from command-line args + the dataset's meta, and run it with
one protocol.

Names
-----
lstm                    ConvLSTM (FEConvLSTMPlus, v_range = 0)
felstm                  FEConvLSTM (integer velocity lattice [-v_range, v_range]^2)
melstm                  MEConvLSTMPlus (velocity_source from --velocity_source)
melstm_oracle           MEConvLSTMPlus fed the TRUE motion (velocity_source = 'external')
lstm_stabilized         ConvLSTM run in the frame co-moving with the TRUE frame motion
persistence_eulerian    copy the last frame
persistence_lagrangian  shift the last frame by the phase-correlation velocity
persistence_oracle      shift the last frame by the true future motion

run_model() mirrors train_eval_utils._run_model: MEConvLSTM always receives target_seq when it is
available, and the decoder velocity is tracked against it during training and frozen at
evaluation unless an oracle evaluation is asked for (--eval_velocity_mode).
"""
import torch

from .. import _repo  # noqa: F401
from velocity_model_based_MEConvLSTM_model import Seq2SeqMEConvLSTM  # noqa: E402
from channel_based_FEConvLSTM_model import Seq2SeqFEConvLSTM  # noqa: E402

from .felstm_plus import FEConvLSTMPlus  # noqa: E402
from .melstm_plus import MEConvLSTMPlus, MeanFlowConnection  # noqa: E402
from .persistence import Persistence  # noqa: E402
from .stabilizer import OracleStabilized  # noqa: E402

MODEL_NAMES = ("lstm", "felstm", "melstm", "melstm_oracle", "lstm_stabilized",
               "persistence_eulerian", "persistence_lagrangian", "persistence_oracle")
TRAINABLE = ("lstm", "felstm", "melstm", "melstm_oracle", "lstm_stabilized")
NEEDS_MOTION = ("melstm_oracle", "lstm_stabilized", "persistence_oracle")


def add_model_args(p):
    g = p.add_argument_group("model")
    g.add_argument("--model", choices=MODEL_NAMES, default="melstm")
    g.add_argument("--hidden_size", type=int, default=32)
    g.add_argument("--decoder_hidden_size", type=int, default=None)
    g.add_argument("--decoder_conv_layers", type=int, default=1)
    g.add_argument("--kernel_size", type=int, default=3)
    g.add_argument("--v_range", type=int, default=2, help="felstm velocity lattice radius")
    g.add_argument("--num_vel_modes", type=int, default=1, help="melstm slots K")
    g.add_argument("--velocity_source", default="track",
                   choices=["track", "frame_pair", "external", "mean_flow"])
    g.add_argument("--pc_alpha", type=float, default=None,
                   help="phase-correlation whitening exponent (default: the dataset's "
                        "recommendation, 1.0 if none)")
    g.add_argument("--pc_subpixel", type=int, default=1, help="1 = parabolic sub-pixel peaks")
    g.add_argument("--pc_search_radius", type=int, default=None,
                   help="peak search window |d|_inf <= r (default: the dataset's speed bound "
                        "if it declares one; -1 = no window)")
    g.add_argument("--pc_suppress_radius", type=int, default=1)
    g.add_argument("--pc_channels", type=str, default=None,
                   help="comma-separated channel indices for phase correlation "
                        "(default: the dataset's recommendation, else all)")
    g.add_argument("--pc_window", type=int, default=None,
                   help="1 = Hann-taper phase-correlation inputs (default: the dataset's "
                        "recommendation -- on for non-periodic crops)")
    g.add_argument("--frame_pair_mode", default="peaks", choices=["peaks", "residual"])
    g.add_argument("--track_gate_radius", type=int, default=None)
    g.add_argument("--track_gate_min_conf", type=float, default=None)
    g.add_argument("--track_gate_erode", type=float, default=None)
    g.add_argument("--residual", default="none", choices=["none", "eulerian", "lagrangian"])
    g.add_argument("--decoder_input", default="previous", choices=["previous", "warped", "zeros"],
                   help="rollout input: 'previous' = the original models (unwarped previous "
                        "prediction; not exactly equivariant), 'warped' = aligned with each "
                        "slot / velocity copy (exactly equivariant), 'zeros' = autonomous")
    g.add_argument("--forget_bias", type=float, default=None,
                   help="forget-gate bias (default: the cells' own 1.0); see models/gates.py")
    g.add_argument("--forget_bias_long", type=float, default=None)
    g.add_argument("--long_fraction", type=float, default=0.5)
    g.add_argument("--mean_flow_lag", default="previous", choices=["previous", "current", "average"])
    g.add_argument("--mean_flow_decoder", default="predicted", choices=["predicted", "frozen"])
    g.add_argument("--no_gauge_fix", action="store_true",
                   help="mean_flow: feed raw velocity channels instead of u - <u>")
    return g


def _channels(s):
    if s is None or s == "":
        return None
    return [int(c) for c in str(s).split(",")]


def phase_corr_kwargs(args, meta):
    alpha = args.pc_alpha if args.pc_alpha is not None else meta.get("pc_alpha", 1.0)
    radius = args.pc_search_radius
    if radius is None:
        radius = meta.get("pc_search_radius")        # the data's speed bound, if it has one
    elif radius < 0:
        radius = None                                # explicit "no window"
    kw = dict(alpha=float(alpha), subpixel=bool(args.pc_subpixel), search_radius=radius,
              suppress_radius=int(args.pc_suppress_radius))
    return kw


def pc_window(args, meta):
    if getattr(args, "pc_window", None) is not None:
        return bool(args.pc_window)
    return bool(meta.get("pc_window", not meta.get("periodic", True)))


def build_model(args, meta):
    name = args.model
    cin = int(meta["in_channels"])
    cout = int(meta.get("out_channels", cin))
    hid = args.hidden_size
    dec = args.decoder_hidden_size or hid
    fb = dict(forget_bias=args.forget_bias, forget_bias_long=args.forget_bias_long,
              long_fraction=args.long_fraction)
    pc_ch = _channels(args.pc_channels)
    if pc_ch is None:
        pc_ch = meta.get("pc_channels")
    pc_kw = phase_corr_kwargs(args, meta)

    if name in ("lstm", "felstm", "lstm_stabilized"):
        m = FEConvLSTMPlus(cin, hid, output_channels=cout, kernel_size=args.kernel_size,
                           v_range=(args.v_range if name == "felstm" else 0),
                           decoder_conv_layers=args.decoder_conv_layers, decoder_channels=dec,
                           residual=("eulerian" if args.residual == "lagrangian"
                                     and name == "lstm_stabilized" else args.residual),
                           pc_alpha=pc_kw["alpha"], pc_subpixel=pc_kw["subpixel"],
                           pc_channels=pc_ch, pc_window=pc_window(args, meta),
                           pc_radius=pc_kw["search_radius"], decoder_input=args.decoder_input,
                           **fb)
        return OracleStabilized(m) if name == "lstm_stabilized" else m

    if name in ("melstm", "melstm_oracle"):
        src = "external" if name == "melstm_oracle" else args.velocity_source
        mean_flow = None
        if src == "mean_flow":
            mf = meta.get("mean_flow")
            if mf is None:
                raise ValueError("velocity_source=mean_flow needs a dataset with velocity "
                                 "channels (meta['mean_flow'])")
            mean_flow = MeanFlowConnection(**mf)
        gate = None
        if args.track_gate_radius is not None or args.track_gate_min_conf is not None \
                or args.track_gate_erode is not None:
            gate = dict(radius=args.track_gate_radius, min_conf=args.track_gate_min_conf,
                        erode_quantile=args.track_gate_erode)
        return MEConvLSTMPlus(cin, hid, output_channels=cout, n_slots=args.num_vel_modes,
                              kernel_size=args.kernel_size, decoder_layers=args.decoder_conv_layers,
                              decoder_channels=dec, phase_corr_kwargs=pc_kw,
                              velocity_source=src, pc_channels=pc_ch,
                              pc_window=pc_window(args, meta),
                              frame_pair_mode=args.frame_pair_mode, track_gate=gate,
                              mean_flow=mean_flow, mean_flow_lag=args.mean_flow_lag,
                              mean_flow_decoder=args.mean_flow_decoder,
                              gauge_fix=not args.no_gauge_fix, residual=args.residual,
                              decoder_input=args.decoder_input, **fb)

    if name.startswith("persistence_"):
        mode = "eulerian" if name == "persistence_eulerian" else "lagrangian"
        vel = "oracle" if name == "persistence_oracle" else "estimated"
        return Persistence(mode=mode, velocity=vel, alpha=pc_kw["alpha"],
                           subpixel=pc_kw["subpixel"], channels=pc_ch,
                           radius=pc_kw["search_radius"], window=pc_window(args, meta))
    raise ValueError(f"unknown model {name!r}")


def run_model(model, inp, pred_len, target=None, motion=None, track=None, want_velocity=False):
    """
    One call site for every model. Returns (prediction (B, pred_len, C, H, W), velocities or None).

    track=None -> tracked decoder velocities while training, frozen at evaluation (the repo's
    protocol); True -> oracle (tracked against the target / the true motion) at evaluation.
    """
    if isinstance(model, OracleStabilized):
        return model(inp, pred_len, motion), None
    if isinstance(model, Persistence):
        return model(inp, pred_len, motion), None
    if isinstance(model, Seq2SeqMEConvLSTM):
        if track is None:
            track = model.training
        kw = {}
        if isinstance(model, MEConvLSTMPlus):
            kw["velocities"] = motion
        res = model(inp, pred_len=pred_len, target_seq=target, track_decoder_velocity=track,
                    return_velocity=want_velocity, **kw)
        if want_velocity:
            return res[0], res[1]
        return res, None
    if isinstance(model, Seq2SeqFEConvLSTM):
        return model(inp, pred_len=pred_len), None
    return model(inp, pred_len), None


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
