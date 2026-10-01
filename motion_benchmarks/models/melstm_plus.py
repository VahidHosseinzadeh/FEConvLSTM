"""
MEConvLSTMPlus -- Seq2SeqMEConvLSTM with opt-in extras for real and physical data.

Subclass, not a fork: the cell, the decoder, the slot pooling and the phase-correlation modules
are the parent's own, built by the parent's __init__ in the parent's order, so a model constructed
with the default arguments has bit-identical initial weights and bit-identical outputs to
Seq2SeqMEConvLSTM (tests/test_models.py checks this). Everything below is off by default.

velocity_source -- where the per-slot transport velocity comes from
    'track'      the parent's protocol: bootstrap K peaks from (X0, X1), then each slot
                 correlates its own hidden state against the next frame. Optional `track_gate`
                 adds the three tracker fixes measured on articulating subjects: a local search
                 window around the previous velocity, a confidence gate that coasts on the
                 previous velocity, and template erosion.
    'frame_pair' K peaks from the raw consecutive frames at every step (optionally the residual
                 multi-motion estimator), slot identity kept by nearest-previous matching. Robust
                 when hidden states are poor templates (noise textures, fluids).
    'external'   velocities supplied by the caller: forward(..., velocities=motion) with the
                 dataset's ground-truth motion (B, T, N, 2). The ORACLE-velocity MEConvLSTM: an
                 upper bound that separates "transport helps" from "estimation is hard".
    'mean_flow'  the Galilean connection: the spatial mean of the horizontal velocity channels,
                 <u_h>[psi > X] = <u_h>[X] + v_t, exactly the abelian connection law. Parameter-
                 free and exactly equivariant under Galilean frame changes (NOT under the regular
                 action, where <u_h> is invariant -- use phase correlation there).

pc_channels     channels fed to phase correlation (e.g. the mid-plane temperature only);
                None = mean over all channels, the parent's behaviour.
pc_window       Hann-taper both correlation inputs -- for NON-periodic crops (real radar,
                satellite, calcium), where the implied wrap-around otherwise biases the peak
                towards zero. Routes phase correlation through common/phase_correlation.py.
decoder_input   what the decoder cell consumes at each rollout step:
                'previous' (parent) the previous frame / prediction, UNWARPED -- while the state
                           has already been warped to the NEXT frame's position. Input and state
                           are then misaligned by the step velocity v, and under a frame motion u
                           by v + u instead: the rollout is not exactly motion-equivariant (the
                           encoder is). tests/test_models.py demonstrates both facts.
                'warped'   the previous frame warped with each slot's velocity, so input and
                           state are aligned in every slot: exactly equivariant. The input is
                           then the Lagrangian-persistence guess of the next frame.
                'zeros'    autonomous rollout (trivially equivariant).
residual        'none' (parent) | 'eulerian' | 'lagrangian'. Lagrangian: prediction at lead t =
                the last context frame shifted by the cumulative slot velocity + the decoder
                output -- "Lagrangian persistence plus learned evolution". Eulerian is the
                in-place skip of Fromme et al.'s latent ConvLSTM, the right one only without
                frame motion.
forget_bias / forget_bias_long / long_fraction   see models/gates.py.

Velocity conventions: (vx, vy) px per step; the dataset's motion[:, s] is the displacement from
frame s to frame s+1. Encoder step t (consuming X_t) warps the state by the displacement
X_{t-1} -> X_t = motion[:, t-1]; decoder step t (predicting X_{T_in+t}) by motion[:, T_in-1+t].
"""
import torch
import torch.nn as nn

from .. import _repo  # noqa: F401  (puts moving_mnist/ on sys.path)
from velocity_model_based_MEConvLSTM_model import Seq2SeqMEConvLSTM  # noqa: E402

from ..common.phase_correlation import (gated_track, hann2d, match_to_slots,  # noqa: E402
                                        phase_correlate, residual_peaks)
from ..common.shifts import shift_torch  # noqa: E402
from .gates import set_forget_bias  # noqa: E402

VELOCITY_SOURCES = ("track", "frame_pair", "external", "mean_flow")
RESIDUALS = ("none", "eulerian", "lagrangian")


class MeanFlowConnection(nn.Module):
    """
    Velocity from the fluid itself: spatial mean of the horizontal velocity channels.

    ux_channels / uy_channels : indices of the channels holding u_x / u_y (e.g. one per height)
    channel_mean / channel_std: the dataset's per-channel normalisation, x = (u - mean) / std
    px_per_unit              : (sx, sy) pixels per step per unit of physical velocity
                               = snapshot interval / grid spacing
    """

    def __init__(self, ux_channels, uy_channels, channel_mean, channel_std, px_per_unit=(1.0, 1.0)):
        super().__init__()
        self.ux = [int(i) for i in ux_channels]
        self.uy = [int(i) for i in uy_channels]
        self.register_buffer("mean", torch.as_tensor(channel_mean, dtype=torch.float32).flatten())
        self.register_buffer("std", torch.as_tensor(channel_std, dtype=torch.float32).flatten())
        self.register_buffer("px", torch.as_tensor(px_per_unit, dtype=torch.float32).flatten())

    def _phys(self, x, idx):
        return x[:, idx] * self.std[idx, None, None] + self.mean[idx, None, None]

    def physical_mean(self, x):
        """x (B, C, H, W) normalised -> (B, 2) mean (u_x, u_y) in physical units."""
        ux = self._phys(x, self.ux).mean(dim=(1, 2, 3)) if self.ux else x.new_zeros(x.shape[0])
        uy = self._phys(x, self.uy).mean(dim=(1, 2, 3)) if self.uy else x.new_zeros(x.shape[0])
        return torch.stack([ux, uy], dim=-1)

    def velocity(self, x):
        """(B, C, H, W) -> (B, 2) transport in px per step."""
        return self.physical_mean(x) * self.px.to(x.dtype)

    def _add(self, x, u_mean, sign):
        x = x.clone()
        if self.ux:
            x[:, self.ux] = x[:, self.ux] + sign * u_mean[:, 0, None, None, None] / self.std[self.ux, None, None]
        if self.uy:
            x[:, self.uy] = x[:, self.uy] + sign * u_mean[:, 1, None, None, None] / self.std[self.uy, None, None]
        return x

    def gauge_fix(self, x, u_mean=None):
        """Zero-momentum gauge: subtract the mean flow (physical units) from the velocity channels."""
        return self._add(x, self.physical_mean(x) if u_mean is None else u_mean, -1.0)

    def add_mean(self, x, u_mean):
        return self._add(x, u_mean, +1.0)


class MEConvLSTMPlus(Seq2SeqMEConvLSTM):

    def __init__(self, input_channels, hidden_channels, output_channels=None, n_slots=2,
                 kernel_size=3, slot_reduce="max", decoder_layers=1, decoder_channels=None,
                 bias=True, phase_corr_kwargs=None,
                 velocity_source="track", pc_channels=None, pc_window=False,
                 frame_pair_mode="peaks", track_gate=None,
                 mean_flow=None, mean_flow_lag="previous", mean_flow_decoder="predicted",
                 gauge_fix=True,
                 residual="none", residual_slot=0, residual_shift="fourier",
                 decoder_input="previous",
                 forget_bias=None, forget_bias_long=None, long_fraction=0.5):
        super().__init__(input_channels, hidden_channels, output_channels=output_channels,
                         n_slots=n_slots, kernel_size=kernel_size, slot_reduce=slot_reduce,
                         decoder_layers=decoder_layers, decoder_channels=decoder_channels,
                         bias=bias, batch_first=True, phase_corr_kwargs=phase_corr_kwargs)
        # Velocities here can be sub-pixel (parabolic PC peaks, the oracle, mean flow), so the
        # warp must use its exact path with the 1-px circular pad, not Moving MNIST's
        # whole-pixel fast path (MEConvLSTMCell.integer_shift).
        self.cell.integer_shift = False
        if velocity_source not in VELOCITY_SOURCES:
            raise ValueError(f"velocity_source must be one of {VELOCITY_SOURCES}")
        if residual not in RESIDUALS:
            raise ValueError(f"residual must be one of {RESIDUALS}")
        if frame_pair_mode not in ("peaks", "residual"):
            raise ValueError("frame_pair_mode must be 'peaks' or 'residual'")
        if velocity_source == "mean_flow" and mean_flow is None:
            raise ValueError("velocity_source='mean_flow' needs a MeanFlowConnection (mean_flow=)")
        if mean_flow_lag not in ("previous", "current", "average"):
            raise ValueError("mean_flow_lag must be previous|current|average")
        if decoder_input not in ("previous", "warped", "zeros"):
            raise ValueError("decoder_input must be previous|warped|zeros")
        self.decoder_input = decoder_input

        self.velocity_source = velocity_source
        self.pc_channels = None if pc_channels is None else [int(c) for c in pc_channels]
        self.pc_window = bool(pc_window)
        self._win_cache = {}
        self.frame_pair_mode = frame_pair_mode
        self.track_gate = dict(track_gate) if track_gate else None
        self.mean_flow = mean_flow
        self.mean_flow_lag = mean_flow_lag
        self.mean_flow_decoder = mean_flow_decoder
        self.gauge_fix = bool(gauge_fix) and mean_flow is not None and velocity_source == "mean_flow"
        self.residual = residual
        self.residual_slot = int(residual_slot)
        self.residual_shift = residual_shift

        pc_kw = dict(phase_corr_kwargs or {})
        self._pc_alpha = float(pc_kw.get("alpha", 1.0))
        self._pc_subpixel = bool(pc_kw.get("subpixel", False))
        self._pc_radius = pc_kw.get("search_radius", None)
        self._pc_suppress = int(pc_kw.get("suppress_radius", 1)) or 1

        set_forget_bias(self.cell.conv, hidden_channels, short=forget_bias,
                        long=forget_bias_long, long_fraction=long_fraction)

    # ------------------------------------------------------------------ helpers
    def _pc_frame(self, x):
        return x if self.pc_channels is None else x[:, self.pc_channels]

    def _window(self, H, W, device):
        key = (H, W, device)
        if key not in self._win_cache:
            self._win_cache[key] = hann2d(H, W, device=device)
        return self._win_cache[key]

    def _windowed_peaks(self, a, b, k):
        v, _, _ = phase_correlate(a, b, k=k, alpha=self._pc_alpha, radius=self._pc_radius,
                                  subpixel=self._pc_subpixel, suppress=self._pc_suppress,
                                  window=self._window(a.shape[-2], a.shape[-1], a.device))
        return v

    def bootstrap_velocities(self, x0, x1):
        if self.pc_window:
            return self._windowed_peaks(self._pc_frame(x0), self._pc_frame(x1),
                                        self.n_slots).to(x0.dtype)
        return super().bootstrap_velocities(self._pc_frame(x0), self._pc_frame(x1))

    def _track(self, h, frame, v_prev):
        frame = self._pc_frame(frame)
        if self.track_gate is None and not self.pc_window:
            return super().track_velocities(h, frame)
        B, K, Ch, H, W = h.shape
        C = frame.shape[1]
        tmpl = h.mean(dim=2).reshape(B * K, H, W)
        f = frame.unsqueeze(1).expand(B, K, C, H, W).reshape(B * K, C, H, W)
        g = self.track_gate or {}
        v, _, _ = gated_track(tmpl, f, v_prev.reshape(B * K, 2).float(),
                              alpha=g.get("alpha", self._pc_alpha),
                              radius=g.get("radius", self._pc_radius),
                              min_conf=g.get("min_conf", None),
                              subpixel=g.get("subpixel", self._pc_subpixel),
                              erode_quantile=g.get("erode_quantile", None),
                              window=self._window(H, W, h.device) if self.pc_window else None)
        return v.reshape(B, K, 2).to(h.dtype)

    def _frame_pair(self, x0, x1):
        a, b = self._pc_frame(x0), self._pc_frame(x1)
        if self.frame_pair_mode == "residual" and self.n_slots > 1:
            v, _ = residual_peaks(a, b, k=self.n_slots, alpha=self._pc_alpha,
                                  radius=self._pc_radius, subpixel=self._pc_subpixel)
        elif self.pc_window:
            v = self._windowed_peaks(a, b, self.n_slots)
        else:
            with torch.no_grad():
                v, _ = self.phase_corr_bootstrap(a, b)
        return v.to(x0.dtype)

    def _external(self, velocities, s):
        if velocities is None:
            raise ValueError("velocity_source='external' needs forward(..., velocities=motion)")
        v = velocities[:, s].to(torch.float32)            # (B, N, 2)
        K, N = self.n_slots, v.shape[1]
        if N >= K:
            return v[:, :K]
        return torch.cat([v, v[:, -1:].expand(-1, K - N, -1)], dim=1)

    def _connection(self, frame):
        B = frame.shape[0]
        return self.mean_flow.velocity(frame)[:, None, :].expand(B, self.n_slots, 2)

    def _cell_input(self, x):
        return self.mean_flow.gauge_fix(x) if self.gauge_fix else x

    def _decoder_step(self, x, h, c, v):
        """One decoder cell update honouring decoder_input (see class docstring)."""
        if self.decoder_input == "previous":
            return self.cell(x, h, c, v)
        if self.decoder_input == "zeros":
            return self.cell(torch.zeros_like(x), h, c, v)
        cell = self.cell
        B, K, Ch, H, W = h.shape
        C = x.shape[1]
        h = cell.warp(h, v)
        c = cell.warp(c, v)
        xk = cell.warp(x.unsqueeze(1).expand(B, K, C, H, W).contiguous(), v)
        i, f, o, g = torch.chunk(cell.conv(torch.cat([xk.reshape(B * K, C, H, W),
                                                      h.reshape(B * K, Ch, H, W)], dim=1)), 4, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c = f * c.reshape(B * K, Ch, H, W) + i * g
        h = o * torch.tanh(c)
        return h.view(B, K, Ch, H, W), c.view(B, K, Ch, H, W)

    # ------------------------------------------------------------------ velocities
    def _encoder_velocity(self, input_seq, t, h, v_prev, velocities):
        src = self.velocity_source
        if src == "track":
            if t == 1:
                return self.bootstrap_velocities(input_seq[:, 0], input_seq[:, 1])
            return self._track(h, input_seq[:, t], v_prev)
        if src == "frame_pair":
            cand = self._frame_pair(input_seq[:, t - 1], input_seq[:, t])
            return cand if t == 1 else match_to_slots(cand, v_prev)
        if src == "external":
            return self._external(velocities, t - 1).to(input_seq.dtype)
        # mean_flow
        prev, cur = input_seq[:, t - 1], input_seq[:, t]
        if self.mean_flow_lag == "previous":
            return self._connection(prev)
        if self.mean_flow_lag == "current":
            return self._connection(cur)
        return 0.5 * (self._connection(prev) + self._connection(cur))

    def _decoder_velocity(self, t, T_in, h, v, cur, input_seq, target_seq, velocities,
                          track_decoder_velocity):
        """Returns (v, tracked)."""
        src = self.velocity_source
        have_target = target_seq is not None and track_decoder_velocity
        if src == "track":
            return (self._track(h, target_seq[:, t], v), True) if have_target else (v, False)
        if src == "frame_pair":
            if not have_target:
                return v, False
            prev_true = input_seq[:, -1] if t == 0 else target_seq[:, t - 1]
            return match_to_slots(self._frame_pair(prev_true, target_seq[:, t]), v), True
        if src == "external":
            ok = (track_decoder_velocity and velocities is not None
                  and velocities.shape[1] > T_in - 1 + t)
            return (self._external(velocities, T_in - 1 + t).to(v.dtype), True) if ok else (v, False)
        # mean_flow
        if self.mean_flow_lag == "previous":
            # connection of frame T_in-1+t: at t=0 that is the last CONTEXT frame (deployable)
            if t == 0:
                return self._connection(input_seq[:, -1]), have_target
            if have_target:
                return self._connection(target_seq[:, t - 1]), True
            if self.mean_flow_decoder == "predicted":
                return self._connection(cur), False
            return v, False
        if have_target:
            nxt = self._connection(target_seq[:, t])
            if self.mean_flow_lag == "current":
                return nxt, True
            prv = self._connection(input_seq[:, -1] if t == 0 else target_seq[:, t - 1])
            return 0.5 * (prv + nxt), True
        return v, False

    # ------------------------------------------------------------------ encoder / forward
    def encode(self, input_seq, return_states=False, velocities=None):
        B, T_in, C, H, W = input_seq.shape
        K = self.n_slots
        h, c = self.cell.init_hidden(B, K, H, W, input_seq.device, input_seq.dtype)
        estimated_velocities = []
        h_states = [] if return_states else None
        v = torch.zeros(B, K, 2, device=input_seq.device, dtype=input_seq.dtype)
        for t in range(T_in):
            if t == 0:
                v = torch.zeros(B, K, 2, device=input_seq.device, dtype=input_seq.dtype)
            else:
                v = self._encoder_velocity(input_seq, t, h, v, velocities)
            h, c = self.cell(self._cell_input(input_seq[:, t]), h, c, v)
            if t > 0:
                estimated_velocities.append(v.detach())
            if return_states:
                h_states.append(h.mean(dim=2).detach())
        return h, c, v, estimated_velocities, h_states

    def forward(self, input_seq, pred_len, target_seq=None, track_decoder_velocity=True,
                return_velocity=False, return_states=False, velocities=None):
        B, T_in, C, H, W = input_seq.shape
        h, c, v, estimated_velocities, h_states = self.encode(
            input_seq, return_states=return_states, velocities=velocities)

        x_last = input_seq[:, -1]
        prev_frame = x_last
        disp = torch.zeros(B, 2, device=input_seq.device, dtype=input_seq.dtype)
        outputs = []
        for t in range(pred_len):
            current_frame = prev_frame.detach()
            v, tracked = self._decoder_velocity(t, T_in, h, v, current_frame, input_seq,
                                                target_seq, velocities, track_decoder_velocity)
            if tracked:
                estimated_velocities.append(v.clone().detach())
            h, c = self._decoder_step(self._cell_input(current_frame), h, c, v)
            out = self.decoder(self.pool_slots(h))
            if self.residual == "eulerian":
                pred = x_last + out
            elif self.residual == "lagrangian":
                disp = disp + v[:, self.residual_slot].to(disp.dtype)
                pred = shift_torch(x_last, disp, self.residual_shift) + out
            elif self.gauge_fix:
                # the network works in the zero-momentum gauge; hand the mean flow back
                pred = self.mean_flow.add_mean(out, self.mean_flow.physical_mean(current_frame))
            else:
                pred = out
            outputs.append(pred)
            prev_frame = pred
            if return_states:
                h_states.append(h.mean(dim=2).detach())

        outputs = torch.stack(outputs, dim=1)
        result = [outputs]
        if return_velocity:
            result.append(torch.stack(estimated_velocities, dim=1))
        if return_states:
            result.append({"h": torch.stack(h_states, dim=1)})
        if len(result) == 1:
            return result[0]
        return tuple(result)
