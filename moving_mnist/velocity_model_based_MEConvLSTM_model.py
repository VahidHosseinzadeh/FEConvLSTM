import random
from itertools import permutations
import torch
import torch.nn as nn
import torch.nn.functional as F

from velocity_predictor_model import PhaseCorrelation


class MEConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim,
            kernel_size, padding=padding,
            padding_mode="circular", bias=bias
        )

        if bias:
            nn.init.constant_(self.conv.bias[hidden_dim:2 * hidden_dim], 1.0)

        # (H, W, device, dtype) -> (yy, xx) base pixel-index grid. Depends only
        # on shape/device/dtype, not on the batch or velocity -- identical on
        # every warp() call within a run, so build it once instead of every
        # timestep of every forward pass. Plain dict (not a buffer): a stale
        # entry for an old device just goes unused after .to(device), doesn't
        # need saving/loading with the model.
        self._meshgrid_cache = {}

    def _get_base_grid(self, H, W, device, dtype):
        key = (H, W, device, dtype)
        cached = self._meshgrid_cache.get(key)
        if cached is None:
            cached = torch.meshgrid(
                torch.arange(H, device=device, dtype=dtype),
                torch.arange(W, device=device, dtype=dtype),
                indexing="ij"
            )
            self._meshgrid_cache[key] = cached
        return cached

    def warp(self, x, u):
        """
        x : (B, K, C, H, W)
        u : (B, K, 2)   [vx, vy] in pixel units
        """
        B, K, C, H, W = x.shape
        x = x.reshape(B * K, C, H, W)
        u = u.reshape(B * K, 2)

        dx = u[:, 0, None, None]
        dy = u[:, 1, None, None]

        yy, xx = self._get_base_grid(H, W, x.device, x.dtype)
        yy = yy.unsqueeze(0).expand(B * K, -1, -1)
        xx = xx.unsqueeze(0).expand(B * K, -1, -1)

        # remainder() puts the source coordinate in [0, H), but align_corners
        # normalisation only reaches pixel H-1 at +1. A fractional residual in
        # (H-1, H) -- the band straddling the wrap -- would normalise above 1
        # and get clamped to the last row instead of interpolating against row
        # 0. Sampling from a 1-px circular pad makes that band a real interior
        # interpolation: source p in [0, H) sits at padded coordinate p+1,
        # normalised over the padded extent H+2 (align_corners -> divide by
        # H+1). Integer velocities are unaffected (they land on grid points
        # either way); this is what makes sub-pixel velocities safe.
        x = F.pad(x, (1, 1, 1, 1), mode="circular")

        yy = 2 * (torch.remainder(yy - dy, H) + 1) / (H + 1) - 1
        xx = 2 * (torch.remainder(xx - dx, W) + 1) / (W + 1) - 1

        grid = torch.stack([xx, yy], dim=-1)
        x = F.grid_sample(x, grid, mode="bilinear",
                          padding_mode="border", align_corners=True)
        return x.view(B, K, C, H, W)

    def forward(self, x, h, c, u):
        B, K, Ch, H, W = h.shape

        h = self.warp(h, u)
        c = self.warp(c, u)

        x_exp = x.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(B * K, -1, H, W)
        h     = h.reshape(B * K, Ch, H, W)
        c     = c.reshape(B * K, Ch, H, W)

        i, f, o, g = torch.chunk(
            self.conv(torch.cat([x_exp, h], dim=1)), 4, dim=1
        )
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)

        c = f * c + i * g
        h = o * torch.tanh(c)

        return h.view(B, K, Ch, H, W), c.view(B, K, Ch, H, W)

    def init_hidden(self, B, K, H, W, device, dtype):
        z = torch.zeros(B, K, self.hidden_dim, H, W, device=device, dtype=dtype)
        return z, torch.zeros_like(z)


class Seq2SeqMEConvLSTM(nn.Module):
    """
    Encoder-decoder video predictor with K independently tracked velocity slots.

    Velocity vs cell-input: two different frames, two different roles
    -----------------------------------------------------------------
    These must be kept separate or the decoder produces v≈0 at t=0:

        velocity frame  : "where did h move TO?" → needs the NEXT frame
        cell input frame: "what new observation do I update h with?"
                          → always the previous frame / own prediction

    If you use current_frame for both (as in a naive implementation),
    then at decoder t=0:
        current_frame = input_seq[:, -1]   (last encoder frame)
        h was just built from input_seq[:, -1] in the encoder
        → track(h, input_seq[:, -1]) ≈ 0   (template vs itself)
    This is the same zero-velocity bug as encoder t=0.

    Decoder protocol (training, target_seq available)
    --------------------------------------------------
    velocity always comes from track(h, target_seq[:, t]):
        - slot k asks "where in target_seq[:, t] did my content go?"
        - no assignment problem: each slot queries its own h^k
        - target_seq[:, t] is the true next frame → accurate velocity

    cell input is always the model's own previous prediction
    (input_seq[:, -1] at t=0) — no teacher forcing.

    Decoder protocol (inference, target_seq is None)
    -------------------------------------------------
    No next frame exists. Tracking against own predictions reintroduces
    the circular dependency (corrupted h vs corrupted prediction).
    Correct choice: freeze v_last (last encoder velocity).
    For constant-velocity data (Moving MNIST) this is exact.
    For time-varying velocities, no better option exists without GT.
    """

    def __init__(self,
                 input_channels,
                 hidden_channels,
                 output_channels=None,
                 n_slots=2,
                 kernel_size=3,
                 slot_reduce='max',
                 decoder_layers=1,
                 decoder_channels=None,
                 bias=True,
                 batch_first=True,
                 phase_corr_kwargs=None,
                 track_mode="h",
                 track_prior_weight=0.5,
                 n_proposals=None,
                 track_window=2,
                 detach_feedback=True,
                 decoder_act="relu",
                 decoder_skip=False):
        """
        track_mode -- how a slot's velocity is read at every tracking step (encoder
            t >= 2, and decoder steps in tracked mode):
              "h"       : peak of PC(h_k, x_t) over the whole plane (the original).
              "propose" : the raw frame pair proposes, the slot chooses. The top
                          n_proposals peaks of PC(x_{t-1}, x_t) are the candidates;
                          each slot independently takes the candidate its own
                          surface PC(h_k, x_t) scores highest. No slot-to-candidate
                          assignment step, and no shift the frames do not support.
              "prior"   : the raw pair as a soft prior. Each slot takes the peak of
                          PC(h_k, x_t)/max + track_prior_weight * PC(x_{t-1}, x_t)/max.
              "window"  : "h" restricted to |vx|, |vy| <= track_window (the data's
                          velocity range; FELSTM's lattice has the same range).
              "tie"     : "h", but the slots must end on distinct velocities: slots
                          in order of their peak score each take their best velocity
                          not already taken (only bites when slots coincide).
              "propose_distinct" : "propose", but the slots take DISTINCT
                          candidates -- the injective slot->candidate map with the
                          highest summed slot score (K = 2: a 2x2 choice).
            All use the same normalized phase correlation. The bootstrap at t = 1 is
            unchanged.
        n_proposals -- "propose" / "propose_distinct"; defaults to n_slots.
        track_window -- "window" only.
        detach_feedback -- decoder feeds its prediction back as the next input;
            True cuts the gradient through that path (the original), False keeps it.
        decoder_act -- "relu" (original) or "leaky" (LeakyReLU 0.01, cannot die).
        decoder_skip -- add a 1x1 linear read-out of the pooled state to the
            decoder output (33 parameters at hidden 32), a path that bypasses the
            nonlinearity so gradients always reach h.

        frozen_prob (attribute, default 0 = off; set by the training loop, used in
            training only, tracked decoding): each sequence decodes with the encoder's
            final velocity frozen -- exactly the inference protocol -- with this
            probability, and with tracked velocities otherwise (scheduled sampling of
            the decoder velocity mode; the tracked-only protocol never trains the
            decoder on a constant-velocity rollout).
        x_track_until (attribute, default 0 = off; set by the training loop, used in
            training only): encoder steps 2 <= t < x_track_until take their velocity
            from the raw pair PC(x_{t-1}, x_t) (top n_slots peaks), assigned to slots
            by continuity with each slot's previous velocity, instead of from h.
            A curriculum lowers it to 2 (= pure h-tracking); evaluation is always h.
        x_track_p (attribute, default None = off; set by the training loop, used in
            training only): the stochastic handover. A sequence of per-step
            probabilities indexed by encoder step t; at every step t >= 2 each sequence
            independently takes the raw-pair velocity (continuity-assigned, as for
            x_track_until) with probability x_track_p[t], and its own h-tracked
            velocity otherwise. All ones is x-tracking everywhere, all zeros is pure
            h-tracking; evaluation is always h.
        """
        super().__init__()
        if track_mode not in ("h", "propose", "prior", "window", "tie", "propose_distinct"):
            raise ValueError(f"unknown track_mode {track_mode!r}")
        if decoder_act not in ("relu", "leaky"):
            raise ValueError(f"decoder_act {decoder_act!r}: expected 'relu' or 'leaky'")
        self.track_mode         = track_mode
        self.track_prior_weight = track_prior_weight
        self.n_proposals        = n_proposals or n_slots
        self.track_window       = track_window
        self.detach_feedback    = detach_feedback
        self.x_track_until      = 0
        self.x_track_p          = None
        self.frozen_prob        = 0.0

        self.batch_first     = batch_first
        self.n_slots         = n_slots
        self.hidden_channels = hidden_channels
        self.slot_reduce     = slot_reduce
        output_channels      = output_channels or input_channels
        # Decoder width is independent of the recurrent width: the cell's
        # hidden_channels is carried per slot across every timestep (and
        # kept for BPTT), the decoder runs once per predicted frame on the
        # already slot-pooled (B, hidden, H, W) map. None keeps them equal
        # (previous behavior, and what existing checkpoints were built at).
        decoder_channels     = decoder_channels or hidden_channels
        self.decoder_channels = decoder_channels

        pc_kw = phase_corr_kwargs or {}

        self.phase_corr_bootstrap = PhaseCorrelation(n_modes=n_slots, **pc_kw)
        self.phase_corr_track     = PhaseCorrelation(n_modes=1,       **pc_kw)

        self.cell = MEConvLSTMCell(input_channels, hidden_channels,
                                   kernel_size, bias)

        layers = []
        in_ch = hidden_channels
        for _ in range(decoder_layers):
            layers += [nn.Conv2d(in_ch, decoder_channels,
                                 3, padding=1, padding_mode='circular', bias=True),
                       nn.LeakyReLU(0.01) if decoder_act == "leaky" else nn.ReLU()]
            in_ch = decoder_channels
        layers += [nn.Conv2d(in_ch, output_channels,
                             3, padding=1, padding_mode='circular', bias=True)]
        self.decoder = nn.Sequential(*layers)
        # Created after everything else, so decoder_skip=False keeps the original
        # parameters, initialisation and RNG stream exactly.
        self.decoder_skip = (nn.Conv2d(hidden_channels, output_channels, 1, bias=True)
                             if decoder_skip else None)

    def decode(self, pooled):
        """Decoder output for the slot-pooled state (plus the linear skip, if on)."""
        pred = self.decoder(pooled)
        if self.decoder_skip is not None:
            pred = pred + self.decoder_skip(pooled)
        return pred

    # ------------------------------------------------------------------
    # Velocity helpers
    # ------------------------------------------------------------------

    def bootstrap_velocities(self, x0, x1):
        """K peaks from raw frame pair. Called once: encoder t=1."""
        v, _ = self.phase_corr_bootstrap(x0, x1)
        return v   # (B, K, 2)

    def _pc_surface(self, a, b):
        """
        Normalized phase-correlation surface of a against b, (..., H, W): the map
        PhaseCorrelation takes its peaks from (pad_factor 1, periodic). Entry (y, x)
        scores velocity (-x', -y'), x' and y' wrapped to [-W/2, W/2] -- see
        _index_to_velocity.
        """
        H, W = a.shape[-2:]
        R = torch.fft.rfft2(a) * torch.conj(torch.fft.rfft2(b))
        R = R / (R.abs() + self.phase_corr_track.eps)
        return torch.fft.irfft2(R, s=(H, W))

    @staticmethod
    def _index_to_velocity(idx, H, W, dtype):
        """Flat surface index -> (vx, vy), exactly as PhaseCorrelation converts it."""
        y = torch.div(idx, W, rounding_mode="floor").to(dtype)
        x = (idx % W).to(dtype)
        x = torch.where(x > W / 2, x - W, x)
        y = torch.where(y > H / 2, y - H, y)
        return torch.stack((-x, -y), dim=-1)

    def track_velocities(self, h, frame, prev_frame=None):
        """
        Per-slot self-tracking: correlate each slot's h against frame.
        All B*K pairs in one batched call.

        h          : (B, K, Ch, H, W)
        frame      : (B, C,  H,  W)
        prev_frame : (B, C,  H,  W), the frame before `frame`. Used by the
                     "propose" / "prior" track modes; ignored by "h".
        ->           (B, K, 2)
        """
        if self.track_mode in ("window", "tie"):
            return self._track_h_constrained(h, frame)
        if self.track_mode != "h" and prev_frame is not None:
            return self._track_with_frames(h, frame, prev_frame)
        B, K, Ch, H, W = h.shape
        _, C, _, _     = frame.shape

        h_tmpl = h.mean(dim=2).reshape(B * K, 1, H, W)
        f_rep  = (frame.unsqueeze(1)
                       .expand(B, K, C, H, W)
                       .reshape(B * K, C, H, W))

        with torch.no_grad():
            v_flat, _ = self.phase_corr_track(h_tmpl, f_rep)
        return v_flat.squeeze(1).reshape(B, K, 2)

    def _track_with_frames(self, h, frame, prev_frame):
        """track_mode "propose" / "prior": see __init__."""
        B, K, Ch, H, W = h.shape
        with torch.no_grad():
            f   = frame.mean(dim=1)                                   # (B, H, W)
            tm  = h.mean(dim=2)                                       # (B, K, H, W)
            s_h = self._pc_surface(tm, f.unsqueeze(1).expand_as(tm)).reshape(B, K, H * W)
            s_x = self._pc_surface(prev_frame.mean(dim=1), f).reshape(B, H * W)
            if self.track_mode == "propose":
                cand  = s_x.topk(self.n_proposals, dim=-1).indices      # (B, M)
                cand  = cand.unsqueeze(1).expand(B, K, -1)              # (B, K, M)
                score = torch.gather(s_h, 2, cand)                      # slot k's score of each
                idx   = torch.gather(cand, 2, score.argmax(-1, keepdim=True)).squeeze(-1)
            elif self.track_mode == "propose_distinct":
                cand  = s_x.topk(self.n_proposals, dim=-1).indices      # (B, M)
                score = torch.gather(s_h, 2, cand.unsqueeze(1).expand(B, K, -1))   # (B, K, M)
                perms = list(permutations(range(cand.shape[1]), K))     # injective maps
                total = torch.stack([sum(score[:, k, p[k]] for k in range(K)) for p in perms], 1)
                best  = torch.tensor(perms, device=h.device)[total.argmax(1)]      # (B, K)
                idx   = torch.gather(cand, 1, best)
            else:                                                       # "prior"
                nh  = s_h / s_h.amax(-1, keepdim=True).clamp_min(1e-8)
                nx  = s_x / s_x.amax(-1, keepdim=True).clamp_min(1e-8)
                idx = (nh + self.track_prior_weight * nx.unsqueeze(1)).argmax(-1)   # (B, K)
            return self._index_to_velocity(idx, H, W, h.dtype)

    def _track_h_constrained(self, h, frame):
        """track_mode "window" / "tie": the h surfaces with a constraint (see __init__)."""
        B, K, Ch, H, W = h.shape
        with torch.no_grad():
            f   = frame.mean(dim=1)
            tm  = h.mean(dim=2)
            s_h = self._pc_surface(tm, f.unsqueeze(1).expand_as(tm)).reshape(B, K, H * W)
            if self.track_mode == "window":
                v_all = self._index_to_velocity(torch.arange(H * W, device=h.device), H, W, h.dtype)
                inwin = (v_all.abs() <= self.track_window).all(-1)            # (H*W,)
                idx = s_h.masked_fill(~inwin, float("-inf")).argmax(-1)       # (B, K)
            else:                                                             # "tie"
                order = s_h.amax(-1).argsort(dim=1, descending=True)         # (B, K)
                taken = torch.zeros(B, H * W, dtype=torch.bool, device=h.device)
                idx   = torch.empty(B, K, dtype=torch.long, device=h.device)
                rows  = torch.arange(B, device=h.device)
                for r in range(K):
                    k  = order[:, r]
                    ix = s_h[rows, k].masked_fill(taken, float("-inf")).argmax(-1)
                    idx[rows, k] = ix
                    taken[rows, ix] = True
            return self._index_to_velocity(idx, H, W, h.dtype)

    def _track_x_continuity(self, prev_frame, frame, v_prev):
        """
        x_track_until: the top n_slots peaks of the raw pair PC(x_{t-1}, x_t), each
        slot given the candidate closest to its own previous velocity (the injective
        map minimising the summed squared distance; K! maps, K is 2). From the
        x-based-velocity-tracking branch.
        """
        with torch.no_grad():
            cand, _ = self.phase_corr_bootstrap(prev_frame, frame)          # (B, K, 2)
            K = cand.shape[1]
            perms = list(permutations(range(K)))
            cost = torch.stack([(cand[:, list(p)] - v_prev).pow(2).sum((1, 2)) for p in perms])
            best = torch.tensor(perms, device=cand.device)[cost.argmin(0)]  # (B, K)
            return torch.gather(cand, 1, best.unsqueeze(-1).expand(-1, -1, 2))

    def pool_slots(self, h):
        """h : (B, K, Ch, H, W) -> (B, Ch, H, W)"""
        if self.slot_reduce == 'mean':
            return h.mean(dim=1)
        elif self.slot_reduce == 'sum':
            return h.sum(dim=1)
        else:
            return h.max(dim=1).values

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self,
                input_seq,
                pred_len,
                target_seq=None,
                track_decoder_velocity=True,
                return_velocity=False,
                return_states=False):
        """
        input_seq : (B, T_in, C, H, W),  T_in >= 2
        pred_len  : int
        target_seq : (B, pred_len, C, H, W) or None
        track_decoder_velocity : if True (and target_seq is given), decoder
            velocities are tracked against the true next frame each step
            (v = track(h, target_seq[:, t])) — the training protocol, and an
            oracle when used at evaluation. If False, the last encoder
            velocity is frozen for the whole rollout — the deployable
            inference behavior — regardless of whether target_seq is given
            (so velocity metrics/losses can still be computed against a
            target without leaking its motion into the prediction).
        """
        if not self.batch_first:
            input_seq = input_seq.permute(1, 0, 2, 3, 4)

        B, T_in, C, H, W = input_seq.shape
        K = self.n_slots

        h, c = self.cell.init_hidden(B, K, H, W, input_seq.device, input_seq.dtype)

        # ---- Encoder ------------------------------------------------
        # v_last = torch.zeros(B, K, 2, device=input_seq.device, dtype=input_seq.dtype)
        estimated_velocities = []
        h_states = [] if return_states else None

        for t in range(T_in):

            if t == 0:
                # h=0, warp(0,v)=0 for any v. h_1 = σ(U★X_0).
                v = torch.zeros(B, K, 2, device=input_seq.device,
                                         dtype=input_seq.dtype)

            elif t == 1:
                # First non-zero h. Bootstrap from (X_0, X_1).
                v = self.bootstrap_velocities(input_seq[:, 0], input_seq[:, 1])

            elif self.training and t < self.x_track_until:
                # x-tracking curriculum (training only): raw pair + continuity.
                v = self._track_x_continuity(input_seq[:, t - 1], input_seq[:, t], v)

            elif self.training and self.x_track_p is not None and self.x_track_p[t] > 0:
                # stochastic handover (training only): per sequence, the raw-pair
                # velocity with probability p_t, the slot's own h-tracking otherwise
                p_t = float(self.x_track_p[t])
                v_x = self._track_x_continuity(input_seq[:, t - 1], input_seq[:, t], v)
                if p_t >= 1:
                    v = v_x
                else:
                    v_h = self.track_velocities(h, input_seq[:, t],
                                                prev_frame=input_seq[:, t - 1])
                    use_x = torch.rand(B, device=input_seq.device) < p_t
                    v = torch.where(use_x.view(B, 1, 1), v_x, v_h)

            else:
                # Slot self-tracking. X_t consumed exactly once.
                v = self.track_velocities(h, input_seq[:, t],
                                          prev_frame=input_seq[:, t - 1])

            h, c   = self.cell(input_seq[:, t], h, c, v)

            if t > 0:
                estimated_velocities.append(v.detach())

            if return_states:
                h_states.append(h.mean(dim=2).detach())

        # ---- Decoder ------------------------------------------------
        prev_frame = input_seq[:, -1]
        outputs    = []
        v_enc_last = v
        # scheduled freezing (training only): which sequences decode like inference
        freeze = None
        if (self.training and self.frozen_prob > 0 and target_seq is not None
                and track_decoder_velocity):
            freeze = (torch.rand(B, device=input_seq.device) < self.frozen_prob).view(B, 1, 1)

        for t in range(pred_len):
            current_frame = prev_frame.detach() if self.detach_feedback else prev_frame

            if target_seq is not None and track_decoder_velocity:
                # the true frame before target_seq[:, t] (tracked mode already
                # reads the true frames; "h" mode ignores it)
                true_prev = input_seq[:, -1] if t == 0 else target_seq[:, t - 1]
                v = self.track_velocities(h, target_seq[:, t], prev_frame=true_prev)
                if freeze is not None:
                    v = torch.where(freeze, v_enc_last, v)
                estimated_velocities.append(v.clone().detach())
            # else: v keeps the last encoder estimate (frozen rollout)

            h, c  = self.cell(current_frame, h, c, v)
            pred  = self.decode(self.pool_slots(h))
            outputs.append(pred)
            prev_frame = pred

                
            if return_states:
                h_states.append(h.mean(dim=2).detach())

        outputs = torch.stack(outputs, dim=1)

        result = [outputs]

        if return_velocity:
            result.append(torch.stack(estimated_velocities, dim=1))

        if return_states:
            result.append({
                "h": torch.stack(h_states, dim=1),  # (B, T_in+pred_len, K, H, W)
            })

        if len(result) == 1:
            return result[0]
        return tuple(result)