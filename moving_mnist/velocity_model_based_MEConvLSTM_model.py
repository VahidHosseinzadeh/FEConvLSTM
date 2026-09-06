import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from velocity_predictor_model import PhaseCorrelation
from velocity_dynamics_model import VelocityDynamicsHead


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

    Velocity dynamics head (optional, --use_velocity_dynamics)
    ----------------------------------------------------------
    "No better option exists without GT" is true only as long as velocity is
    treated as something to be *measured*. VelocityDynamicsHead adds a process
    model: a small GRU over the velocity history that predicts u_{t+1}, so the
    decoder can extrapolate instead of freezing. Together with the correlation
    head this is a predict/correct filter -- measurement + process model.

    It is exactly motion-equivariant (see velocity_dynamics_model.py), so
    turning it on does not cost the property the whole architecture exists for.
    With use_velocity_dynamics=False the forward pass below is byte-for-byte
    the old one; with it True but freshly initialized, the zero-initialized
    output layer makes the head's prediction equal to the frozen velocity, so
    it nests the old behaviour rather than replacing it.
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
                 track_corr_alpha=None,
                 use_velocity_dynamics=False,
                 vel_dyn_state_dim=32,
                 vel_dyn_use_h=False,
                 vel_dyn_gain='fixed',
                 vel_dyn_h_embed_dim=16,
                 vel_dyn_v_max=None,
                 vel_dyn_openloop_k=0,
                 vel_dyn_arch='gru',
                 vel_dyn_layers=1,
                 vel_dyn_decoder_supervision='none'):
        super().__init__()

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

        # The two correlators do genuinely different jobs and want different
        # whitening (see PhaseCorrelation's docstring):
        #
        #   bootstrap  frame vs frame -- both SHARP. Full phase correlation
        #              (alpha=1) is exact here and measures 100% on this data.
        #   track      h vs frame -- the template is h.mean(dim=2), a smooth
        #              tanh-saturated map. alpha=1 whitens bands that h barely
        #              occupies up to unit gain, which is noise, and the peak
        #              collapses: 0% on a blur+tanh template where plain
        #              cross-correlation gets 65%.
        #
        # track_corr_alpha=None keeps them identical, i.e. exactly the previous
        # behaviour; set it (0.0 is the measured best) to decouple them.
        track_kw = dict(pc_kw)
        if track_corr_alpha is not None:
            track_kw["alpha"] = track_corr_alpha
        self.track_corr_alpha = track_corr_alpha

        self.phase_corr_bootstrap = PhaseCorrelation(n_modes=n_slots, **pc_kw)
        self.phase_corr_track     = PhaseCorrelation(n_modes=1,       **track_kw)

        self.cell = MEConvLSTMCell(input_channels, hidden_channels,
                                   kernel_size, bias)

        layers = []
        in_ch = hidden_channels
        for _ in range(decoder_layers):
            layers += [nn.Conv2d(in_ch, decoder_channels,
                                 3, padding=1, padding_mode='circular', bias=True),
                       nn.ReLU()]
            in_ch = decoder_channels
        layers += [nn.Conv2d(in_ch, output_channels,
                             3, padding=1, padding_mode='circular', bias=True)]
        self.decoder = nn.Sequential(*layers)

        # ---- optional velocity process model -------------------------
        # Off by default: every existing run, checkpoint and result must be
        # bit-for-bit unaffected, so nothing below is even constructed unless
        # asked for (a constructed-but-unused module would still change the
        # parameter count and the RNG stream during init).
        if vel_dyn_gain not in ('fixed', 'learned'):
            raise ValueError(f"vel_dyn_gain must be 'fixed' or 'learned', got {vel_dyn_gain!r}")

        if vel_dyn_decoder_supervision not in ('none', 'teacher', 'openloop'):
            raise ValueError("vel_dyn_decoder_supervision must be one of "
                             f"'none'/'teacher'/'openloop', got {vel_dyn_decoder_supervision!r}")
        self.vel_dyn_decoder_supervision = vel_dyn_decoder_supervision
        self.use_velocity_dynamics = use_velocity_dynamics
        self.vel_dyn_gain          = vel_dyn_gain
        self.vel_dyn_openloop_k    = vel_dyn_openloop_k
        self.vel_dyn               = None
        self.vel_dyn_gain_mlp      = None

        if use_velocity_dynamics:
            self.vel_dyn = VelocityDynamicsHead(
                state_dim=vel_dyn_state_dim,
                hidden_channels=hidden_channels,
                use_h=vel_dyn_use_h,
                h_embed_dim=vel_dyn_h_embed_dim,
                v_max=vel_dyn_v_max,
                arch=vel_dyn_arch,
                n_layers=vel_dyn_layers,
            )
            if vel_dyn_gain == 'learned':
                # Kalman-ish gain from the correlation peak strength: a weak
                # peak means the measurement is unreliable and the process
                # model should be trusted more. The peak score is INVARIANT
                # under global translation (phase correlation shifts the peak,
                # it does not change its height), so k is invariant and
                #     u = u_pred + k * (v_meas - u_pred)
                # stays equivariant: both terms carry +v, their difference
                # carries none, and k scales an invariant quantity.
                self.vel_dyn_gain_mlp = nn.Sequential(
                    nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1),
                )
                # Start at k = sigmoid(6) ~ 0.9975, i.e. essentially "trust the
                # measurement" -- the same regime as the 'fixed' default -- so
                # the learned gain has to earn its departure from it rather
                # than starting somewhere arbitrary. (It cannot nest 'fixed'
                # exactly: sigmoid never reaches 1.)
                nn.init.zeros_(self.vel_dyn_gain_mlp[-1].weight)
                nn.init.constant_(self.vel_dyn_gain_mlp[-1].bias, 6.0)

    # ------------------------------------------------------------------
    # Velocity helpers
    # ------------------------------------------------------------------

    def bootstrap_velocities(self, x0, x1, return_score=False):
        """K peaks from raw frame pair. Called once: encoder t=1.

        return_score additionally returns the (B, K) correlation peak heights,
        which the learned dynamics gain uses as a confidence signal. Default
        False keeps the original single-return signature for every existing
        caller."""
        v, s = self.phase_corr_bootstrap(x0, x1)
        return (v, s) if return_score else v   # (B, K, 2), (B, K)

    def track_velocities(self, h, frame, return_score=False):
        """
        Per-slot self-tracking: correlate each slot's h against frame.
        All B*K pairs in one batched call.

        h     : (B, K, Ch, H, W)
        frame : (B, C,  H,  W)
        ->      (B, K, 2)   (and (B, K) peak scores if return_score)

        The scores were always computed and thrown away; the learned dynamics
        gain needs them, so they can now be asked for. Nothing about the
        velocity path changes.
        """
        B, K, Ch, H, W = h.shape
        _, C, _, _     = frame.shape

        h_tmpl = h.mean(dim=2).reshape(B * K, 1, H, W)
        f_rep  = (frame.unsqueeze(1)
                       .expand(B, K, C, H, W)
                       .reshape(B * K, C, H, W))

        with torch.no_grad():
            v_flat, s_flat = self.phase_corr_track(h_tmpl, f_rep)
        v = v_flat.squeeze(1).reshape(B, K, 2)
        if not return_score:
            return v
        return v, s_flat.squeeze(1).reshape(B, K)

    def pool_slots(self, h):
        """h : (B, K, Ch, H, W) -> (B, Ch, H, W)"""
        if self.slot_reduce == 'mean':
            return h.mean(dim=1)
        elif self.slot_reduce == 'sum':
            return h.sum(dim=1)
        else:
            return h.max(dim=1).values

    def _dyn_gain(self, score):
        """
        Blending weight k for u = u_pred + k * (v_meas - u_pred).

        'fixed' returns None, which the caller reads as "k = 1, take the
        measurement verbatim". It is deliberately not the float 1.0: writing
        u_pred + 1.0 * (v_meas - u_pred) is not bit-identical to v_meas in
        floating point, and the nesting guarantee (dynamics head on == old
        behaviour) has to hold exactly, not to 1e-7.
        """
        if self.vel_dyn_gain == 'fixed':
            return None
        k = torch.sigmoid(self.vel_dyn_gain_mlp(score.unsqueeze(-1)))  # (B,K,1)
        return k

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self,
                input_seq,
                pred_len,
                target_seq=None,
                track_decoder_velocity=True,
                predict_decoder_velocity=False,
                decoder_sampling_p=0.0,
                return_velocity=False,
                return_dyn_loss=False,
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
        decoder_sampling_p : scheduled sampling on the DECODER VELOCITY. With
            this probability, a training step runs the decoder on the head's
            own predicted velocity instead of the tracked measurement. It
            exists because the model is otherwise optimised in a regime it is
            never evaluated in: training always hands the decoder an oracle
            velocity, so the ConvLSTM learns to depend on one and is never
            given a gradient that would let it cope without. Measured on a
            fixed held-out set while training, the two protocols pull apart --
            oracle-velocity loss improved 6x while frozen-velocity loss on the
            SAME data got 43% worse. The coin is flipped once per forward call
            rather than per step, so each rollout is a coherent protocol rather
            than a mixture. Only active in training mode, and only with the
            dynamics head present. 0.0 = off, previous behaviour exactly.
        predict_decoder_velocity : the third decoder mode. Requires
            use_velocity_dynamics. The dynamics head rolls the velocity
            forward with no measurement at all (k = 0) — deployable like
            "frozen", but able to follow a velocity that keeps changing after
            the context ends. Ignored when the tracked (oracle) protocol is
            active: a real measurement always beats a prediction.
        return_dyn_loss : return the mean one-step-ahead dynamics loss (a
            scalar tensor) for the caller to add to its training objective.
            Zero when the head is off or when no supervised step occurred.

        Return order when several flags are set:
            outputs, [velocities], [dyn_loss], [states]
        """
        if not self.batch_first:
            input_seq = input_seq.permute(1, 0, 2, 3, 4)

        B, T_in, C, H, W = input_seq.shape
        K = self.n_slots
        device, dtype = input_seq.device, input_seq.dtype

        h, c = self.cell.init_hidden(B, K, H, W, input_seq.device, input_seq.dtype)

        # ---- Velocity dynamics bookkeeping --------------------------
        # use_dyn gates every line this feature added. With it False the code
        # below is exactly the pre-existing forward pass — same ops, same
        # order, same RNG consumption — which is what the nesting test checks.
        use_dyn   = self.vel_dyn is not None
        dyn_state = self.vel_dyn.init_state(B, K, device, dtype) if use_dyn else None
        u_prev    = torch.zeros(B, K, 2, device=device, dtype=dtype)
        du_prev   = torch.zeros(B, K, 2, device=device, dtype=dtype)
        n_meas    = 0                      # measurements consumed so far
        dyn_terms = []                     # per-step smooth-L1 terms
        # (t, h_at_t, v_meas) per supervised encoder step, for the optional
        # open-loop replay. Holds references to tensors autograd is keeping
        # alive anyway, so it costs bookkeeping, not memory.
        dyn_records = []
        ol_fork     = None
        # Earliest usable fork is t=2: t=1 is the first measurement, so t=2 is
        # the first step at which "predict the next velocity" is even defined.
        t_fork      = max(2, T_in - self.vel_dyn_openloop_k) if use_dyn else None

        # ---- Encoder ------------------------------------------------
        # v_last = torch.zeros(B, K, 2, device=input_seq.device, dtype=input_seq.dtype)
        estimated_velocities = []
        h_states = [] if return_states else None

        for t in range(T_in):

            score = None

            if t == 0:
                # h=0, warp(0,v)=0 for any v. h_1 = σ(U★X_0).
                v = torch.zeros(B, K, 2, device=input_seq.device,
                                         dtype=input_seq.dtype)

            elif t == 1:
                # First non-zero h. Bootstrap from (X_0, X_1).
                if use_dyn and self.vel_dyn_gain == 'learned':
                    v, score = self.bootstrap_velocities(
                        input_seq[:, 0], input_seq[:, 1], return_score=True)
                else:
                    v = self.bootstrap_velocities(input_seq[:, 0], input_seq[:, 1])

            else:
                # Slot self-tracking. X_t consumed exactly once.
                if use_dyn and self.vel_dyn_gain == 'learned':
                    v, score = self.track_velocities(h, input_seq[:, t],
                                                     return_score=True)
                else:
                    v = self.track_velocities(h, input_seq[:, t])

            # Predict/correct. t=0 produces no measurement (v is a placeholder
            # zero, not an estimate), so the filter starts at t=1.
            if use_dyn and t >= 1:
                if self.vel_dyn_openloop_k > 0 and t == t_fork and n_meas >= 1:
                    ol_fork = (u_prev, du_prev, dyn_state)

                u_pred, dyn_state = self.vel_dyn(u_prev, du_prev, h, dyn_state)

                # The measurement is the TARGET and is detached: the dynamics
                # head learns from phase correlation, phase correlation must
                # never learn from the dynamics head. (It is non-differentiable
                # anyway — the velocity comes from an argmax index — so this
                # detach is documentation as much as it is a stop-gradient.)
                # No term at the first measurement: there is nothing to have
                # predicted it from.
                if n_meas >= 1:
                    dyn_terms.append(F.smooth_l1_loss(u_pred, v.detach()))

                # Blend only once the process model has seen something. At the
                # very first measurement u_pred is just the zero-initialized
                # u_prev, so blending would shrink the bootstrap velocity
                # toward zero on the strength of a prediction made from no
                # evidence at all.
                k = self._dyn_gain(score) if n_meas >= 1 else None
                if k is not None:
                    v = u_pred + k * (v - u_pred)
                # k is None => k = 1 => v is the measurement, untouched.

                # du is only meaningful once two measurements exist; zeros
                # otherwise (feeding u_1 - 0 would tell the GRU the digit just
                # accelerated from rest, which never happened).
                du_prev = (v - u_prev) if n_meas >= 1 else torch.zeros_like(v)
                u_prev  = v
                n_meas += 1
                if self.vel_dyn_openloop_k > 0:
                    dyn_records.append((t, h, v.detach()))

            h, c   = self.cell(input_seq[:, t], h, c, v)

            if t > 0:
                estimated_velocities.append(v.detach())

            if return_states:
                h_states.append(h.mean(dim=2).detach())

        # ---- Optional open-loop velocity supervision ----------------
        # Multi-step supervision for free: re-run the head from a fork point
        # partway through the encoder, feeding it its OWN predictions instead
        # of the measurements, and score it against the measurements already
        # in hand. This trains the extrapolation regime (exactly what the
        # decoder does) without rolling out a single image. The main path
        # above is untouched — the fork is a side branch.
        if use_dyn and self.vel_dyn_openloop_k > 0 and ol_fork is not None:
            u_ol, du_ol, st_ol = ol_fork
            for t_rec, h_rec, v_rec in dyn_records:
                if t_rec < t_fork:
                    continue
                u_ol_next, st_ol = self.vel_dyn(u_ol, du_ol, h_rec, st_ol)
                dyn_terms.append(F.smooth_l1_loss(u_ol_next, v_rec))
                du_ol = u_ol_next - u_ol
                u_ol  = u_ol_next

        # ---- Decoder ------------------------------------------------
        # Scheduled sampling: decided once for the whole rollout (see the
        # docstring) and only while training.
        sample_predicted = (self.training and use_dyn and decoder_sampling_p > 0.0
                            and random.random() < decoder_sampling_p)

        prev_frame = input_seq[:, -1]
        outputs    = []

        for t in range(pred_len):
            current_frame = prev_frame.detach()

            if target_seq is not None and track_decoder_velocity and not sample_predicted:
                v = self.track_velocities(h, target_seq[:, t])

                # What, if anything, the velocity head is allowed to take from
                # the FUTURE frames. v here is measured against target_seq, so
                # every use of it past this point is information the head will
                # not have at inference.
                #
                #   'none'     the head is not run at all here. Its only
                #              supervision comes from the context, which is the
                #              only thing it will ever see. Default.
                #   'teacher'  run it, feed it the ORACLE previous velocity,
                #              and score its one-step-ahead output. This was
                #              the original behaviour and it is why the head
                #              learned to LAG: "given the true u_t, predict
                #              u_{t+1}" is a teacher-forced task whose optimal
                #              solution, when the signal is hard, is to repeat
                #              the input. Measured at one-step-lag quality
                #              (0.1179 against a 0.1149 lag ceiling).
                #   'openloop' run it on its OWN previous output and score that
                #              against the measurement. The future frames are
                #              then used only as TARGETS, never as inputs, so
                #              the head is trained to extrapolate rather than
                #              to track. Uses target frames; 'none' does not.
                sup = self.vel_dyn_decoder_supervision
                if use_dyn and sup != 'none':
                    u_pred, dyn_state = self.vel_dyn(u_prev, du_prev, h, dyn_state)
                    if n_meas >= 1:
                        dyn_terms.append(F.smooth_l1_loss(u_pred, v.detach()))
                    nxt = v if sup == 'teacher' else u_pred
                    du_prev = (nxt - u_prev) if n_meas >= 1 else torch.zeros_like(nxt)
                    u_prev  = nxt
                    n_meas += 1
                estimated_velocities.append(v.clone().detach())

            elif use_dyn and (predict_decoder_velocity or sample_predicted):
                # No measurement exists here, so k = 0: the process model runs
                # the velocity forward on its own. Gradients from the image
                # loss do reach the head through the warp, which is intended —
                # in this mode the head is part of the rollout, not a bystander.
                v, dyn_state = self.vel_dyn(u_prev, du_prev, h, dyn_state)
                du_prev = v - u_prev
                u_prev  = v
                estimated_velocities.append(v.detach())
            # else: v keeps the last encoder estimate (frozen rollout)

            h, c  = self.cell(current_frame, h, c, v)
            pred  = self.decoder(self.pool_slots(h))
            outputs.append(pred)
            prev_frame = pred


            if return_states:
                h_states.append(h.mean(dim=2).detach())

        outputs = torch.stack(outputs, dim=1)

        result = [outputs]

        if return_velocity:
            result.append(torch.stack(estimated_velocities, dim=1))

        if return_dyn_loss:
            # Mean, not sum: the magnitude must not depend on T_in / pred_len,
            # or --vel_dyn_loss_weight would silently mean something different
            # for every sequence length.
            result.append(torch.stack(dyn_terms).mean() if dyn_terms
                          else torch.zeros((), device=device, dtype=dtype))

        if return_states:
            result.append({
                "h": torch.stack(h_states, dim=1),  # (B, T_in+pred_len, K, H, W)
            })

        if len(result) == 1:
            return result[0]
        return tuple(result)