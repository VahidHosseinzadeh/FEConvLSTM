import random
from itertools import permutations as _permutations

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

        yy = 2 * torch.remainder(yy - dy, H) / (H - 1) - 1
        xx = 2 * torch.remainder(xx - dx, W) / (W - 1) - 1

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

    Encoder protocol: bootstrap_until
    ----------------------------------
    t=0                   v=0 (h is zero, warp(0,v)=0 for any v)
    t=1                   bootstrap(X_0, X_1) -- DEFINES slot identity
    2 <= t <= B_until     bootstrap(X_{t-1}, X_t), then bound back to slots
                          by assign_by_continuity
    t > B_until           track_velocities(h, X_t)

    bootstrap_until=1 is the original behaviour. Raising it trades the
    learned tracker for frame-pair phase correlation, which is parameter-free
    and does not degrade as h drifts -- useful both as a diagnostic (does the
    val loss still climb?) and as a curriculum: start high so early training
    sees correct velocities, then anneal toward 1 as h becomes a usable
    template. Set it >= T_in to take h out of the encoder velocity path
    entirely.

    Note the asymmetry that makes this possible only in the encoder: it has
    real frames to correlate. The decoder has only its own predictions, so it
    keeps the tracked/frozen protocol described below.

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
                 bootstrap_until=1,
                 phase_corr_kwargs=None):
        super().__init__()

        self.batch_first     = batch_first
        self.n_slots         = n_slots
        # Encoder steps 1..bootstrap_until read velocity from the raw frame
        # pair (x_{t-1}, x_t); steps after that self-track against h. 1 is the
        # original behaviour (bootstrap at t=1 only). Overridable per forward()
        # call so a training loop can anneal it down toward 1.
        self.bootstrap_until = bootstrap_until
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
                       nn.ReLU()]
            in_ch = decoder_channels
        layers += [nn.Conv2d(in_ch, output_channels,
                             3, padding=1, padding_mode='circular', bias=True)]
        self.decoder = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Velocity helpers
    # ------------------------------------------------------------------

    def bootstrap_velocities(self, x0, x1):
        """K peaks from a raw frame pair. Unordered w.r.t. slots."""
        v, _ = self.phase_corr_bootstrap(x0, x1)
        return v   # (B, K, 2)

    @staticmethod
    def assign_by_continuity(v_new, v_prev):
        """
        Bind unordered frame-pair peaks to slots by velocity continuity.

        The bootstrap returns its K peaks ordered by correlation-peak
        STRENGTH, which carries no slot identity and reshuffles from frame to
        frame. Re-running it mid-sequence therefore needs an assignment step,
        or slot k silently changes which digit it means. Continuity is the
        prior that needs no labels: whichever pairing moves the slots least is
        the one that keeps identities. Exact for constant motion (zero cost),
        and for transition_mode='smooth' a change is at most 1 per component,
        so the true pairing is still the cheapest unless two digits nearly
        swap velocities.

        Deliberately NOT an exclusion rule: two slots are free to end up on
        the same velocity, which the dataset does allow after t=0
        (_update_velocity only forbids a digit's own previous velocity).

        v_new  : (B, K, 2) unordered peaks
        v_prev : (B, K, 2) velocity each slot held at the previous step
        ->       (B, K, 2) peaks reordered so index k belongs to slot k
        """
        B, K, _ = v_new.shape
        perms = torch.tensor(list(_permutations(range(K))),
                             device=v_new.device)                # (P, K)

        # cost[b, k, j] = how far slot k would have to jump to take peak j
        cost = torch.cdist(v_prev.float(), v_new.float())         # (B, K, K)
        ar   = torch.arange(K, device=v_new.device)
        # total cost of each candidate pairing; argmin takes the FIRST
        # minimum and perms[0] is the identity, so an already-correct
        # ordering (e.g. constant motion) is left untouched.
        totals = torch.stack([cost[:, ar, p].sum(-1) for p in perms], dim=1)
        sel    = perms[totals.argmin(dim=1)]                      # (B, K)
        return torch.gather(v_new, 1, sel.unsqueeze(-1).expand(B, K, 2))

    def track_velocities(self, h, frame):
        """
        Per-slot self-tracking: correlate each slot's h against frame.
        All B*K pairs in one batched call.

        h     : (B, K, Ch, H, W)
        frame : (B, C,  H,  W)
        ->      (B, K, 2)
        """
        B, K, Ch, H, W = h.shape
        _, C, _, _     = frame.shape

        h_tmpl = h.mean(dim=2).reshape(B * K, 1, H, W)
        f_rep  = (frame.unsqueeze(1)
                       .expand(B, K, C, H, W)
                       .reshape(B * K, C, H, W))

        with torch.no_grad():
            v_flat, _ = self.phase_corr_track(h_tmpl, f_rep)
        return v_flat.squeeze(1).reshape(B, K, 2)

    def tracking_logits(self, h, frame):
        """
        The tracker's correlation surface, WITH gradient w.r.t. h.

        Same computation track_velocities runs, minus the argmax -- so this is
        the differentiable view of "where does slot k think its content went".
        Standardised per (sequence, slot) because the phase-normalised surface
        has a tiny dynamic range: fed raw into a softmax it is nearly uniform
        and the gradient all but vanishes.

        h     : (B, K, Ch, H, W)
        frame : (B, C,  H,  W)
        ->      (B, K, H*W) logits over candidate displacements
        """
        B, K, Ch, H, W = h.shape
        C = frame.shape[1]

        h_tmpl = h.mean(dim=2).reshape(B * K, 1, H, W)
        f_rep  = (frame.unsqueeze(1)
                       .expand(B, K, C, H, W)
                       .reshape(B * K, C, H, W))

        surf = self.phase_corr_track.surface(h_tmpl, f_rep)     # (B*K, H, W)
        surf = surf.reshape(B * K, -1)
        surf = (surf - surf.mean(dim=1, keepdim=True)) / (
            surf.std(dim=1, keepdim=True) + 1e-6)
        return surf.reshape(B, K, -1)

    def tracking_ce(self, h, frame, v_target, temperature=1.0):
        """
        Cross-entropy pulling each slot's correlation peak onto v_target.

        The teacher is frame-pair phase correlation on (x_{t-1}, x_t) -- no
        dataset labels, and measured ~99% correct on constant motion / ~93%
        on piecewise. The student is the h-based tracker. Because each slot
        gets its OWN target, this also penalises collapse directly: two slots
        peaking at the same displacement cannot both be right unless the two
        digits genuinely share a velocity.

        Returns a scalar. Gradient flows into h (and thus the whole cell);
        v_target is treated as a constant.
        """
        B, K, _, H, W = h.shape
        logits = self.tracking_logits(h, frame) / temperature
        target = self.phase_corr_track.velocity_to_index(
            v_target.detach(), H, W)                            # (B, K)
        return F.cross_entropy(logits.reshape(B * K, -1),
                               target.reshape(B * K))

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
                bootstrap_until=None,
                track_loss_temperature=1.0,
                return_velocity=False,
                return_states=False,
                return_track_loss=False):
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

        # None -> the constructor's value. Anneal this toward 1 during
        # training to hand the job over to h-tracking gradually.
        boot_until = (self.bootstrap_until if bootstrap_until is None
                      else bootstrap_until)

        # Raw CE; the caller scales it (train.py's --track_loss_weight) and
        # decides whether to backprop it. Worth requesting even at weight 0:
        # it is the readout for "is h a good enough template to take over
        # yet", i.e. when to anneal bootstrap_until down.
        want_track_loss = return_track_loss
        track_ce = torch.zeros((), device=input_seq.device,
                               dtype=input_seq.dtype)
        n_ce = 0

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
                # First non-zero h. Bootstrap from (X_0, X_1). No assignment
                # needed here: this call DEFINES slot identity -- slot k just
                # is peak k, and everything downstream inherits that.
                v = self.bootstrap_velocities(input_seq[:, 0], input_seq[:, 1])

            else:
                # Teacher: motion read straight off the raw frame pair, bound
                # back to slot identities. Computed whenever it is needed --
                # to DRIVE the warp while t <= boot_until, and/or to SUPERVISE
                # the tracker when the distillation loss is on. Keeping it on
                # after boot_until is what makes annealing a hand-over rather
                # than the removal of a crutch: h keeps learning either way.
                need_teacher = (t <= boot_until) or want_track_loss
                if need_teacher:
                    v_teacher = self.assign_by_continuity(
                        self.bootstrap_velocities(input_seq[:, t - 1],
                                                  input_seq[:, t]),
                        v)

                if want_track_loss:
                    # h here is still h_{t-1}: the same state track_velocities
                    # would correlate against x_t, so student and teacher
                    # describe the same transition.
                    track_ce = track_ce + self.tracking_ce(
                        h, input_seq[:, t], v_teacher, track_loss_temperature)
                    n_ce += 1

                if t <= boot_until:
                    v = v_teacher
                else:
                    # Slot self-tracking. X_t consumed exactly once.
                    v = self.track_velocities(h, input_seq[:, t])

            h, c   = self.cell(input_seq[:, t], h, c, v)

            if t > 0:
                estimated_velocities.append(v.detach())

            if return_states:
                h_states.append(h.mean(dim=2).detach())

        # ---- Decoder ------------------------------------------------
        prev_frame = input_seq[:, -1]
        outputs    = []

        for t in range(pred_len):
            current_frame = prev_frame.detach()

            if target_seq is not None and track_decoder_velocity:
                v = self.track_velocities(h, target_seq[:, t])
                estimated_velocities.append(v.clone().detach())
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

        if return_states:
            result.append({
                "h": torch.stack(h_states, dim=1),  # (B, T_in+pred_len, K, H, W)
            })

        if return_track_loss:
            # Mean over the encoder steps that produced a teacher, so the
            # magnitude does not depend on T_in or on bootstrap_until.
            result.append(track_ce / max(n_ce, 1))

        if len(result) == 1:
            return result[0]
        return tuple(result)