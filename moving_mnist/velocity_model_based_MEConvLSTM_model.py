import random
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
        self.input_dim  = input_dim
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

    def encode_input(self, x):
        """
        Input-side half of the gate convolution — the frame seen through this
        cell's own kernels, with no extra parameters.

        self.conv runs on cat([x, h], dim=1), so its weight splits along the
        in-channel axis into an x-block [:, :input_dim] and an h-block
        [:, input_dim:]. Slicing the x-block gives exactly the filters the
        cell applies to a frame. Bias is deliberately dropped: it is a
        per-channel constant, i.e. a pure DC term, and after the
        phase-correlation magnitude normalisation the DC bin only adds a
        shift-independent offset to the correlation surface (it cannot move
        the peak), so carrying it buys nothing.

        x : (B, C, H, W) -> (B, 4 * hidden_dim, H, W)
        """
        ph, pw = self.conv.padding
        x = F.pad(x, (pw, pw, ph, ph), mode="circular")   # match conv's padding_mode
        return F.conv2d(x, self.conv.weight[:, :self.input_dim])

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
                 phase_corr_kwargs=None):
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
        """K peaks from raw frame pair. Called once: encoder t=1."""
        v, _ = self.phase_corr_bootstrap(x0, x1)
        return v   # (B, K, 2)

    def track_velocities(self, h, frame):
        """
        Per-slot self-tracking: correlate each slot's h against the *encoded*
        frame. All B*K pairs in one batched call.

        h lives in the cell's feature space, the raw frame does not, so the
        frame is first pushed through the cell's own input kernels
        (cell.encode_input — the x-block of the gate conv, no new weights).
        Both operands are then channel-means of the same learned feature
        space, which is what phase correlation compares.

        h     : (B, K, Ch, H, W)
        frame : (B, C,  H,  W)
        ->      (B, K, 2)
        """
        B, K, Ch, H, W = h.shape

        h_tmpl = h.mean(dim=2).reshape(B * K, 1, H, W)

        with torch.no_grad():
            # PhaseCorrelation collapses channels with mean(dim=1) as its first
            # op, so reducing here is identical — and avoids materialising the
            # (B*K, 4*hidden, H, W) copy that expanding first would cost.
            f_enc = self.cell.encode_input(frame).mean(dim=1, keepdim=True)
            f_rep = (f_enc.unsqueeze(1)
                          .expand(B, K, 1, H, W)
                          .reshape(B * K, 1, H, W))

            v_flat, _ = self.phase_corr_track(h_tmpl, f_rep)
        return v_flat.squeeze(1).reshape(B, K, 2)

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

        if len(result) == 1:
            return result[0]
        return tuple(result)