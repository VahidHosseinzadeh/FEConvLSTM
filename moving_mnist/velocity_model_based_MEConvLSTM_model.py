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
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim,
            kernel_size, padding=padding,
            padding_mode="circular", bias=bias
        )

        if bias:
            nn.init.constant_(self.conv.bias[hidden_dim:2 * hidden_dim], 1.0)

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

        yy, xx = torch.meshgrid(
            torch.arange(H, device=x.device, dtype=x.dtype),
            torch.arange(W, device=x.device, dtype=x.dtype),
            indexing="ij"
        )
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
                          → teacher forcing controls this

    If you use current_frame for both (as in a naive implementation),
    then at decoder t=0 with no teacher forcing:
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

    cell input is controlled by teacher forcing:
        - ratio=1.0 → cell sees GT frame  (fast, stable training)
        - ratio=0.0 → cell sees own prediction (harder, closer to inference)
        - both get the same accurate velocity from GT

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
                 bias=True,
                 batch_first=True,
                 phase_corr_kwargs=None):
        super().__init__()

        self.batch_first     = batch_first
        self.n_slots         = n_slots
        self.hidden_channels = hidden_channels
        self.slot_reduce     = slot_reduce
        output_channels      = output_channels or input_channels

        pc_kw = phase_corr_kwargs or {}

        self.phase_corr_bootstrap = PhaseCorrelation(n_modes=n_slots, **pc_kw)
        self.phase_corr_track     = PhaseCorrelation(n_modes=1,       **pc_kw)

        self.cell = MEConvLSTMCell(input_channels, hidden_channels,
                                   kernel_size, bias)

        layers = []
        for _ in range(decoder_layers):
            layers += [nn.Conv2d(hidden_channels, hidden_channels,
                                 3, padding=1, padding_mode='circular', bias=True),
                       nn.ReLU()]
        layers += [nn.Conv2d(hidden_channels, output_channels,
                             3, padding=1, padding_mode='circular', bias=True)]
        self.decoder = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Velocity helpers
    # ------------------------------------------------------------------

    def _bootstrap_velocities(self, x0, x1):
        """K peaks from raw frame pair. Called once: encoder t=1."""
        v, _ = self.phase_corr_bootstrap(x0, x1)
        return v   # (B, K, 2)

    def _track_velocities(self, h, frame):
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

    def _pool_slots(self, h):
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
                teacher_forcing_ratio=0.0,
                target_seq=None,
                return_velocity=False):
        """
        input_seq : (B, T_in, C, H, W),  T_in >= 2
        pred_len  : int
        teacher_forcing_ratio : float in [0, 1]
        target_seq : (B, pred_len, C, H, W) or None
        """
        if not self.batch_first:
            input_seq = input_seq.permute(1, 0, 2, 3, 4)

        B, T_in, C, H, W = input_seq.shape
        K = self.n_slots

        h, c = self.cell.init_hidden(B, K, H, W, input_seq.device, input_seq.dtype)

        # ---- Encoder ------------------------------------------------
        v_last = torch.zeros(B, K, 2, device=input_seq.device, dtype=input_seq.dtype)
        estimated_velocities = []

        for t in range(T_in):

            if t == 0:
                # h=0, warp(0,v)=0 for any v. h_1 = σ(U★X_0).
                v = torch.zeros(B, K, 2, device=input_seq.device,
                                         dtype=input_seq.dtype)

            elif t == 1:
                # First non-zero h. Bootstrap from (X_0, X_1).
                v = self._bootstrap_velocities(input_seq[:, 0], input_seq[:, 1])

            else:
                # Slot self-tracking. X_t consumed exactly once.
                v = self._track_velocities(h, input_seq[:, t])

            h, c   = self.cell(input_seq[:, t], h, c, v)
            v_last = v
            
            if t > 0:
                estimated_velocities.append(v.detach())

        # ---- Decoder ------------------------------------------------
        prev_frame = input_seq[:, -1]
        outputs    = []

        for t in range(pred_len):

            if target_seq is not None:
                # ---------------------------------------------------------
                # Training / scheduled sampling.
                #
                # Velocity: track(h, target_seq[:, t])
                #   h was built up to time T_in + t - 1.
                #   target_seq[:, t] is the true next frame.
                #   → "where in the GT next frame did each slot go?"
                #   → accurate velocity, no circular dependency.
                #
                # Cell input: teacher forcing decides.
                #   The cell input does NOT need to be the velocity frame.
                #   ratio=1 → cell sees GT (stable training signal)
                #   ratio=0 → cell sees own prediction (closer to inference)
                # ---------------------------------------------------------
                v = self._track_velocities(h, target_seq[:, t])

                if (self.training and
                        torch.rand(1).item() < teacher_forcing_ratio):
                    current_frame = target_seq[:, t]
                else:
                    current_frame = prev_frame.detach()

            else:
                # ---------------------------------------------------------
                # Inference: no target_seq.
                # Freeze v_last — the last encoder velocity.
                # Tracking against own predictions reintroduces circular
                # dependency (corrupted h vs corrupted prediction → bad v).
                # For constant-velocity data this is exact. For time-varying
                # motion over the horizon, no better option exists.
                # ---------------------------------------------------------
                v             = v_last
                current_frame = prev_frame.detach()

            h, c  = self.cell(current_frame, h, c, v)
            pred  = self.decoder(self._pool_slots(h))
            outputs.append(pred)
            prev_frame = pred

        return torch.stack(outputs, dim=1)