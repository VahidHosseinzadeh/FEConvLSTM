import torch
import torch.nn as nn
import random


class FEConvLSTMCell(nn.Module):

    def __init__(self,
                 input_channels,
                 hidden_channels,
                 kernel_size=3,
                 v_range=0,
                 velocity_mixing=False):

        super().__init__()

        self.hidden_channels = hidden_channels
        self.v_range = v_range
        self.v_side = 2 * v_range + 1

        self.v_list = [
            (vx, vy)
            for vx in range(-v_range, v_range + 1)
            for vy in range(-v_range, v_range + 1)
        ]

        self.num_v = len(self.v_list)

        # Velocity-lattice mixing: each step, state may hop to a NEIGHBORING
        # velocity channel (|dvx|<=1, |dvy|<=1). Without this, channels never
        # communicate, so when a digit changes velocity the newly-correct
        # channel must re-encode it from raw input. With piecewise-smooth
        # motion (velocity steps to a lattice neighbor), a 3x3 kernel over
        # the (vx, vy) lattice matches the data's transition prior exactly.
        # 9 learnable logits -> softmax -> row-renormalized stochastic
        # transition weights (convex combination: cannot blow up c). Center
        # logit init 4.0 gives ~0.87 self-weight: ~identity at init, so the
        # model starts as (nearly) the unmixed FELSTM and learns to open up.
        self.velocity_mixing = velocity_mixing and self.num_v > 1
        if self.velocity_mixing:
            logits = torch.zeros(9)
            logits[4] = 4.0
            self.vel_mix_logits = nn.Parameter(logits)
            self.register_buffer(
                "_vel_adj", self._build_vel_adjacency(), persistent=False
            )

        padding = kernel_size // 2

        # Shared gate convolution
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
            padding_mode='circular',
            bias=True
        )

        # Initialize forget gate bias positively
        nn.init.constant_(
            self.conv.bias[
                hidden_channels : 2 * hidden_channels
            ],
            1.0
        )

        # (H, W, device) -> flat gather index (num_v, H*W) reproducing the
        # per-velocity torch.roll pattern below. Depends only on shape/
        # device, not on B/C/the actual tensor values -- identical every
        # call, so build it once instead of doing num_v separate torch.roll
        # kernel launches (in a Python loop, twice per cell step, every
        # timestep) on every single forward pass.
        self._shift_index_cache = {}

    def _get_shift_index(self, H, W, device):
        key = (H, W, device)
        idx = self._shift_index_cache.get(key)
        if idx is None:
            yy, xx = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing="ij",
            )
            rows = []
            for (vx, vy) in self.v_list:
                # matches torch.roll(x, shifts=(vy, vx), dims=(2, 3)):
                # output[h, w] = input[(h - vy) mod H, (w - vx) mod W]
                src_y = (yy - vy) % H
                src_x = (xx - vx) % W
                rows.append((src_y * W + src_x).reshape(-1))
            idx = torch.stack(rows, dim=0)  # (num_v, H*W), long
            self._shift_index_cache[key] = idx
        return idx

    def _build_vel_adjacency(self):
        """
        (num_v, num_v, 9) one-hot adjacency of the velocity lattice:
        adj[u, v, k] = 1 iff channel v's velocity equals channel u's velocity
        plus the k-th offset, k enumerating (dvx, dvy) in {-1,0,1}^2
        (k = (dvx+1)*3 + (dvy+1), so k=4 is the identity tap). Edge channels
        simply lack the out-of-range taps; mix_velocity renormalizes rows.
        """
        side, num_v = self.v_side, self.num_v
        adj = torch.zeros(num_v, num_v, 9)
        for u, (vx, vy) in enumerate(self.v_list):
            for dvx in (-1, 0, 1):
                for dvy in (-1, 0, 1):
                    nvx, nvy = vx + dvx, vy + dvy
                    if abs(nvx) <= self.v_range and abs(nvy) <= self.v_range:
                        v = (nvx + self.v_range) * side + (nvy + self.v_range)
                        k = (dvx + 1) * 3 + (dvy + 1)
                        adj[u, v, k] = 1.0
        return adj

    def mix_velocity(self, x):
        """
        x : (B, num_v, C, H, W) -> same shape, each channel a convex
        combination of its 3x3 velocity-lattice neighborhood.
        """
        w = torch.softmax(self.vel_mix_logits, dim=0)      # (9,)
        M = (self._vel_adj * w).sum(dim=-1)                # (num_v, num_v)
        M = M / M.sum(dim=1, keepdim=True)                 # rows sum to 1

        B, num_v, C, H, W = x.shape
        out = torch.matmul(M, x.reshape(B, num_v, -1))
        return out.reshape(B, num_v, C, H, W)

    def shift_tensor(self, x):

        # x:
        # (B, num_v, C, H, W)
        B, num_v, C, H, W = x.shape

        idx = self._get_shift_index(H, W, x.device)          # (num_v, H*W)
        idx = idx.view(1, num_v, 1, H * W).expand(B, num_v, C, H * W)

        x_flat = x.reshape(B, num_v, C, H * W)
        shifted = torch.gather(x_flat, dim=3, index=idx)
        return shifted.reshape(B, num_v, C, H, W)

    def forward(self, u_t, state):

        # u_t:
        # (B, C, H, W)

        # h_prev, c_prev:
        # (B, num_v, hidden, H, W)

        h_prev, c_prev = state

        B, _, H, W = u_t.shape

        # Transport hidden and cell states
        h_shift = self.shift_tensor(h_prev)
        c_shift = self.shift_tensor(c_prev)

        # Optional handoff between neighboring velocity channels (shared
        # transition kernel for h and c: one motion prior, applied to both)
        if self.velocity_mixing:
            h_shift = self.mix_velocity(h_shift)
            c_shift = self.mix_velocity(c_shift)

        # Expand input over velocity channels
        u_expand = (
            u_t
            .unsqueeze(1)
            .expand(-1, self.num_v, -1, -1, -1)
        )

        # Flatten velocity dimension
        u_flat = u_expand.reshape(
            B * self.num_v,
            u_t.shape[1],
            H,
            W
        )

        h_flat = h_shift.reshape(
            B * self.num_v,
            self.hidden_channels,
            H,
            W
        )

        combined = torch.cat([u_flat, h_flat], dim=1)

        gates = self.conv(combined)

        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c_shift.reshape(B * self.num_v,
                                     self.hidden_channels,
                                     H,
                                     W) + i * g

        h_next = o * torch.tanh(c_next)

        # Restore velocity dimension
        c_next = c_next.reshape(
            B,
            self.num_v,
            self.hidden_channels,
            H,
            W
        )

        h_next = h_next.reshape(
            B,
            self.num_v,
            self.hidden_channels,
            H,
            W
        )

        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):

        h = torch.zeros(
            batch_size,
            self.num_v,
            self.hidden_channels,
            height,
            width,
            device=device
        )

        c = torch.zeros_like(h)

        return h, c


class Seq2SeqFEConvLSTM(nn.Module):

    def __init__(self,
                 input_channels,
                 hidden_channels,
                 output_channels=None,
                 kernel_size=3,
                 v_range=0,
                 pool_type='max',
                 pool_temperature=1.0,
                 velocity_mixing=False,
                 decoder_conv_layers=1):

        super().__init__()

        self.pool_type = pool_type
        # Plain attribute (not a Parameter/buffer) so it can be annealed
        # from the training loop: higher -> closer to max (selective),
        # lower -> closer to mean (every channel gets gradient).
        self.pool_temperature = pool_temperature

        self.output_channels = (
            output_channels
            if output_channels is not None
            else input_channels
        )

        self.cell = FEConvLSTMCell(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            v_range=v_range,
            velocity_mixing=velocity_mixing
        )

        self.hidden_channels = hidden_channels
        self.num_v = self.cell.num_v

        decoder = []

        for _ in range(decoder_conv_layers):

            decoder += [
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode='circular',
                    bias=True
                ),
                nn.ReLU()
            ]

        decoder += [
            nn.Conv2d(
                hidden_channels,
                self.output_channels,
                kernel_size=3,
                padding=1,
                padding_mode='circular',
                bias=True
            )
        ]

        self.decoder_conv = nn.Sequential(*decoder)

    def pool_velocity(self, h):

        if self.pool_type == 'max':
            return h.max(dim=1)[0]

        elif self.pool_type == 'mean':
            return h.mean(dim=1)

        elif self.pool_type == 'sum':
            return h.sum(dim=1)

        elif self.pool_type == 'softmax':
            # Smooth, per-pixel/per-feature relaxation of max over the
            # velocity channels: every channel receives gradient (weighted
            # by how much it contributes) instead of max's winner-take-all,
            # which starves the newly-correct channel whenever the true
            # velocity changes mid-sequence. temperature -> inf recovers
            # max; -> 0 recovers mean.
            w = torch.softmax(self.pool_temperature * h, dim=1)
            return (w * h).sum(dim=1)

        else:
            return h.max(dim=1)[0]

    def forward(self,
                input_seq,
                pred_len,
                teacher_forcing_ratio=0.0,
                target_seq=None,
                return_hidden=False):

        B, T_in, C, H, W = input_seq.shape

        device = input_seq.device

        h, c = self.cell.init_hidden(
            B,
            H,
            W,
            device
        )

        if return_hidden:

            encoder_hidden = torch.zeros(
                B,
                T_in,
                self.num_v,
                self.hidden_channels,
                H,
                W,
                device=device
            )

            decoder_hidden = torch.zeros(
                B,
                pred_len,
                self.num_v,
                self.hidden_channels,
                H,
                W,
                device=device
            )

        # Encoder
        for t in range(T_in):

            h, c = self.cell(
                input_seq[:, t],
                (h, c)
            )

            if return_hidden:
                encoder_hidden[:, t] = h.detach()

        prev = input_seq[:, -1]

        outputs = []

        # Decoder
        for t in range(pred_len):

            if (
                self.training
                and target_seq is not None
                and random.random() < teacher_forcing_ratio
            ):
                frame = target_seq[:, t]

            else:
                frame = prev.detach()

            h, c = self.cell(
                frame,
                (h, c)
            )

            if return_hidden:
                decoder_hidden[:, t] = h.detach()

            feat = self.pool_velocity(h)

            out = self.decoder_conv(feat)

            outputs.append(out)

            prev = out

        outputs = torch.stack(outputs, dim=1)

        if return_hidden:
            return outputs, encoder_hidden, decoder_hidden

        return outputs