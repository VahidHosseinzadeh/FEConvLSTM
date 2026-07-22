import torch
import torch.nn as nn


class FEConvLSTMCell(nn.Module):

    def __init__(self,
                 input_channels,
                 hidden_channels,
                 kernel_size=3,
                 v_range=0):

        super().__init__()

        self.hidden_channels = hidden_channels

        self.v_list = [
            (vx, vy)
            for vx in range(-v_range, v_range + 1)
            for vy in range(-v_range, v_range + 1)
        ]

        self.num_v = len(self.v_list)

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
                 decoder_conv_layers=1):

        super().__init__()

        self.pool_type = pool_type

        self.output_channels = (
            output_channels
            if output_channels is not None
            else input_channels
        )

        self.cell = FEConvLSTMCell(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            v_range=v_range
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

        else:
            return h.max(dim=1)[0]

    def forward(self,
                input_seq,
                pred_len,
                return_states=False):

        B, T_in, C, H, W = input_seq.shape

        device = input_seq.device

        h, c = self.cell.init_hidden(
            B,
            H,
            W,
            device
        )

        h_states = [] if return_states else None

        # Encoder
        for t in range(T_in):

            h, c = self.cell(
                input_seq[:, t],
                (h, c)
            )

            if return_states:
                h_states.append(h.mean(dim=2).detach())

        prev = input_seq[:, -1]

        outputs = []

        # Decoder
        for t in range(pred_len):

            frame = prev.detach()

            h, c = self.cell(
                frame,
                (h, c)
            )

            if return_states:
                h_states.append(h.mean(dim=2).detach())

            feat = self.pool_velocity(h)

            out = self.decoder_conv(feat)

            outputs.append(out)

            prev = out

        outputs = torch.stack(outputs, dim=1)

        if return_states:
            # (B, T_in+pred_len, num_v, H, W) -- one channel-mean map per
            # candidate (vx, vy) slot, same reduction/layout as
            # Seq2SeqMEConvLSTM's return_states so the two share a plotting
            # path (see log_state_evolution).
            return outputs, {"h": torch.stack(h_states, dim=1)}

        return outputs