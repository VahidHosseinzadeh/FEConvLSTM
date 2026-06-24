import torch
import torch.nn as nn
import random


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
        
    def shift_tensor(self, x):

        # x:
        # (B, num_v, C, H, W)

        shifted = []

        for i, (vx, vy) in enumerate(self.v_list):

            shifted.append(
                torch.roll(
                    x[:, i],
                    shifts=(vy, vx),
                    dims=(2, 3)
                )
            )

        return torch.stack(shifted, dim=1)

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