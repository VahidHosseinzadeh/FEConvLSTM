import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 kernel_size=3,
                 bias=True):

        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        padding = (
            kernel_size[0] // 2,
            kernel_size[1] // 2
        )

        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode='circular',
            bias=bias
        )

        # Initialize forget gate bias
        if bias:
            nn.init.constant_(
                self.conv.bias[hidden_dim:2 * hidden_dim],
                1.0
            )

    def forward(self, x, state):

        h, c = state

        combined = torch.cat([x, h], dim=1)

        gates = self.conv(combined)

        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):

        h = torch.zeros(
            batch_size,
            self.hidden_dim,
            height,
            width,
            device=device
        )

        c = torch.zeros_like(h)

        return h, c


class ConvLSTM(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 kernel_size=3,
                 bias=True,
                 batch_first=True):

        super().__init__()

        self.batch_first = batch_first

        self.cell = ConvLSTMCell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            bias=bias
        )

    def forward(self, x, hidden_state=None):

        # Input:
        # batch_first=True  -> (B,T,C,H,W)
        # batch_first=False -> (T,B,C,H,W)

        if not self.batch_first:
            x = x.permute(1, 0, 2, 3, 4)

        B, T, C, H, W = x.shape

        if hidden_state is None:
            h, c = self.cell.init_hidden(
                batch_size=B,
                height=H,
                width=W,
                device=x.device
            )
        else:
            h, c = hidden_state

        outputs = []

        for t in range(T):

            h, c = self.cell(
                x[:, t],
                (h, c)
            )

            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)

        return outputs, (h, c)