"""
Classify a MOTION-DEFINED digit from the recurrent state.

The task
--------
Show the model T context frames of Common-Fate Moving MNIST and ask which digit
is there. No frame contains the digit: figure and background are the same
band-limited noise, and only a velocity difference separates them. A per-frame
CNN has nothing to classify.

What the three backbones have to work with is therefore their VELOCITY
STRUCTURE, and they differ in exactly that:

    lstm   : one hidden state, no transport            -> the control. If this
             trains above chance, something is leaking and the dataset is wrong.
    felstm : (2R+1)^2 hidden copies, each transported at a FIXED lattice
             velocity. The copy whose velocity matches the figure accumulates it
             in register. Needs v_range >= the figure's max speed or no copy can
             represent it.
    melstm : K slots, each transported at a TRACKED velocity. Cheaper than
             felstm's whole lattice, but only works if a slot actually locks onto
             the figure rather than spending both slots on the background.

All three end the encoder with a state of shape (B, V, C, H, W) -- V is 1,
(2R+1)^2 and K respectively -- so one head serves all three and the comparison
is about the transport structure rather than about head capacity.

Why max-pooling over the velocity axis is the wrong default HERE
----------------------------------------------------------------
In ordinary Moving MNIST the digit is bright on black, so the correctly
transported copy simply has the biggest activations and a max over V finds it.

In this dataset every copy sees noise of the same amplitude. What distinguishes
the right velocity is not magnitude but COHERENCE: the copy transported at the
figure's velocity adds the figure in register frame after frame, so it develops
spatial STRUCTURE, while every other copy averages a drifting texture toward a
smooth blur. Their maxima stay comparable -- a noise field has large values too
-- so `max` reduces to picking the luckiest spike.

`attention` (the default) scores each velocity copy from its per-channel spatial
mean AND spatial standard deviation, then takes a softmax over V. The standard
deviation is the coherence signal, and it is the one statistic that actually
separates "accumulated in register" from "averaged into mush". The weights are
returned, so you can check which velocity the model chose against the ground
truth rather than assuming it chose sensibly.
"""
import torch
import torch.nn as nn

from channel_based_FEConvLSTM_model import Seq2SeqFEConvLSTM
from velocity_model_based_MEConvLSTM_model import Seq2SeqMEConvLSTM
from velocity_predictor_model import PhaseCorrelation


# --------------------------------------------------------------- velocity pooling
class VelocityPool(nn.Module):
    """
    (B, V, C, H, W) -> (B, C_out, H, W), reducing the velocity axis.

    mode
    ----
    'attention' : learned softmax over V, scored from each copy's per-channel
                  spatial (mean, std). The std term is what lets it prefer the
                  copy that accumulated coherently. C_out = C.
    'max'       : elementwise max over V. The cheap default everywhere else in
                  this repo, and the one this dataset is built to defeat -- kept
                  so that claim is measurable rather than asserted.
    'mean'      : elementwise mean. Dilutes the one informative copy by V.
    'concat'    : keep every copy, C_out = V * C. The most expressive option and
                  the least structured: at felstm's v_range=2 that is 25*C
                  channels into the head, so the head, not the transport, may end
                  up doing the work.
    """

    def __init__(self, mode, n_velocities, hidden_channels, score_hidden=32,
                 temperature=1.0):
        super().__init__()
        if mode not in ("attention", "max", "mean", "concat"):
            raise ValueError(f"unknown velocity_pool {mode!r}")
        self.mode = mode
        self.n_velocities = n_velocities
        self.temperature = temperature
        self.out_channels = hidden_channels * (n_velocities if mode == "concat" else 1)

        if mode == "attention":
            # Input is [spatial mean, spatial std] per feature channel.
            self.score = nn.Sequential(
                nn.Linear(2 * hidden_channels, score_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(score_hidden, 1),
            )

    def forward(self, h):
        """h: (B, V, C, H, W) -> (features, weights or None)."""
        if self.mode == "max":
            return h.amax(dim=1), None
        if self.mode == "mean":
            return h.mean(dim=1), None
        if self.mode == "concat":
            B, V, C, H, W = h.shape
            return h.reshape(B, V * C, H, W), None

        mu = h.mean(dim=(-2, -1))                       # (B, V, C)
        sd = h.std(dim=(-2, -1))                        # (B, V, C) <- coherence
        w = torch.softmax(
            self.score(torch.cat([mu, sd], dim=-1)) / self.temperature, dim=1)
        return (h * w[..., None, None]).sum(dim=1), w.squeeze(-1)   # (B,C,H,W), (B,V)


# ------------------------------------------------------------------------- head
class ConvClassifierHead(nn.Module):
    """
    (B, C, H, W) -> (B, n_classes).

    Circular padding and global pooling, both on purpose: the canvas is a torus
    and the figure's position is uniform over it, so the head must be
    translation-invariant or it can only learn where digits tend to sit. Circular
    convolutions plus a global pool give exactly that invariance on the torus.

    The global stage concatenates average AND max pooling. Spatial max here is
    not the mistake the module docstring warns about -- that was max over the
    VELOCITY axis, across copies with equal-amplitude noise. This max is over
    space, after learned convolutions, and it is the right tool for an object
    covering ~3% of the frame: a pure average would drown the figure in the
    background it shares its statistics with.
    """

    def __init__(self, in_channels, n_classes=10, channels=64, n_blocks=3,
                 mlp_hidden=128, dropout=0.0):
        super().__init__()
        layers, ch = [], in_channels
        for _ in range(n_blocks):
            layers += [
                nn.Conv2d(ch, channels, 3, stride=2, padding=1,
                          padding_mode="circular", bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ]
            ch = channels
        self.conv = nn.Sequential(*layers)
        self.mlp = nn.Sequential(
            nn.Linear(2 * channels, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(mlp_hidden, n_classes),
        )

    def forward(self, x):
        f = self.conv(x)
        pooled = torch.cat([f.mean(dim=(-2, -1)), f.amax(dim=(-2, -1))], dim=1)
        return self.mlp(pooled)


# ------------------------------------------------------------------- full model
class MotionDigitClassifier(nn.Module):
    """
    One of the three backbones' encoders, a velocity-axis pool, and a conv head.

    The backbone's decoder is constructed (so the recurrent cell is built exactly
    as in the prediction experiments and the comparison stays honest) but never
    run. `head_parameters()` and `backbone_encoder_parameters()` let the training
    script hand the optimizer only the parameters this task actually uses.
    """

    def __init__(self, model="melstm", hidden_channels=32, kernel_size=3,
                 v_range=2, n_slots=2, n_classes=10,
                 velocity_pool="attention", pool_temperature=1.0,
                 velocity_source="frame_pair",
                 head_channels=64, head_blocks=3, head_mlp_hidden=128,
                 head_dropout=0.0, input_channels=1):
        super().__init__()
        self.model = model
        self.hidden_channels = hidden_channels
        self.velocity_source = velocity_source

        if model in ("lstm", "felstm"):
            if model == "lstm":
                v_range = 0
            self.backbone = Seq2SeqFEConvLSTM(
                input_channels=input_channels, hidden_channels=hidden_channels,
                kernel_size=kernel_size, v_range=v_range, pool_type="max",
                decoder_conv_layers=1, decoder_channels=hidden_channels)
            self.n_velocities = self.backbone.num_v
        elif model == "melstm":
            self.backbone = Seq2SeqMEConvLSTM(
                input_channels=input_channels, hidden_channels=hidden_channels,
                kernel_size=kernel_size, n_slots=n_slots, slot_reduce="max",
                decoder_layers=1, decoder_channels=hidden_channels)
            self.n_velocities = n_slots
        else:
            raise ValueError(f"unknown model {model!r}")

        if velocity_source not in ("tracked", "frame_pair"):
            raise ValueError(f"unknown velocity_source {velocity_source!r}")
        # Own module: n_modes must equal the slot count, which the backbone's
        # bootstrap module only coincidentally matches.
        self._frame_pair_pc = PhaseCorrelation(n_modes=self.n_velocities)

        self.pool = VelocityPool(velocity_pool, self.n_velocities,
                                 hidden_channels, temperature=pool_temperature)
        self.head = ConvClassifierHead(
            self.pool.out_channels, n_classes=n_classes, channels=head_channels,
            n_blocks=head_blocks, mlp_hidden=head_mlp_hidden, dropout=head_dropout)

    # ------------------------------------------------------------------
    @staticmethod
    def _match_to_slots(cand, v_prev):
        """
        Permute candidate velocities so candidate k is the one nearest slot k's
        previous velocity, greedily and without replacement.

        Phase-correlation peaks come out ordered by score, and on this data the
        ordering is unstable: the figure's peak trades rank with noise bumps from
        frame to frame. Assigning by score would therefore shuffle which slot
        holds the figure every few steps, and a slot that keeps changing which
        motion it follows accumulates nothing. Matching to the previous velocity
        is what gives a slot a persistent IDENTITY.
        """
        B, K, _ = cand.shape
        cost = (cand[:, :, None, :] - v_prev[:, None, :, :]).abs().sum(-1)  # (B,cand,slot)
        out = torch.zeros_like(cand)
        taken = torch.zeros(B, K, dtype=torch.bool, device=cand.device)
        rows = torch.arange(B, device=cand.device)
        for k in range(K):
            c = cost[:, :, k].masked_fill(taken, float("inf"))
            j = c.argmin(dim=1)
            out[:, k] = cand[rows, j]
            taken[rows, j] = True
        return out

    def _encode_melstm_frame_pair(self, seq):
        """
        Encoder that reads slot velocities from RAW FRAME PAIRS instead of
        self-tracking each slot's hidden state.

        Why this exists. MEConvLSTM's own protocol estimates a slot's velocity by
        correlating its hidden state against the next frame. That works when the
        object is bright on black: the slot's h is a clean template. Here every
        layer is the same band-limited noise, h is a smoothed nonlinear function
        of mostly-background texture, and the correlation peak is unreliable --
        measured on this data, after 15 frames NO slot held either the figure or
        the background velocity, and the slots collapsed from 4 distinct
        velocities onto 2.0 on average. Slot collapse is self-sustaining: two
        slots at the same velocity get the same warp, the same input and the same
        weights, so they stay merged.

        Phase correlation on the raw frame pair does not have this problem on the
        same data: v_bg 100% at every K, v_fg 65.6% / 83.7% / 97.8% at K = 2/4/6.
        So take the velocities from there and keep slot identity with
        `_match_to_slots`.

        The prediction model is deliberately untouched; this lives here because it
        is a property of THIS task's data, not a correction to MEConvLSTM.
        """
        cell = self.backbone.cell
        B, T, C, H, W = seq.shape
        K = self.n_velocities

        h, c = cell.init_hidden(B, K, H, W, seq.device, seq.dtype)
        v = torch.zeros(B, K, 2, device=seq.device, dtype=seq.dtype)
        vels = []

        for t in range(T):
            if t > 0:
                with torch.no_grad():
                    cand, _ = self._frame_pair_pc(seq[:, t - 1], seq[:, t])
                cand = cand.to(seq.dtype)
                v = cand if t == 1 else self._match_to_slots(cand, v)
            h, c = cell(seq[:, t], h, c, v)
            if t > 0:
                vels.append(v.detach())

        return h, c, vels

    def encode(self, seq):
        """seq (B, T, C, H, W) -> h_T (B, V, Ch, H, W), velocities or None."""
        if self.model == "melstm":
            if self.velocity_source == "frame_pair":
                h, _, vels = self._encode_melstm_frame_pair(seq)
            else:
                h, _, _, vels, _ = self.backbone.encode(seq)
            v = torch.stack(vels, dim=1) if vels else None    # (B, T-1, K, 2)
            return h, v
        h, _, _ = self.backbone.encode(seq)
        return h, None

    def forward(self, seq, return_aux=False):
        h, velocities = self.encode(seq)
        features, weights = self.pool(h)
        logits = self.head(features)
        if not return_aux:
            return logits
        return logits, {"velocities": velocities, "pool_weights": weights}

    # ------------------------------------------------------------------
    def backbone_encoder_parameters(self):
        """Everything in the backbone except the unused decoder."""
        dec = self.backbone.decoder if self.model == "melstm" else self.backbone.decoder_conv
        dec_ids = {id(p) for p in dec.parameters()}
        return [p for p in self.backbone.parameters() if id(p) not in dec_ids]

    def head_parameters(self):
        return list(self.pool.parameters()) + list(self.head.parameters())

    def trainable_parameters(self):
        """The parameters this task uses -- the unused decoder is excluded."""
        return self.backbone_encoder_parameters() + self.head_parameters()

    def parameter_report(self):
        n = lambda ps: sum(p.numel() for p in ps)
        enc, hd = n(self.backbone_encoder_parameters()), n(self.head_parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"encoder": enc, "head": hd, "trained": enc + hd,
                "unused_decoder": total - enc - hd,
                "n_velocities": self.n_velocities}


def build_classifier(cfg):
    """
    Construct the classifier a run config describes.

    Accepts the argparse Namespace used during training or the config dict stored
    in the history JSON, mirroring build_model() in train_eval_utils so offline
    evaluation rebuilds the architecture rather than re-specifying it by hand.
    """
    get = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    return MotionDigitClassifier(
        model=get("model", "melstm"),
        hidden_channels=get("hidden_size", 32),
        kernel_size=get("kernel_size", 3),
        v_range=get("v_range", 2),
        n_slots=get("num_vel_modes", 2),
        n_classes=get("n_classes", 10),
        velocity_pool=get("velocity_pool", "attention"),
        pool_temperature=get("pool_temperature", 1.0),
        velocity_source=get("velocity_source", "frame_pair"),
        head_channels=get("head_channels", 64),
        head_blocks=get("head_blocks", 3),
        head_mlp_hidden=get("head_mlp_hidden", 128),
        head_dropout=get("head_dropout", 0.0),
    )
