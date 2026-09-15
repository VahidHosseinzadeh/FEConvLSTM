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
import numpy as np
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

    @staticmethod
    def output_channels(mode, n_velocities, hidden_channels):
        """Channel count without building the module, so the head can be sized first."""
        return hidden_channels * (n_velocities if mode == "concat" else 1)

    def __init__(self, mode, n_velocities, hidden_channels, score_hidden=32,
                 temperature=1.0):
        super().__init__()
        if mode not in ("attention", "max", "mean", "concat"):
            raise ValueError(f"unknown velocity_pool {mode!r}")
        self.mode = mode
        self.n_velocities = n_velocities
        self.temperature = temperature
        self.out_channels = self.output_channels(mode, n_velocities, hidden_channels)

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

    Normalisation is BatchNorm, and on this task it is load-bearing rather than a
    convenience. Measured over 14 epochs on train accuracy, which is the one
    number all variants report identically:

        batch  lr 1e-3   0.109 -> 0.230
        group  lr 1e-3   0.104 -> 0.109   (flat)
        group  lr 3e-3   0.103 -> 0.096   (flat)
        none   lr 1e-3   0.110 -> 0.106   (flat)

    Only BatchNorm learns at all. The likely reason is what it normalises OVER:
    per channel across the batch, which removes the component of the activation
    that is common to every sample. Here that common component is the background
    noise field, which dominates the small, low-contrast structure the figure
    contributes -- so batch statistics act as a background subtraction, while
    GroupNorm divides each sample by its own standard deviation (dominated by
    that same background) and leaves the ratio untouched.

    The catch is that BatchNorm's RUNNING statistics, used at eval, lag badly:
    this head's input is an attention-weighted pool of a recurrent state, so the
    distribution moves as both the cell and the attention weights train, and the
    EMA never catches up. Left alone that made val accuracy oscillate between
    chance and 0.94 across epochs while training accuracy rose smoothly. The fix
    is not to abandon BatchNorm but to RECOMPUTE its statistics before each
    evaluation -- see `recompute_bn_stats`.
    """

    def __init__(self, in_channels, n_classes=10, channels=64, n_blocks=3,
                 mlp_hidden=128, dropout=0.0, norm="batch", groups=8):
        super().__init__()
        if norm not in ("group", "batch", "none"):
            raise ValueError(f"unknown head norm {norm!r}")
        self.norm = norm

        def make_norm(c):
            if norm == "batch":
                return nn.BatchNorm2d(c)
            if norm == "group":
                return nn.GroupNorm(min(groups, c), c)
            return nn.Identity()

        layers, ch = [], in_channels
        for _ in range(n_blocks):
            layers += [
                nn.Conv2d(ch, channels, 3, stride=2, padding=1,
                          padding_mode="circular", bias=(norm == "none")),
                make_norm(channels),
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
                 velocity_source="bootstrap",
                 head_channels=64, head_blocks=3, head_mlp_hidden=128,
                 head_dropout=0.0, head_norm="batch", input_channels=1):
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

        if velocity_source not in ("tracked", "frame_pair", "bootstrap"):
            raise ValueError(f"unknown velocity_source {velocity_source!r}")
        # Own modules: n_modes must equal the slot count, which the backbone's
        # bootstrap module only coincidentally matches.
        self._frame_pair_pc = PhaseCorrelation(n_modes=max(1, self.n_velocities))
        self._pc1 = PhaseCorrelation(n_modes=1)

        # The head is built BEFORE the pool, and the pool last of all, so that
        # changing --velocity_pool cannot shift anyone else's initialisation.
        #
        # 'attention' allocates a score MLP and the others do not, so building the
        # pool first made it consume RNG and hand the head different weights. At
        # V=1 that produced two visibly different lstm runs from what is
        # mathematically the SAME model -- softmax over one element is constant
        # 1.0, so max and attention compute the same thing and the score MLP gets
        # exactly zero gradient. One run reached 0.60 val accuracy and another sat
        # at chance purely on that initialisation difference. With this ordering
        # the backbone and the head are bit-identical across pooling modes, and a
        # seed sweep measures the pooling rather than the random stream.
        head_in = VelocityPool.output_channels(
            velocity_pool, self.n_velocities, hidden_channels)
        self.head = ConvClassifierHead(
            head_in, n_classes=n_classes, channels=head_channels,
            n_blocks=head_blocks, mlp_hidden=head_mlp_hidden, dropout=head_dropout,
            norm=head_norm)
        self.pool = VelocityPool(velocity_pool, self.n_velocities,
                                 hidden_channels, temperature=pool_temperature)

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

    @staticmethod
    def _roll_batch(x, v):
        """Per-sample periodic roll of (B, C, H, W) by integer (vx, vy)."""
        B, C, H, W = x.shape
        dx = v[:, 0].round().long()
        dy = v[:, 1].round().long()
        yy, xx = torch.meshgrid(torch.arange(H, device=x.device),
                                torch.arange(W, device=x.device), indexing="ij")
        sy = (yy.unsqueeze(0) - dy[:, None, None]) % H
        sx = (xx.unsqueeze(0) - dx[:, None, None]) % W
        idx = (sy * W + sx).reshape(B, 1, H * W).expand(-1, C, -1)
        return x.reshape(B, C, H * W).gather(2, idx).view(B, C, H, W)

    def _residual_bootstrap(self, x0, x1, blur=1.5):
        """
        Batched residual bootstrap: the MINORITY motion, recovered from under the
        dominant one.

        The dominant phase-correlation peak is the background -- it owns almost
        every pixel. The figure's peak has to compete with the noise floor, and
        measured on this data a plain top-2 correlation finds it only 62.2% of
        the time. So explain the dominant motion away first: warp frame 1 back by
        it, take the unexplained residual, blur it into a soft region mask, and
        re-correlate only that energy. Same measurement, 98.4%.

        That difference is what makes n_slots=2 the right setting rather than an
        aspiration: one slot per real motion, as the scene actually has.

        Torch port of `residual_bootstrap` in common_fate_diagnostics.py, which
        is the single-image numpy reference.
        """
        v1 = self._pc1(x0, x1)[0][:, 0]                    # (B, 2) dominant
        R = (x0 - self._roll_batch(x1, -v1)).abs()

        H, W = x0.shape[-2:]
        ky = torch.fft.fftfreq(H, device=x0.device)[:, None]
        kx = torch.fft.fftfreq(W, device=x0.device)[None, :]
        g = torch.exp(-2 * (np.pi * blur) ** 2 * (ky ** 2 + kx ** 2))
        R = torch.fft.ifft2(torch.fft.fft2(R) * g).real
        R = R / (R.amax(dim=(-2, -1), keepdim=True) + 1e-8)

        # The residual region itself travels at v1, so the window on frame 1 has
        # to be moved there too, or the second correlation compares misaligned
        # supports.
        W1m = self._roll_batch(R, v1)
        a = R * (x0 - x0.mean(dim=(-2, -1), keepdim=True))
        b = W1m * (x1 - x1.mean(dim=(-2, -1), keepdim=True))
        v2 = self._pc1(a, b)[0][:, 0]                      # (B, 2) minority
        return torch.stack([v1, v2], dim=1)                # (B, 2, 2)

    def _candidate_velocities(self, x0, x1):
        """K candidate velocities for one frame pair, per velocity_source."""
        if self.velocity_source != "bootstrap":
            return self._frame_pair_pc(x0, x1)[0]
        cand = self._residual_bootstrap(x0, x1)            # (B, 2, 2)
        if self.n_velocities <= 2:
            return cand[:, :self.n_velocities]
        # More slots than the scene has motions: top up with ordinary peaks.
        extra = self._frame_pair_pc(x0, x1)[0][:, :self.n_velocities - 2]
        return torch.cat([cand, extra], dim=1)

    def _encode_melstm_frame_pair(self, seq, return_states=False):
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
        states = [] if return_states else None

        for t in range(T):
            if t > 0:
                with torch.no_grad():
                    cand = self._candidate_velocities(seq[:, t - 1], seq[:, t])
                cand = cand.to(seq.dtype)
                v = cand if t == 1 else self._match_to_slots(cand, v)
            h, c = cell(seq[:, t], h, c, v)
            if t > 0:
                vels.append(v.detach())
            if return_states:
                states.append(h.mean(dim=2).detach())

        return h, c, vels, states

    def encode(self, seq, return_states=False):
        """
        seq (B, T, C, H, W) -> h_T (B, V, Ch, H, W), velocities, states.

        `states` is (B, T, V, H, W): the per-timestep CHANNEL-MEAN of each
        velocity copy -- the same h.mean(dim=2) reduction the rest of this repo
        visualises, and what MEConvLSTM's own tracker correlates against. It is
        a summary for looking at, not the tensor the head consumes; the head
        gets the full feature channels.
        """
        if self.model == "melstm":
            # Both frame-pair sources share the encoder; they differ only in how
            # _candidate_velocities picks the peaks. Only "tracked" uses the
            # backbone's own hidden-state tracking.
            if self.velocity_source in ("frame_pair", "bootstrap"):
                h, _, vels, states = self._encode_melstm_frame_pair(
                    seq, return_states=return_states)
            else:
                h, _, _, vels, states = self.backbone.encode(
                    seq, return_states=return_states)
            v = torch.stack(vels, dim=1) if vels else None    # (B, T-1, K, 2)
        else:
            h, _, states = self.backbone.encode(seq, return_states=return_states)
            v = None
        st = torch.stack(states, dim=1) if states else None   # (B, T, V, H, W)
        return (h, v, st) if return_states else (h, v)

    def forward(self, seq, return_aux=False):
        h, velocities = self.encode(seq)[:2]
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

    def submodule_report(self):
        """
        Per-submodule parameter counts, in the order they run.

        Worth printing at the top of every run: the three backbones are meant to
        have IDENTICAL trained parameter counts and to differ only in how many
        velocity copies they transport. If these tables ever stop matching, the
        comparison has silently become one about capacity.
        """
        n = lambda m: sum(p.numel() for p in m.parameters())
        dec = self.backbone.decoder if self.model == "melstm" else self.backbone.decoder_conv
        rows = [
            ("backbone.cell", "recurrent, run on every velocity copy", n(self.backbone.cell)),
            ("pool", f"velocity reduce ({self.pool.mode})", n(self.pool)),
            ("head.conv", f"{len(self.head.conv) // 3} blocks, circular, stride 2", n(self.head.conv)),
            ("head.mlp", "avg+max pool -> logits", n(self.head.mlp)),
        ]
        trained = sum(r[2] for r in rows)
        rows.append(("backbone.decoder", "UNUSED in this task, excluded from the optimizer", n(dec)))
        return rows, trained

    def parameter_report(self):
        n = lambda ps: sum(p.numel() for p in ps)
        enc, hd = n(self.backbone_encoder_parameters()), n(self.head_parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"encoder": enc, "head": hd, "trained": enc + hd,
                "unused_decoder": total - enc - hd,
                "n_velocities": self.n_velocities}

    def describe(self):
        """Human-readable architecture + parameter table for the run header."""
        rows, trained = self.submodule_report()
        w = max(len(r[0]) for r in rows)
        lines = [f"architecture : {self.model}  "
                 f"V={self.n_velocities} velocity "
                 f"{'slots' if self.model == 'melstm' else 'copies'}"
                 + (f"  velocity_source={self.velocity_source}"
                    if self.model == "melstm" else "")]
        if self.model == "felstm":
            lines[0] += f"  (lattice v_range covers |v| <= {max(abs(v[0]) for v in self.backbone.cell.v_list)})"
        lines.append("")
        lines.append(f"  {'submodule':<{w}}  {'params':>9}   note")
        lines.append(f"  {'-' * w}  {'-' * 9}   {'-' * 46}")
        for name, note, cnt in rows:
            tag = "" if name != "backbone.decoder" else ""
            lines.append(f"  {name:<{w}}  {cnt:>9,}   {note}{tag}")
        lines.append(f"  {'-' * w}  {'-' * 9}")
        lines.append(f"  {'TRAINED':<{w}}  {trained:>9,}   what the optimizer updates")
        return "\n".join(lines)


@torch.no_grad()
def recompute_bn_stats(model, loader, device, n_batches=50):
    """
    Re-estimate every BatchNorm's running statistics from scratch, on the current
    weights, before evaluating. ("Precise BN", Wu & Johnson 2021.)

    BatchNorm's running mean/var are an exponential moving average collected while
    the weights were still changing. For a head sitting on a RECURRENT state --
    whose activation distribution moves as both the cell and the attention weights
    train -- that EMA never catches up, and eval-mode accuracy is measured with
    statistics the model was never trained under. Symptom: val accuracy bouncing
    between chance and its true value between epochs while training accuracy rises
    smoothly.

    Setting momentum=None makes PyTorch accumulate a CUMULATIVE average instead of
    an EMA, so after n_batches forward passes the statistics are the exact mean and
    variance over those batches, for these weights. Nothing is learned here: no
    gradients, and the parameters are untouched.
    """
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns or n_batches <= 0:
        return 0

    saved = [(bn.momentum, bn.training) for bn in bns]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None            # cumulative average, not an EMA
        bn.train()                    # so the forward pass updates the statistics

    was_training = model.training
    model.eval()                      # everything else stays in eval; only BN collects
    for bn in bns:
        bn.train()

    seen = 0
    for batch in loader:
        if seen >= n_batches:
            break
        model(batch[0].to(device))
        seen += 1

    model.train(was_training)
    for bn, (mom, mode) in zip(bns, saved):
        bn.momentum = mom
        bn.train(mode)
    return seen


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
        velocity_source=get("velocity_source", "bootstrap"),
        head_channels=get("head_channels", 64),
        head_blocks=get("head_blocks", 3),
        head_mlp_hidden=get("head_mlp_hidden", 128),
        head_dropout=get("head_dropout", 0.0),
        head_norm=get("head_norm", "batch"),
    )
