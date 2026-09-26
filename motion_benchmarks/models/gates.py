"""
Forget-gate initialisation: set the memory time constant, optionally two of them.

With c_{t+1} = f * c_t + i * g and f ~= lam, the memory decays as lam^t, i.e. with time constant
tau = -1 / ln(lam). Biasing the gate pre-activation by b_f = logit(lam) = ln(lam / (1 - lam))
starts the cell at that time constant:

    tau (frames)   lam     b_f
         2        0.607    0.44
         3        0.717    0.93
         4        0.779    1.26
        20        0.951    2.97

Both cells in this repo initialise b_f = 1.0 (tau ~= 3.2). For articulating subjects a long memory
averages the limbs away (claude/common_fate_real_video.md section 7); for long-lived rain and
clouds a long memory is what you want. `two_timescales` splits the hidden channels: a fraction
starts at the long constant (identity/template), the rest at the short one (instantaneous pose).
The gate is input-dependent, so this only BIASES the start; it consumes no RNG, so the rest of
the initialisation is unchanged.
"""
import math

import torch


def bias_for_tau(tau):
    lam = math.exp(-1.0 / float(tau))
    return math.log(lam / (1.0 - lam))


@torch.no_grad()
def set_forget_bias(conv, hidden_channels, short=None, long=None, long_fraction=0.5):
    """
    conv: the gate convolution of an MEConvLSTMCell / FEConvLSTMCell (gate order i, f, o, g, so
    the forget bias is conv.bias[hidden : 2 * hidden]).

    short only           -> every channel gets `short`
    short and long       -> the first round(long_fraction * hidden) channels get `long`
    Values are BIASES (use bias_for_tau to convert a time constant).
    """
    if conv.bias is None or short is None:
        return
    f = conv.bias[hidden_channels:2 * hidden_channels]
    f.fill_(float(short))
    if long is not None:
        n_long = int(round(long_fraction * hidden_channels))
        f[:n_long] = float(long)
