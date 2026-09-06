r"""
Splitting the pixel loss into a shape part and a location part.

The problem this exists for
---------------------------
Squared error on sparse bright-on-black frames has a degenerate minimum at
"predict nothing". Measured on this dataset (64x64, two digits, ~6.5% of pixels
lit), with a 28 px digit:

    digit displaced by  2 px : 0.047
    digit displaced by  5 px : 0.117
    digit displaced by 10 px : 0.234   <-- identical to predicting NOTHING
    digit displaced by 20 px : 0.469
    predict nothing at all   : 0.234

Past roughly half a digit width, a correctly-drawn but misplaced digit is worse
than a blank frame: the loss pays twice, once for the digit that is missing and
once for the one that is spurious. A rollout whose velocity is slightly wrong
walks straight into a regime where the safest thing the model can do is stop
drawing. That is not hypothetical here -- one run plateaued at
train == val == 0.0921, against 0.0904 for the all-zeros predictor.

The decomposition
-----------------
Write the DFT of each frame in polar form, X(k) = |X(k)| e^{i phi_x(k)}. Then
at every frequency k, exactly and with no approximation:

    |X - Y|^2  =  |X|^2 + |Y|^2 - 2 Re(X conj(Y))
               =  |X|^2 + |Y|^2 - 2 |X||Y| cos(phi_x - phi_y)
               = (|X| - |Y|)^2  +  2|X||Y| [1 - cos(phi_x - phi_y)]
                 \____________/    \______________________________/
                     SHAPE                    LOCATION

Both terms are non-negative, and by Parseval, summing over k and dividing by
H*W returns the pixel-space sum of squares -- so shape + location == MSE
exactly. There is a test pinning that to float tolerance.

Why the names are justified -- and where they stop being justified
------------------------------------------------------------------
Under a pure translation the Fourier shift theorem gives
Y(k) = e^{-i <k, tau>} X(k), so |Y| = |X| at every frequency. The SHAPE term is
therefore exactly ZERO for any translation, however large, and the entire error
lands in the LOCATION term. That direction is exact.

The converse is not, and it is the main caveat:

  * Phase encodes the position of every feature, not just a global offset, so
    two DIFFERENT digits at the same place also differ in phase. The location
    term picks up shape change too -- it is "everything that is not a magnitude
    difference", not "translation".
  * Magnitude is not a complete shape descriptor: distinct images can share a
    magnitude spectrum (the phase-retrieval ambiguity), and |X| is sensitive to
    scale and rotation, which are not translations.

So read the split as translation-INVARIANT part versus the rest. That weaker
reading is still exactly what is needed here, because the failure mode being
priced is "the shape is right and the place is wrong".

Two further limits specific to this dataset. With several digits the split is
global, so it cannot attribute a displacement to one object -- that is what the
K velocity slots exist for. And phase is ill-conditioned wherever |X| is small;
the 2|X||Y| factor damps that automatically, which is exactly why the location
term below is computed from Re(X conj(Y)) rather than by extracting angles.

What it buys
------------
Weighting the two terms differently changes what the model is rewarded for.
Raising the shape weight penalises blur and blankness directly: a blank
prediction has |Y| = 0 at every frequency, so its shape term is the full
sum |X|^2 and cannot be hidden by getting the location roughly right. That is
the degenerate minimum above, priced properly.

At weights (1, 1) this IS the MSE, so it strictly generalises the existing
objective and can be switched on without changing what is optimised.
"""

import torch
import torch.nn as nn


class FourierShapePhaseLoss(nn.Module):
    """
    MSE split into a translation-invariant "shape" term and a "location" term.

    forward(pred, target) -> (loss, parts), with parts a dict of detached
    scalars for logging -- "shape", "location", "mse" -- all in pixel-MSE units
    so they are directly comparable with nn.MSELoss.

    shape_weight, location_weight : (1.0, 1.0) reproduces plain MSE.
    """

    def __init__(self, shape_weight=1.0, location_weight=1.0):
        super().__init__()
        self.shape_weight = shape_weight
        self.location_weight = location_weight

    @staticmethod
    def decompose(pred, target):
        """(shape, location) as scalars in pixel-MSE units; they sum to MSE.

        The full fft2 is used rather than rfft2 so that Parseval holds without
        re-weighting the half-spectrum's interior columns. The factor-of-two
        cost is negligible beside the model itself.
        """
        X = torch.fft.fft2(target.float())
        Y = torch.fft.fft2(pred.float())
        mx, my = X.abs(), Y.abs()

        shape = (mx - my) ** 2
        # 2|X||Y|(1 - cos dphi) == 2(|X||Y| - Re(X conj(Y))) -- an identity.
        # Used in this form deliberately: it never extracts an angle, so it is
        # well behaved where the magnitude is near zero. The clamp only absorbs
        # float round-off; the quantity is non-negative by construction.
        location = (2.0 * (mx * my - (X * torch.conj(Y)).real)).clamp_min(0.0)

        # Parseval for the unnormalised DFT: sum_k |F|^2 = H*W * sum_n |f|^2.
        # Dividing by H*W recovers the pixel-space sum of squares; dividing by
        # numel then makes it a per-pixel mean, i.e. nn.MSELoss units.
        hw = X.shape[-2] * X.shape[-1]
        denom = hw * pred.numel()
        return shape.sum() / denom, location.sum() / denom

    def forward(self, pred, target):
        shape, location = self.decompose(pred, target)
        loss = self.shape_weight * shape + self.location_weight * location
        return loss, {"shape": shape.detach(),
                      "location": location.detach(),
                      "mse": (shape + location).detach()}
