"""
Make MEConvLSTM trainable on Apple GPU by swapping the warp for an integer shift.

The problem
-----------
`MEConvLSTMCell.warp` uses `F.grid_sample`, and MPS implements no
`aten::grid_sampler_2d_backward`. Any LOCAL training run of MEConvLSTM on an M-series
Mac therefore dies in `.backward()`. `PYTORCH_ENABLE_MPS_FALLBACK=1` makes it run by
routing that op through the CPU, but measured ~2.5x slower overall
(9.9 vs 3.9 s/batch at 64px / hidden 32 / batch 32).

The fix
-------
For INTEGER velocities a warp is just a periodic index shift, which `gather` expresses
exactly -- and `gather` backpropagates fine on MPS:

    sy = (yy - dy) % H ;  sx = (xx - dx) % W
    x.reshape(B*K, C, H*W).gather(2, (sy*W + sx).expand(-1, C, -1))

This is not an approximation. Measured against `torch.roll` -- which is exact and
unambiguous for an integer shift -- the gather warp is bit-exact for all 81 integer
velocities in [-4, 4]^2, while `grid_sample` is off by ~1.5e-6 at unit scale. That
residual is grid_sample's, not the gather warp's: it scales linearly with the data
(relative error ~4e-7, float32 epsilon territory) and collapses to 5e-15 in float64,
because normalising the coordinate to [-1, 1] and back does not land exactly on the
grid point, so bilinear interpolation blends in a ~4e-7 sliver of the neighbour.

So swapping in the gather warp makes the integer-velocity path MORE accurate, not
less. `verify_equivalence()` checks it against `torch.roll` and demands exactness,
rather than checking it against the less accurate of the two.

Why it is safe
--------------
Every velocity that reaches the warp during TRAINING is a phase-correlation argmax,
hence an integer by construction. Fractional velocities only arise from a learned
dynamics head at evaluation, where no backward pass is needed -- so the patched warp
dispatches on the velocity itself and falls through to the original `grid_sample` path
whenever anything fractional shows up. Nothing silently changes numerics.

Cost: the dispatch test `torch.all(u == u.round())` forces a device sync each call.
That is real but small next to the warp itself, and it is what keeps the fallback
honest rather than assumed.

Usage
-----
    from mps_integer_warp import enable_integer_shift_warp
    enable_integer_shift_warp()            # no-op unless the device needs it
"""
import torch
import torch.nn as nn

from velocity_model_based_MEConvLSTM_model import MEConvLSTMCell

_ORIGINAL_WARP = MEConvLSTMCell.warp
_PATCHED = False


def _integer_shift_warp(self, x, u):
    """
    Periodic integer shift by gather. Exact for integral u.

    x : (B, K, C, H, W)
    u : (B, K, 2)  [vx, vy]
    """
    B, K, C, H, W = x.shape
    u = u.reshape(B * K, 2)
    dx = u[:, 0].round().long()
    dy = u[:, 1].round().long()

    yy, xx = torch.meshgrid(
        torch.arange(H, device=x.device), torch.arange(W, device=x.device),
        indexing="ij")

    # grid_sample path samples source coordinate (y - dy) mod H, (x - dx) mod W;
    # this reproduces that indexing exactly.
    sy = (yy.unsqueeze(0) - dy[:, None, None]) % H
    sx = (xx.unsqueeze(0) - dx[:, None, None]) % W

    idx = (sy * W + sx).reshape(B * K, 1, H * W).expand(-1, C, -1)
    out = x.reshape(B * K, C, H * W).gather(2, idx)
    return out.view(B, K, C, H, W)


def _dispatching_warp(self, x, u):
    """Integer shift when every velocity is integral, original warp otherwise."""
    if bool(torch.all(u == u.round())):
        return _integer_shift_warp(self, x, u)
    return _ORIGINAL_WARP(self, x, u)


def verify_equivalence(device="cpu", v_max=4, seed=0, grid_sample_atol=1e-4):
    """
    Assert the gather warp is EXACTLY a periodic shift, for every integer velocity
    in [-v_max, v_max]^2.

    The reference is `torch.roll`, not `grid_sample`: roll is exact for an integer
    shift, whereas grid_sample carries ~4e-7 relative error from its coordinate
    round-trip. Checking against the less accurate of the two would only ever
    measure grid_sample's noise floor, and would pass a genuinely wrong index map
    that happened to land inside it.

    Returns (max error vs roll, max deviation from grid_sample). The second is
    reported, not asserted tightly -- it is grid_sample's error, not ours.
    """
    torch.manual_seed(seed)
    cell = MEConvLSTMCell(1, 3).to(device)
    x = torch.randn(2, 3, 4, 16, 16, device=device)

    worst_roll = 0.0
    for vx in range(-v_max, v_max + 1):
        for vy in range(-v_max, v_max + 1):
            u = torch.zeros(2, 3, 2, device=device)
            u[..., 0], u[..., 1] = vx, vy
            got = _integer_shift_warp(cell, x, u)
            ref = torch.roll(x, shifts=(vy, vx), dims=(3, 4))
            worst_roll = max(worst_roll, (got - ref).abs().max().item())

    if worst_roll != 0.0:
        raise AssertionError(
            f"integer-shift warp is not an exact periodic shift: max |err| vs "
            f"torch.roll = {worst_roll:.2e}. The index map is wrong.")

    u = torch.randint(-v_max, v_max + 1, (2, 3, 2), device=device).float()
    gs_dev = (_ORIGINAL_WARP(cell, x, u) - _integer_shift_warp(cell, x, u)).abs().max().item()
    if gs_dev > grid_sample_atol:
        raise AssertionError(
            f"deviation from grid_sample is {gs_dev:.2e}, far above its ~1e-6 noise "
            f"floor -- something other than float precision differs.")
    return worst_roll, gs_dev


def enable_integer_shift_warp(force=False, verify=True, device=None):
    """
    Patch MEConvLSTMCell.warp. No-op unless MPS is the active device, or force=True.

    Returns (err_vs_roll, deviation_from_grid_sample), or None when no patch
    was applied.
    """
    global _PATCHED
    if _PATCHED:
        return None

    needs_it = force or (device == "mps") or (
        device is None and torch.backends.mps.is_available())
    if not needs_it:
        return None

    result = verify_equivalence() if verify else None
    MEConvLSTMCell.warp = _dispatching_warp
    _PATCHED = True
    return result


def disable_integer_shift_warp():
    """Restore the original grid_sample warp."""
    global _PATCHED
    MEConvLSTMCell.warp = _ORIGINAL_WARP
    _PATCHED = False
