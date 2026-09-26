#!/usr/bin/env python
"""
3D Rayleigh-Benard convection in Dedalus v3 -- the setup of Fromme et al. (arXiv:2505.13569):

    div u = 0
    dt u + (u . grad) u = -grad p + sqrt(Pr / Ra) lap u + T e_z
    dt T + u . grad T   = (Ra Pr)^(-1/2) lap T

on (0, 2 pi)^2 x (-1, 1), periodic horizontally, no-slip walls at z = -1, +1 held at T_bottom
(default 1) and T_top (default 0). Ra = 2500, Pr = 0.7. Snapshots every 0.5 time units over
t in [100, 300] -> 400 per run; Fromme et al. used 100 runs split 60 / 20 / 20.

Dedalus uses Fourier x Fourier x Chebyshev; the output is resampled (spectrally, by barycentric
interpolation in z) to a UNIFORM vertical grid of cell centres, 32 levels by default, i.e. their
48 x 48 x 32 finite-volume grid. Written incrementally (one snapshot at a time), so memory stays
small; ~470 MB per full run at float32 -- use --nz_out 16 or 8 to shrink it if you will only use a
few heights (train_motion.py --rbc_z_stride subsamples heights anyway).

Note on units: with these equations the length unit is whatever makes the layer height 2, so the
Rayleigh number based on the full layer height is 8 x Ra if the unit is the half-height. The
script reproduces the published equations; it does not reinterpret them.

Output layout (read by datasets/fluids.py RBC3DSource after merge_runs.py):
    /fields  float32 (n_snap, 4, nz_out, Ny, Nx), channels T, u, v, w; attrs carry Ra, Pr,
             kappa, nu, Lx, Ly, Lz, dt_snap, t_start, bottom_T, top_T, seed, z (levels)

Usage (serial; one run per call -- use a Slurm array over --seed, see slurm/rbc3d_dedalus.sbatch)
    python -m motion_benchmarks.physics.rbc3d_dedalus --seed 0 --out rbc_runs/run_0000.h5
    # 1-minute smoke test:
    python -m motion_benchmarks.physics.rbc3d_dedalus --seed 0 --out /tmp/t.h5 \
        --Nx 16 --Ny 16 --Nz 16 --nz_out 8 --t_start 2 --t_end 4

Install: conda install -c conda-forge dedalus   (pip needs FFTW-MPI headers: libfftw3-mpi-dev)
"""
import argparse
import logging
import time

import numpy as np

logger = logging.getLogger("rbc3d_dedalus")


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--Ra", type=float, default=2500.0)
    ap.add_argument("--Pr", type=float, default=0.7)
    ap.add_argument("--Nx", type=int, default=48)
    ap.add_argument("--Ny", type=int, default=48)
    ap.add_argument("--Nz", type=int, default=32, help="Chebyshev modes")
    ap.add_argument("--nz_out", type=int, default=32, help="uniform output levels")
    ap.add_argument("--Lx", type=float, default=2 * np.pi)
    ap.add_argument("--Ly", type=float, default=2 * np.pi)
    ap.add_argument("--Lz", type=float, default=2.0)
    ap.add_argument("--bottom_T", type=float, default=1.0)
    ap.add_argument("--top_T", type=float, default=0.0)
    ap.add_argument("--t_start", type=float, default=100.0)
    ap.add_argument("--t_end", type=float, default=300.0)
    ap.add_argument("--dt_snap", type=float, default=0.5)
    ap.add_argument("--max_dt", type=float, default=0.05)
    ap.add_argument("--cfl_safety", type=float, default=0.5)
    ap.add_argument("--noise", type=float, default=1e-3)
    ap.add_argument("--timestepper", default="RK222", choices=["RK222", "RK443", "SBDF2"])
    return ap.parse_args(argv)


def z_resampler(z_nodes, z_out):
    """Linear map (Nz_out x Nz) from values at the Chebyshev nodes to values at z_out.
    Barycentric Lagrange interpolation on the nodes, which is spectrally exact for the
    polynomial Dedalus represents (stable for Chebyshev-distributed nodes)."""
    z = np.asarray(z_nodes, dtype=np.float64)
    w = np.ones_like(z)
    for j in range(len(z)):
        d = z[j] - np.delete(z, j)
        w[j] = 1.0 / np.prod(d * (2.0 / (z.max() - z.min())))     # scaled to avoid overflow
    M = np.zeros((len(z_out), len(z)))
    for i, zo in enumerate(z_out):
        diff = zo - z
        hit = np.isclose(diff, 0.0, atol=1e-14)
        if hit.any():
            M[i, np.argmax(hit)] = 1.0
            continue
        t = w / diff
        M[i] = t / t.sum()
    return M


def main(argv=None):
    a = parse(argv)
    import h5py
    import dedalus.public as d3
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    coords = d3.CartesianCoordinates("x", "y", "z")
    dist = d3.Distributor(coords, dtype=np.float64)
    if dist.comm.size != 1:
        raise SystemExit("run serially (one run per process); parallelise over --seed instead")
    z_bot, z_top = -a.Lz / 2, a.Lz / 2
    xb = d3.RealFourier(coords["x"], size=a.Nx, bounds=(0, a.Lx), dealias=3 / 2)
    yb = d3.RealFourier(coords["y"], size=a.Ny, bounds=(0, a.Ly), dealias=3 / 2)
    zb = d3.ChebyshevT(coords["z"], size=a.Nz, bounds=(z_bot, z_top), dealias=3 / 2)

    p = dist.Field(name="p", bases=(xb, yb, zb))
    T = dist.Field(name="T", bases=(xb, yb, zb))
    u = dist.VectorField(coords, name="u", bases=(xb, yb, zb))
    tau_p = dist.Field(name="tau_p")
    tau_T1 = dist.Field(name="tau_T1", bases=(xb, yb))
    tau_T2 = dist.Field(name="tau_T2", bases=(xb, yb))
    tau_u1 = dist.VectorField(coords, name="tau_u1", bases=(xb, yb))
    tau_u2 = dist.VectorField(coords, name="tau_u2", bases=(xb, yb))

    kappa = (a.Ra * a.Pr) ** (-0.5)
    nu = (a.Ra / a.Pr) ** (-0.5)
    x, y, z = dist.local_grids(xb, yb, zb)
    ex, ey, ez = coords.unit_vector_fields(dist)
    lift_basis = zb.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)  # noqa: E731
    grad_u = d3.grad(u) + ez * lift(tau_u1)
    grad_T = d3.grad(T) + ez * lift(tau_T1)
    T_bot, T_top = a.bottom_T, a.top_T

    problem = d3.IVP([p, T, u, tau_p, tau_T1, tau_T2, tau_u1, tau_u2], namespace=locals())
    problem.add_equation("trace(grad_u) + tau_p = 0")
    problem.add_equation("dt(T) - kappa*div(grad_T) + lift(tau_T2) = - u@grad(T)")
    problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - T*ez + lift(tau_u2) = - u@grad(u)")
    problem.add_equation("T(z=z_bot) = T_bot")
    problem.add_equation("u(z=z_bot) = 0")
    problem.add_equation("T(z=z_top) = T_top")
    problem.add_equation("u(z=z_top) = 0")
    problem.add_equation("integ(p) = 0")

    stepper = {"RK222": d3.RK222, "RK443": d3.RK443, "SBDF2": d3.SBDF2}[a.timestepper]
    solver = problem.build_solver(stepper)
    solver.stop_sim_time = a.t_end + 1e-9

    # conduction profile + wall-damped noise (the per-run seed is the only thing that differs)
    T.fill_random("g", seed=a.seed, distribution="normal", scale=a.noise)
    T["g"] *= (z - z_bot) * (z_top - z)
    T["g"] += T_bot + (T_top - T_bot) * (z - z_bot) / a.Lz

    cfl = d3.CFL(solver, initial_dt=a.max_dt / 4, cadence=10, safety=a.cfl_safety,
                 threshold=0.05, max_change=1.5, min_change=0.5, max_dt=a.max_dt)
    cfl.add_velocity(u)

    z_nodes = np.asarray(z).ravel()
    z_out = z_bot + (np.arange(a.nz_out) + 0.5) * a.Lz / a.nz_out
    M = z_resampler(z_nodes, z_out)                    # (nz_out, Nz)
    n_snap = int(round((a.t_end - a.t_start) / a.dt_snap))
    times = a.t_start + a.dt_snap * np.arange(1, n_snap + 1)

    f = h5py.File(a.out, "w")
    d = f.create_dataset("fields", shape=(n_snap, 4, a.nz_out, a.Ny, a.Nx), dtype="float32",
                         chunks=(1, 4, a.nz_out, a.Ny, a.Nx), compression="lzf")
    d.attrs["channel_names"] = np.array([b"T", b"u", b"v", b"w"])
    f.create_dataset("time", data=times)
    for k, v in dict(Ra=a.Ra, Pr=a.Pr, kappa=kappa, nu=nu, Lx=a.Lx, Ly=a.Ly, Lz=a.Lz,
                     dt_snap=a.dt_snap, t_start=a.t_start, bottom_T=T_bot, top_T=T_top,
                     seed=a.seed, Nx=a.Nx, Ny=a.Ny, Nz_cheb=a.Nz, solver="dedalus3").items():
        f.attrs[k] = v
    f.attrs["z"] = z_out

    def snapshot():
        out = []
        for fld in (T, u):
            fld.change_scales(1)
        data = [T["g"], u["g"][0], u["g"][1], u["g"][2]]               # each (Nx, Ny, Nz)
        for g in data:
            g = np.asarray(g)
            gz = np.tensordot(g, M, axes=([2], [1]))                    # (Nx, Ny, nz_out)
            out.append(np.transpose(gz, (2, 1, 0)))                     # (nz_out, Ny, Nx)
        return np.stack(out).astype(np.float32)

    k, t0 = 0, time.time()
    try:
        while solver.proceed and k < n_snap:
            dt = cfl.compute_timestep()
            t_next = times[k]
            if solver.sim_time + dt >= t_next - 1e-12:
                dt = t_next - solver.sim_time                           # land exactly on it
            solver.step(dt)
            if abs(solver.sim_time - t_next) < 1e-9:
                d[k] = snapshot()
                k += 1
                if k % 50 == 0 or k == n_snap:
                    logger.info(f"seed {a.seed}: snapshot {k}/{n_snap} t={solver.sim_time:.2f} "
                                f"iter={solver.iteration} ({time.time() - t0:.0f}s)")
    finally:
        f.attrs["n_written"] = k
        f.close()
    if k < n_snap:
        raise SystemExit(f"stopped after {k}/{n_snap} snapshots")
    logger.info(f"wrote {a.out}")


if __name__ == "__main__":
    main()
