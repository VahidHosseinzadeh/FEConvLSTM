"""
motion_benchmarks -- experiments beyond Moving MNIST for ConvLSTM / FEConvLSTM / MEConvLSTM.

Everything here is ADDITIVE. The models and training code in `moving_mnist/` are reused, never
modified in a way that changes their defaults, so every existing experiment reproduces exactly.

Layout
------
common/     phase correlation (alpha-whitened, sub-pixel, windowed, gated), sub-pixel shifts,
            continuous velocity schedules, random fields, metrics (MSE/CSI/FSS/spectra/Nusselt),
            persistence baselines, DataLoader worker seeding
models/     MEConvLSTMPlus (velocity sources: track / frame_pair / external / mean_flow,
            residual Eulerian/Lagrangian skip, two-timescale forget gate), FEConvLSTMPlus,
            oracle stabiliser, persistence models, factory
datasets/   synthetic STEPS-style radar, real radar (SEVIR / pysteps / generic), rendered real
            trajectories (inD / rounD / exiD / highD + synthetic stand-in), fluids in a moving
            frame (Swift-Hohenberg, Rayleigh-Benard HDF5, The Well), GOES satellite, calcium
            imaging with motion injection, common-fate real video (Weizmann) + deformation sweep
physics/    Swift-Hohenberg solver, Rayleigh-Benard simulation scripts (Dedalus v3, Oceananigans.jl)
scripts/    data preparation / download CLIs, headroom and phase-correlation benchmarks
train_motion.py    prediction trainer for every registered dataset
train_cf_video.py  common-fate video classification trainer
tests/      pytest suite (CPU, a few minutes)

See motion_benchmarks/README.md.
"""
