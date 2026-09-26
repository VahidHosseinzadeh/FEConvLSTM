# 3D Rayleigh-Benard convection with Oceananigans.jl -- the solver Fromme et al.
# (arXiv:2505.13569) used, for an exact replication of their data.
#
#   div u = 0
#   dt u + (u . grad) u = -grad p + sqrt(Pr/Ra) lap u + T e_z
#   dt T + u . grad T   = (Ra Pr)^(-1/2) lap T
#
# (0, 2pi)^2 x (-1, 1), periodic horizontally, no-slip walls, T = T_bottom (1) at z = -1 and
# T_top (0) at z = +1; 48 x 48 x 32 finite-volume cells; Ra = 2500, Pr = 0.7; snapshots every
# 0.5 time units over t in [100, 300].
#
# Usage (one run per seed; parallelise with a Slurm array):
#   julia --project -e 'using Pkg; Pkg.add(["Oceananigans", "JLD2"])'      # once
#   SEED=0 OUT=rbc_runs/run_0000.jld2 julia --project rbc3d_oceananigans.jl
#   # optional env: RA PR NX NY NZ T_START T_END DT_SNAP MAX_DT GPU=1
# then convert to the HDF5 layout used by the Python side:
#   python -m motion_benchmarks.physics.convert_oceananigans --inputs rbc_runs/run_*.jld2 \
#       --outdir rbc_runs_h5
#   python -m motion_benchmarks.physics.merge_runs --inputs rbc_runs_h5/*.h5 --out rbc3d.h5
#
# Buoyancy is carried as a BuoyancyTracer `b` playing the role of T (the momentum equation's
# forcing is + b e_z, which is exactly the + T e_z above).

using Oceananigans
using Oceananigans.Units
using Random
using Printf

getenv(k, d) = get(ENV, k, string(d))
Ra      = parse(Float64, getenv("RA", 2500))
Pr      = parse(Float64, getenv("PR", 0.7))
Nx      = parse(Int,     getenv("NX", 48))
Ny      = parse(Int,     getenv("NY", 48))
Nz      = parse(Int,     getenv("NZ", 32))
t_start = parse(Float64, getenv("T_START", 100))
t_end   = parse(Float64, getenv("T_END", 300))
dt_snap = parse(Float64, getenv("DT_SNAP", 0.5))
max_dt  = parse(Float64, getenv("MAX_DT", 0.05))
T_bot   = parse(Float64, getenv("T_BOTTOM", 1.0))
T_top   = parse(Float64, getenv("T_TOP", 0.0))
seed    = parse(Int,     getenv("SEED", 0))
out     = getenv("OUT", @sprintf("run_%04d.jld2", seed))
arch    = getenv("GPU", "0") == "1" ? GPU() : CPU()

Random.seed!(seed)
ν = sqrt(Pr / Ra)
κ = 1 / sqrt(Ra * Pr)

grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), x=(0, 2π), y=(0, 2π), z=(-1, 1),
                       topology=(Periodic, Periodic, Bounded))

no_slip = FieldBoundaryConditions(top=ValueBoundaryCondition(0), bottom=ValueBoundaryCondition(0))
b_bcs   = FieldBoundaryConditions(top=ValueBoundaryCondition(T_top),
                                  bottom=ValueBoundaryCondition(T_bot))

model = NonhydrostaticModel(; grid,
                            advection = Centered(),
                            timestepper = :RungeKutta3,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            closure = ScalarDiffusivity(ν=ν, κ=κ),
                            boundary_conditions = (u=no_slip, v=no_slip, b=b_bcs))

bᵢ(x, y, z) = T_bot + (T_top - T_bot) * (z + 1) / 2 + 1e-3 * randn() * (1 - z^2)
set!(model, b=bᵢ)

simulation = Simulation(model; Δt=max_dt / 4, stop_time=t_start)
wizard = TimeStepWizard(cfl=0.5, max_Δt=max_dt)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))
progress(sim) = @info @sprintf("seed %d  t = %.2f  Δt = %.4f  max|w| = %.3f", seed,
                               time(sim), sim.Δt, maximum(abs, sim.model.velocities.w))
simulation.callbacks[:progress] = Callback(progress, IterationInterval(1000))

@info "spin-up to t = $t_start"
run!(simulation)

# output only after the spin-up; JLD2Writer was called JLD2OutputWriter before Oceananigans 0.96
Writer = isdefined(Oceananigans, :JLD2Writer) ? Oceananigans.JLD2Writer : JLD2OutputWriter
u, v, w = model.velocities
simulation.output_writers[:fields] = Writer(model, (; T=model.tracers.b, u, v, w);
                                            filename=out, schedule=TimeInterval(dt_snap),
                                            overwrite_existing=true)
simulation.stop_time = t_end
@info "recording to t = $t_end every $dt_snap -> $out"
run!(simulation)
@info "done: $out"
