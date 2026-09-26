#!/bin/bash
# Generate the Fromme et al. Rayleigh-Benard set with Dedalus: one serial run per array task.
#
#   mkdir -p rbc_runs && sbatch --array=0-99 motion_benchmarks/slurm/rbc3d_dedalus.sbatch
#   # afterwards (login node is fine, it only writes a virtual dataset):
#   python -m motion_benchmarks.physics.merge_runs --inputs "rbc_runs/run_*.h5" --out rbc3d_ra2500.h5
#
# 48 x 48 x 32 to t = 300 takes on the order of an hour or two per run on one core.
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_rbc_%A_%a.log

set -e
export OMP_NUM_THREADS=1
PYTHON=${PYTHON:-python}
SEED=${SLURM_ARRAY_TASK_ID:-0}
OUT=$(printf "rbc_runs/run_%04d.h5" "$SEED")
mkdir -p rbc_runs
$PYTHON -m motion_benchmarks.physics.rbc3d_dedalus --seed "$SEED" --out "$OUT" ${RBC_ARGS:-}
