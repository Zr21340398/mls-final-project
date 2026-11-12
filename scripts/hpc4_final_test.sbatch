#!/bin/bash
#SBATCH --job-name=energyfl-final-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-final-test-%j.out
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source "${ENERGYFL_ENV:-hpc/env.sh}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
echo "Final test job ${SLURM_JOB_ID:-unknown} on $(hostname)"
exec python -u scripts/final_experiments.py "$@"
