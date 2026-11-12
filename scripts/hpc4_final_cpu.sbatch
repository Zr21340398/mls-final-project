#!/bin/bash
#SBATCH --job-name=energyfl-compress
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-compress-%j.out
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source "${ENERGYFL_ENV:-hpc/env.sh}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
echo "CPU compression job ${SLURM_JOB_ID:-unknown} on $(hostname)"
exec python -u scripts/final_experiments.py "$@"
