#!/bin/bash
#SBATCH --job-name=energyfl-controller
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=24:00:00
#SBATCH --output=controller-%j.out

# Set the CPU account, partition and allowed wall time when submitting.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source "${ENERGYFL_ENV:-hpc/env.sh}"
export PYTHONUNBUFFERED=1
exec python -u scripts/sequential_slurm.py run "$@"
