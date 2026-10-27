#!/bin/bash
#SBATCH --job-name=energyfl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.log
set -euo pipefail
# Submit from the project directory. Supply the site's partition/account via sbatch.
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
: "${ENERGYFL_PYTHON:?Set ENERGYFL_PYTHON to your node virtualenv Python}"
CONFIG="${ENERGYFL_CONFIG:-configs/l20.json}"
RUN="${ENERGYFL_RUN:-runs/l20_${SLURM_JOB_ID}}"
"${ENERGYFL_PYTHON}" -m energyfl train --config "$CONFIG" --run "$RUN"
# Test evaluation is intentionally a separate command after validation choices freeze.
