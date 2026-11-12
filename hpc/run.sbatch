#!/bin/bash
#SBATCH --job-name=energyfl
#SBATCH --account=mscaiehpc42026
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -e
cd "$SLURM_SUBMIT_DIR"
source hpc/env.sh
if [ "$#" -eq 0 ]; then
    echo 'Usage: sbatch hpc/run.sbatch -m energyfl train ...'
    exit 2
fi
hostname
date
nvidia-smi
git rev-parse HEAD
srun python -u "$@"
date
