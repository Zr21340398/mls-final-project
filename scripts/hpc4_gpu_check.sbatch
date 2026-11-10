#!/bin/bash
#SBATCH --job-name=energyfl-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-check-%j.out

# Set account and partition on the sbatch command line for the current site.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
ENV_FILE="${ENERGYFL_ENV:-hpc/env.sh}"
if [ ! -f "$ENV_FILE" ]; then
    echo "Environment file not found: $ENV_FILE" >&2
    exit 1
fi
# The environment file loads the site's Python module and the project venv.
source "$ENV_FILE"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

echo "Job ${SLURM_JOB_ID:-unknown} on $(hostname)"
nvidia-smi
python - <<'PY'
import json
import torch
from energyfl.common import load_config
from energyfl.data import load_dataset

config = load_config('configs/l20.json')
manifest, households = load_dataset(config['data'])
print('Dataset:', config['data'], 'households:', len(households),
      'fingerprint:', manifest['fingerprint'], 'synthetic:', manifest['synthetic'])
if manifest['synthetic'] or len(households) != 60:
    raise SystemExit('Expected 60 real households; inspect the dataset and configuration.')
print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable; check GPU allocation and the installed PyTorch wheel.')
print('GPU:', torch.cuda.get_device_name(0))
x = torch.randn(512, 512, device='cuda')
y = x @ x
torch.cuda.synchronize()
print('CUDA computation completed:', tuple(y.shape))
PY

case "${1:-check}" in
    check)
        echo 'GPU and data checks complete; no calibration or training was started.'
        ;;
    calibrate)
        # New paths preserve the original template and earlier measurements.
        python -m energyfl calibrate --config configs/l20.json \
            --out-config configs/l20.hpc4_v1.json \
            --report reports/hpc4_setup_v1/profile.json
        python -m pip freeze > reports/hpc4_setup_v1/environment.freeze.txt
        echo 'GPU checks and node calibration complete.'
        ;;
    *)
        echo 'Usage: sbatch scripts/hpc4_gpu_check.sbatch [check|calibrate]' >&2
        exit 2
        ;;
esac
