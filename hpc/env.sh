source /opt/shared/.spack-edge/dist/bin/setup-env.sh -y || return 1
module load python/3.12.9-3lxwd5b || return 1
source .venv/bin/activate || return 1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
