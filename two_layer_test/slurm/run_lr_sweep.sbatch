#!/bin/bash
#SBATCH --job-name=two-layer-lr
#SBATCH --partition=kempner,kempner_h100               # Partition (queue) to submit to
#SBATCH --account=kempner_grads
#SBATCH -N 1                       # number of nodes
#SBATCH -n 24                      # number of cores , A100: 64 cores, H100: 96 cores
#SBATCH --gres=gpu:1               # Number of GPUs per node
#SBATCH --mem 250G                 # Memory per node, H100: 1.5 TB, A100: 1 TB RAM
#SBATCH -t 00-06:00                # time (D-HH:MM)
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/workspace/teacher_temperature/teacher_temperature/two_layer_test}"
mkdir -p logs runs

source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/mamba.sh
mamba activate scratch

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

CONFIG="${CONFIG:-configs/lr_sweep.yaml}"
OUT_DIR="${OUT_DIR:-runs/two_layer_lr_sweep}"
BETAS_ARG=()
if [[ -n "${BETAS:-}" ]]; then
    BETAS_ARG=(--betas "$BETAS")
fi

echo "CONFIG=$CONFIG"
echo "OUT_DIR=$OUT_DIR"
echo "BETAS=${BETAS:-all}"
echo "Minibatch size is read from minibatch_size, or derived as round(p ** delta) when minibatch_size is unset."

python -m two_layer_experiment.run_lr_sweep --config "$CONFIG" --out_dir "$OUT_DIR" "${BETAS_ARG[@]}"
python -m scripts.process_results --runs_dir "$OUT_DIR"
