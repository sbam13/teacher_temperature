#!/bin/bash
#SBATCH --job-name=two-layer-lr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/workspace/teacher_temperature/teacher_temperature/two_layer_test}"
mkdir -p logs runs

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=false

python3 -m two_layer_experiment.run_lr_sweep --config configs/lr_sweep.yaml
python3 scripts/process_results.py --runs_dir runs/two_layer_lr_sweep
