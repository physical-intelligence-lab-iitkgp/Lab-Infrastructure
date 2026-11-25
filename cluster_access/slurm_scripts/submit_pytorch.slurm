#!/usr/bin/env bash
#SBATCH --job-name=pytorch-smoke-test
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu              # Replace with your cluster's GPU partition name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1                 # Adjust or remove if testing on CPU partition
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -euo pipefail

# Resolve cluster_access root whether submitted from cluster_access/ or slurm_scripts/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
mkdir -p "${CLUSTER_DIR}/logs"

echo "Job ID: ${SLURM_JOB_ID:-local} | Node: $(hostname)"
date

# --- Environment Setup (uncomment & adjust to match your cluster) ---
# module purge
# module load cuda/12.1 python/3.10
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate pytorch_env

python3 --version
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

srun python3 "${CLUSTER_DIR}/pytorch_submission/test_pytorch.py"

echo "Finished at:"
date