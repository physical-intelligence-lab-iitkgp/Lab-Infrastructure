#!/usr/bin/env bash
#SBATCH --job-name=pytorch-gpu-test
#SBATCH --output=/scratch/scratch26/26at91r15/GPT2_Project/logs/%x-%j.out
#SBATCH --error=/scratch/scratch26/26at91r15/GPT2_Project/logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -euo pipefail

# Project directory
PROJECT_DIR="/scratch/scratch26/26at91r15/GPT2_Project"

# Test program directory
CLUSTER_DIR="/scratch/scratch26/26at91r15/Lab-Infrastructure/cluster_access"

# Display job information
echo "Job ID: ${SLURM_JOB_ID}"
echo "Compute node: $(hostname)"
echo "Project directory: ${PROJECT_DIR}"
date

# Move to project directory
cd "${PROJECT_DIR}"

# Load CUDA
module purge
module load compiler/cuda/12.4

# Activate existing Conda environment
source /home/apps/anaconda3/etc/profile.d/conda.sh
unset PYTHONPATH
conda activate /home/26at91r15/.conda/envs/torchenv

# Configure CPU threads
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Verify software
python --version
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda)"

# Require GPU availability
python -c "import torch; assert torch.cuda.is_available(), 'CUDA GPU is not available'; print('GPU:', torch.cuda.get_device_name(0))"

# Run PyTorch test
srun --ntasks=1 python "${CLUSTER_DIR}/pytorch_submission/test_pytorch.py"

echo "PyTorch GPU test completed successfully."
date
