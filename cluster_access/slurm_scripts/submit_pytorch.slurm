
#!/usr/bin/env bash
#SBATCH --job-name=pytorch-smoke-test
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -euo pipefail

# Locate project directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "$CLUSTER_DIR"

# Load PARAM Shakti modules
module purge
module load compiler/cuda/12.4
module load custom/python/3.10.13-custom

# Activate Python environment
unset PYTHONPATH
source /scratch/scratch26/26at91r15/GPT2_Project/environment/gpt2_env/bin/activate

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Compute node: $(hostname)"
echo "Python version:"
python --version

# Verify PyTorch and CUDA
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available')"

# Run PyTorch test
srun python "${CLUSTER_DIR}/pytorch_submission/test_pytorch.py"

echo "Job finished:"
date
