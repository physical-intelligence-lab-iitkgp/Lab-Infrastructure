#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "CLUSTER ENVIRONMENT & SANITY CHECK"
echo "=========================================="

echo "--- Host, User & Directory ---"
echo "Hostname : $(hostname)"
echo "User     : $(whoami)"
echo "PWD      : $(pwd)"
echo "Date     : $(date)"

echo "--- Slurm Status ---"
if command -v sinfo >/dev/null 2>&1; then
    sinfo || true
else
    echo "sinfo not found."
fi

if command -v squeue >/dev/null 2>&1; then
    squeue -u "$USER" || true
else
    echo "squeue not found."
fi

echo "--- GPU & CUDA Check ---"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "nvidia-smi not found on this node."
fi

if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | tail -n 4
else
    echo "nvcc not found in PATH."
fi

echo "--- Python & PyTorch Check ---"
if command -v python3 >/dev/null 2>&1; then
    echo "Python3  : $(command -v python3)"
    python3 --version
    python3 - <<'EOF'
try:
    import torch
    print("PyTorch  :", torch.__version__)
    print("CUDA ok  :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        print("GPU 0    :", torch.cuda.get_device_name(0))
except ImportError:
    print("PyTorch  : not installed in current Python environment")
EOF
else
    echo "python3 not found."
fi

echo "=========================================="
echo "ENVIRONMENT CHECK COMPLETE"
echo "=========================================="