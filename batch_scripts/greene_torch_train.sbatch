#!/bin/bash
#SBATCH -J gnn-train
#SBATCH -p gpu
#SBATCH --gres=rtx8000:1                  # or: --gres=gpu:rtx8000:1 / --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail

# Activate your Conda environment (bundles CUDA via pytorch-cuda 12.1 / cudatoolkit)
# Do NOT load any CUDA module.
source /home/$USER/activation_scripts/torch-gpu10_activate.sh

# Keep user site-packages from interfering
export PYTHONNOUSERSITE=True

# Optional: control threading (dataloaders/BLAS)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

# CUDA sanity prints in the job log
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("torch CUDA build:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

# Run training (absolute paths recommended)
python /home/$USER/GNN_project/hhhgraph/torch_train.py \
  -c /home/$USER/GNN_project/hhhgraph/config/ml_emd.yaml \
  -u /home/$USER/GNN_project/hhhgraph/config/user_Seb_greene.yaml
