#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH -o /yourpath/results/logs/log_%j.out
#SBATCH -e /yourpath/results/logs/log_%j.err

set -euo pipefail

LOGDIR="/yourpath/results/logs"
mkdir -p "${LOGDIR}"

: "${HF_TOKEN:?}"
: "${MODEL_NAME:?}"
: "${AUGMENTATION:?}"

: "${DATASET_NAME:=crc}"
: "${DATA_ROOT:=/yourpath/data/crc_eva_layout}"
: "${RESULTS_BASE:=/yourpath/results}"
: "${NORMALIZE_MEAN:=[0.485,0.456,0.406]}"
: "${NORMALIZE_STD:=[0.229,0.224,0.225]}"
: "${IN_FEATURES:=1536}"

export HF_TOKEN MODEL_NAME AUGMENTATION DATASET_NAME DATA_ROOT RESULTS_BASE NORMALIZE_MEAN NORMALIZE_STD IN_FEATURES

IMG="/yourpath/containers/nvidia+pytorch+23.11-py3.sqsh"

MOUNTS="/yourpath/code:/yourpath/code:ro,/yourpath/data:/yourpath/data:ro,/yourpath/public_datasets/CRC:/yourpath/public_datasets/CRC:ro,/yourpath/results:/yourpath/results:rw"

CODE_ROOT="/yourpath/code"
SWEEP_SCRIPT="${CODE_ROOT}/robustness/run_patchcam_sweep.py"

srun --gres=gpu:1 \
  --container-image="${IMG}" \
  --container-mounts="${MOUNTS}" \
  bash -lc '
    set -euo pipefail

    cd /workspace
    rm -rf eva
    git clone --recursive https://github.com/kaiko-ai/eva.git
    cd eva

    python3 -m pip install --no-cache-dir -e ".[vision]"
    python3 -m pip install --no-cache-dir --force-reinstall \
      "numpy<2" "torch<2.6" "torchvision<0.21" "lightning==2.5.5"

    MODE=type1 python3 /yourpath/code/robustness/run_patchcam_sweep.py
    MODE=type2 python3 /yourpath/code/robustness/run_patchcam_sweep.py
    MODE=type1 python3 /yourpath/code/robustness/plot_robustness_curves_multi_models.py 
    MODE=type2 python3 /yourpath/code/robustness/plot_robustness_curves_multi_models.py
  '