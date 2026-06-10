# Robustness Evaluation of Pathology Foundation Models

A comprehensive evaluation framework for assessing the robustness of foundation models in digital pathology under various image corruption scenarios. This codebase is fully integrated with the [Kaiko EVA](https://github.com/kaiko-ai/eva) framework.

## Overview

This project evaluates how well state-of-the-art foundation models for pathology (Uni2, Prov-Gigapath, Virchow2, Pathryoshka, PhikonV2) maintain their performance when confronted with realistic image corruptions. The evaluation measures robustness through controlled perturbations applied at multiple severity levels, providing insights into model reliability and generalization in clinical deployment scenarios.

## Key Features

- **Multi-Model Evaluation**: Test multiple pathology foundation models systematically
- **Comprehensive Corruptions**: Apply 7 different corruption types:
  - Random rotation (90°)
  - Grayscale conversion (black & white)
  - JPEG compression
  - Color jitter
  - Pepper noise
  - Gamma adjustment
  - Gaussian blur

- **Multiple Datasets**: Evaluate on:
  - CRC (Colorectal Cancer) dataset
  - PatchCamelyon dataset

- **Dual Mode Testing**: Type1 and Type2 evaluation modes capturing different corruption behavior patterns

- **Severity Levels**: Test corruptions at progressive severity levels to capture robustness degradation curves

- **Comprehensive Metrics**: AUROC, Balanced Accuracy, F1 Score, and Recall for thorough model characterization

## Repository Structure

```
Robustness-Evaluation-of-Pathology-FMs/
├── MeanCorruptionDrop.py
├── make_robustness_delta_matrix_plot.py
├── crc_sweep.sbatch
├── patchcam_sweep.sbatch
└── robustness/
    ├── run_patchcam_sweep.py
    ├── plot_robustness_curves.py
    ├── plot_robustness_curves_multi_models.py
    ├── testsweep.py
    ├── transforms/
    ├── crc_blur_env.yaml
    ├── crc_blur_type2_env.yaml
    ├── crc_bw_env.yaml
    ├── crc_bw_type2_env.yaml
    ├── crc_colorjitter_env.yaml
    ├── crc_colorjitter_type2_env.yaml
    ├── crc_gamma_env.yaml
    ├── crc_gamma_type2_env.yaml
    ├── crc_jpeg_env.yaml
    ├── crc_jpeg_type2_env.yaml
    ├── crc_pepper_env.yaml
    ├── crc_pepper_type2_env.yaml
    ├── crc_rotate_env.yaml
    ├── crc_rotate_type2_env.yaml
    ├── patchcam_blur_env.yaml
    ├── patchcam_blur_type2_env.yaml
    ├── patchcam_bw_env.yaml
    ├── patchcam_bw_type2_env.yaml
    ├── patchcam_colorjitter_env.yaml
    ├── patchcam_colorjitter_type2_env.yaml
    ├── patchcam_gamma_env.yaml
    ├── patchcam_gamma_type2_env.yaml
    ├── patchcam_jpeg_env.yaml
    ├── patchcam_jpeg_type2_env.yaml
    ├── patchcam_pepper_env.yaml
    ├── patchcam_pepper_type2_env.yaml
    ├── patchcam_rotate_env.yaml
    ├── patchcam_rotate_type2_env.yaml
    ├── pathryoshka_timm_backbone.yaml
    ├── pathryoshka_timm_backbone_crc.yaml
    └── crc_public_paths.yaml
```

## Quick Start

### Prerequisites

- PyTorch 23.11+ (nvidia container or custom installation)
- EVA framework from Kaiko AI
- CUDA-capable GPU (minimum H100 recommended)
- Required Python packages: pandas, matplotlib, numpy, torch, torchvision, lightning

### Configuration

Before running, update path placeholders in the job scripts. All paths marked as `/yourpath/*` must be configured to your system:

| Placeholder | Description |
|---|---|
| `/yourpath/data/` | Path to datasets |
| `/yourpath/results/` | Output directory for results and logs |
| `/yourpath/containers/` | Location of container image |
| `/yourpath/code/` | This repository location |

### Running Evaluations

#### Option 1: Using SLURM (Recommended for clusters)

```bash
# CRC dataset evaluation
sbatch crc_sweep.sbatch

# PatchCamelyon dataset evaluation
sbatch patchcam_sweep.sbatch
```

#### Option 2: Direct Python execution

```bash
export HF_TOKEN="your_huggingface_token"
export MODEL_NAME="pathology/mahmood_uni2_h"
export AUGMENTATION="gaussian_blur"
export DATA_ROOT="/path/to/crc_eva_layout"
export RESULTS_BASE="/path/to/results"

python robustness/run_patchcam_sweep.py
```

## Evaluation Pipeline

1. **Data Loading**: Loads pathology images from EVA-compatible dataset structure
2. **Model Inference**: Passes clean and corrupted images through the foundation model
3. **Metric Computation**: Computes AUROC and secondary metrics at each corruption level
4. **Aggregation**: Aggregates results across:
   - Corruption severity levels
   - Individual augmentations
   - Datasets and models
5. **Visualization**: Generates robustness curves and comparison plots

## Output & Results

The evaluation produces several outputs:

- **Robustness Curves**: `*robustness_curves.png` - Performance vs corruption severity
- **Comparison Plots**: `*robustness_delta_matrix_plot.png` - Model-to-model comparisons
- **Excel Results**: `robustness_auroc_*.xlsx` containing:
  - Per-level AUROC drops
  - Per-augmentation aggregates
  - Per-dataset/model summaries
  - Plot data for reproduction

## Metrics & Scoring

### Mean Corruption Drop (MCD)

The primary robustness metric is the Mean Corruption Drop (MCD), computed as:

```
MCD = mean(baseline_AUROC - corrupted_AUROC) across all severity levels
```

Lower MCD indicates higher robustness. Results are aggregated across:

1. All corruption levels for each augmentation
2. All augmentations for each model/dataset/mode combination

## Container Setup

The evaluation uses the **nvidia pytorch 23.11** container:

```bash
IMG="/path/to/nvidia+pytorch+23.11-py3.sqsh"
```

This container is pre-configured with:

- PyTorch 23.11
- CUDA support
- Python 3
- Standard ML libraries

Within the container, EVA is cloned and installed fresh for each run to ensure reproducibility.

## Configuration Files

Configuration files in the `robustness/` directory specify:

- Model checkpoints and paths
- Dataset locations and splits
- Augmentation parameters (type and severity levels)
- Evaluation modes (Type1/Type2)
- Normalization statistics

**Naming Convention**: Files follow the pattern `{dataset}_{augmentation}_{mode}_env.yaml`

Examples:
- `crc_blur_env.yaml` - Gaussian blur on CRC dataset (Type1 mode)
- `patchcam_colorjitter_type2_env.yaml` - Color jitter on PatchCamelyon (Type2 mode)

## Supported Models

The framework evaluates the following pathology foundation models:

- **Uni2** (`pathology/mahmood_uni2_h`)
- **Prov-Gigapath** (`pathology/prov_gigapath`)
- **Virchow2** (`pathology/paige_virchow2`)
- **Pathryoshka** (`hf-hub:SchuefflerLab/pathryoshka-b`)
- **PhikonV2** (`pathology/owkin_phikon_v2`)

## Supported Augmentations

- `random_rotate90` - 90-degree rotation
- `black_white` - Grayscale conversion
- `jpeg_compression` - JPEG quality reduction
- `color_jitter` - Color brightness/contrast/saturation changes
- `pepper_noise` - Salt-and-pepper noise
- `gamma` - Gamma correction
- `gaussian_blur` - Gaussian blur at various kernel sizes

## Troubleshooting

### Path Configuration Errors

Ensure all `/yourpath/` placeholders in `.sbatch` files are updated to your actual paths. You can verify by checking:

```bash
grep -n "yourpath" *.sbatch
```

### GPU/Container Issues

- Verify GPU availability: `nvidia-smi`
- Check container image exists at specified path: `ls -lh /path/to/container.sqsh`
- Ensure sufficient disk space for results and container runtime

### EVA Integration

- Verify EVA submodule is properly initialized in the cloned repository
- Check Hugging Face token is valid: `echo $HF_TOKEN`
- Install EVA with vision extras: `python3 -m pip install -e ".[vision]"`

### Import Errors

If you encounter import errors when running directly (not in container):

```bash
python3 -m pip install --no-cache-dir "numpy<2" "torch<2.6" "torchvision<0.21" "lightning==2.5.5"
```

## License

[Add your license information here]

## Contact

For questions or issues, please open an issue in this repository.

## Acknowledgments

This project leverages the [Kaiko AI EVA framework](https://github.com/kaiko-ai/eva) for foundation model evaluation.
