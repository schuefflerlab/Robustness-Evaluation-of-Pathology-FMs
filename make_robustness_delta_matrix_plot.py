#!/usr/bin/env python3

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# LEVEL DEFINITIONS (only level indices matter, values unused)
# ============================================================

REGISTRY_LEVELS: Dict[str, Dict[int, object]] = {
    "gaussian_blur":      {0: None, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
    "color_jitter":       {0: None, 1: 0, 2: 0, 3: 0},
    "black_white":        {0: None, 1: 0},
    "gamma":              {0: None, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
    "jpeg_compression":   {0: None, 1: 0, 2: 0, 3: 0},
    "random_rotate90":    {0: None, 1: 0},
    "pepper_noise":       {0: None, 1: 0, 2: 0, 3: 0},
}

AUG_ORDER = [
    "random_rotate90",
    "black_white",
    "jpeg_compression",
    "color_jitter",
    "pepper_noise",
    "gamma",
    "gaussian_blur",
]

MODEL_DISPLAY_NAMES = {
    "mahmood_uni2_h": "Uni2",
    "prov_gigapath": "Prov-Gigapath",
    "paige_virchow2": "Virchow2",
    "pathryoshka-b": "Pathryoshka",
    "owkin_phikon_v2": "PhikonV2",
}
# ============================================================
# METRICS
# ============================================================

TARGET_METRICS = {
    "AUROC": ["BinaryAUROC", "MulticlassAUROC"],
    "BalancedAccuracy": ["BinaryBalancedAccuracy", "MulticlassBalancedAccuracy"],
    "F1Score": ["BinaryF1Score", "MulticlassF1Score"],
    "Recall": ["BinaryRecall", "MulticlassRecall"],
}

METRIC_ORDER = ["AUROC", "BalancedAccuracy", "F1Score"]


# ============================================================
# HELPERS
# ============================================================

def require_env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise RuntimeError(f"{name} must be set")
    return v


def model_to_dirname(model_name: str) -> str:
    return model_name.replace("/", "__")


def short_model_name(model_dirname: str) -> str:
    """
    Extract only the part AFTER '/' and map to a paper-friendly display name.
    Example:
      pathology__owkin_phikon_v2 -> PhikonV2
    """
    display = model_dirname.replace("__", "/")
    raw_name = display.split("/")[-1] if "/" in display else display
    return MODEL_DISPLAY_NAMES.get(raw_name, raw_name)


LEVEL_RE = re.compile(r"level-(\d+)$")


def find_latest_results_json(level_dir: Path) -> Optional[Path]:
    if not level_dir.exists():
        return None
    candidates = []
    for run_dir in level_dir.iterdir():
        if run_dir.is_dir():
            res = run_dir / "results.json"
            if res.exists():
                candidates.append(res)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.parent.stat().st_mtime)
    return candidates[-1]


def load_metrics_test_else_val(results_json: Path) -> Dict[str, float]:
    with open(results_json, "r") as f:
        data = json.load(f)

    metrics_root = data.get("metrics", {})
    test_entries = metrics_root.get("test", [])

    if test_entries:
        block = test_entries[0]
        prefix = "test/"
    else:
        val_entries = metrics_root.get("val", [])
        if not val_entries:
            raise ValueError("No test or val metrics found")
        block = val_entries[0]
        prefix = "val/"

    out = {}
    for full_name, stats in block.items():
        if full_name.startswith(prefix):
            metric = full_name.split("/", 1)[1]
            out[metric] = float(stats.get("mean", float("nan")))

    return out


def pick_metric(metrics: Dict[str, float], wanted: str) -> Optional[float]:
    for k in TARGET_METRICS[wanted]:
        if k in metrics:
            return metrics[k]
    return None


def compute_delta_strict(aug_root: Path, aug_name: str) -> Dict[str, float]:
    expected_max = max(REGISTRY_LEVELS[aug_name].keys())

    level0 = aug_root / "level-00"
    levelM = aug_root / f"level-{expected_max:02d}"

    if not level0.exists() or not levelM.exists():
        raise ValueError("Missing baseline or expected max level")

    r0 = find_latest_results_json(level0)
    rM = find_latest_results_json(levelM)

    if r0 is None or rM is None:
        raise ValueError("Missing results.json")

    m0 = load_metrics_test_else_val(r0)
    mM = load_metrics_test_else_val(rM)

    out = {}
    for metric in METRIC_ORDER:
        v0 = pick_metric(m0, metric)
        vM = pick_metric(mM, metric)
        if v0 is None or vM is None:
            raise ValueError(f"Missing {metric}")
        out[metric] = vM - v0
    return out


# ============================================================
# PLOTTING
# ============================================================

def wrap_label(s: str, max_len: int = 14) -> str:
    if "_" not in s:
        return s
    parts = s.split("_")
    lines = []
    cur = ""
    for p in parts:
        if not cur:
            cur = p
        elif len(cur) + 1 + len(p) <= max_len:
            cur = f"{cur}_{p}"
        else:
            lines.append(cur)
            cur = p
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def plot_grouped_heatmap(data, model_labels, col_labels, type_title, subtitle, out_path):
    block = len(METRIC_ORDER)
    n_rows = data.shape[0]

    fig_w = max(9.0, 0.75 * len(col_labels) + 4)
    fig_h = max(6.0, 0.33 * n_rows + 2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(data, aspect="auto")

    # X axis
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels([wrap_label(c) for c in col_labels],
                       rotation=35, ha="right", fontsize=9)

    # Y axis (metric names only)
    y_labels = []
    for _ in model_labels:
        for m in METRIC_ORDER:
            y_labels.append(m)

    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)

    # separators
    for k in range(block, n_rows, block):
        ax.axhline(k - 0.5, linewidth=1)

    # title
    fig.suptitle(type_title, fontsize=16, fontweight="bold", y=0.98)
    ax.set_title(subtitle, fontsize=10, loc="left", pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Δ score (max - baseline)", fontsize=10)

    # annotate
    for i in range(n_rows):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.3f}",
                        ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(left=0.22)

    # === MODEL NAMES ===
    # align to AUROC row (top row of each block)
    for mi, name in enumerate(model_labels):
        row_auroc = mi * block  # first row in block
        y_display = ax.transData.transform((0, row_auroc))[1]
        y_fig = fig.transFigure.inverted().transform((0, y_display))[1]

        fig.text(0.02, y_fig, name,
                 ha="left", va="center",
                 fontsize=10, fontweight="bold")

    plt.savefig(out_path, dpi=200)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    RESULTS_BASE = Path(require_env("RESULTS_BASE"))
    DATASET_NAME = require_env("DATASET_NAME")
    global METRIC_ORDER
    if DATASET_NAME == "crc":
        METRIC_ORDER = ["AUROC", "Recall", "F1Score"]

    dataset_root = RESULTS_BASE / DATASET_NAME
    if not dataset_root.exists():
        raise RuntimeError("Dataset root not found")

    types = os.environ.get("TYPES", "type1,type2").split(",")

    raw_models = os.environ.get("MODEL_NAMES", "").strip()
    if raw_models:
        model_dirs = [model_to_dirname(m.strip())
                      for m in raw_models.split(",")]
    else:
        model_dirs = [p.name for p in dataset_root.iterdir()
                      if p.is_dir() and p.name.startswith("pathology__")]

    model_labels = [short_model_name(md) for md in model_dirs]

    augmentations = AUG_ORDER[:]

    out_dir = dataset_root / "_summaries"
    out_dir.mkdir(parents=True, exist_ok=True)

    for t in types:
        R = len(model_dirs) * len(METRIC_ORDER)
        C = len(augmentations)

        mat = np.full((R, C), np.nan)

        for mi, md in enumerate(model_dirs):
            for aj, aug in enumerate(augmentations):
                aug_root = dataset_root / md / t / aug
                if not aug_root.exists():
                    continue
                try:
                    deltas = compute_delta_strict(aug_root, aug)
                except Exception:
                    continue

                for k, metric in enumerate(METRIC_ORDER):
                    row = mi * len(METRIC_ORDER) + k
                    mat[row, aj] = deltas[metric]

        mean_col = np.nanmean(mat, axis=1, keepdims=True)
        median_col = np.nanmedian(mat, axis=1, keepdims=True)
        mat = np.concatenate([mat, mean_col, median_col], axis=1)

        col_labels = augmentations + ["MEAN", "MEDIAN"]
        type_title = t.replace("type", "Type ")
        subtitle = f"{DATASET_NAME} | Metric difference: Δ = score(max level) - score(baseline)"

        out_path = out_dir / f"delta_matrix_grouped__{DATASET_NAME}__{t}.png"

        plot_grouped_heatmap(mat, model_labels, col_labels,
                     type_title, subtitle, out_path)

        print("[INFO] Wrote", out_path)


if __name__ == "__main__":
    main()
