#!/usr/bin/env python3
"""
plot_robustness_curves_combined.py

Creates ONE combined figure per:
  - dataset
  - augmentation

Layout:
  - rows = metrics
  - cols = type1, type2

Scaling:
  - for each (dataset, augmentation, metric), type1 and type2 share the same y-range
  - different metrics / augmentations / datasets can have different ranges

Legend:
  - shown once at the bottom of each combined figure
  - model colors stay consistent across all figures

Outputs:
  RESULTS_BASE/DATASET/multiplots_combined/AUGMENTATION/
    combined__DATASET__AUGMENTATION.png

Also writes a master CSV:
  RESULTS_BASE/_robustness_scales/all_plot_data.csv

Notes:
- Reads ONLY the newest results.json per level-XX (by newest run-folder mtime).
- Uses test metrics if present, else val.
- Multi-model only makes sense with MODEL_NAMES, but single-model also works.
"""

import os
import json
import re
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# ----------------------------
# helpers
# ----------------------------
def require_env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise RuntimeError(f"{name} must be set")
    return v


def parse_csv_env(name: str) -> List[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def model_to_dirname(model_name: str) -> str:
    return model_name.replace("/", "__")


def parse_model_names() -> List[str]:
    raw = os.environ.get("MODEL_NAMES", "").strip()
    if raw:
        models = [m.strip() for m in raw.split(",") if m.strip()]
        if not models:
            raise RuntimeError("MODEL_NAMES is set but empty after parsing")
        return models
    return [require_env("MODEL_NAME")]


def safe_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def short_model_name(model: str) -> str:
    short_name_map = {
        "pathology/mahmood_uni2_h": "Uni2",
        "pathology/prov_gigapath": "Prov-Gigapath",
        "pathology/paige_virchow2": "Virchow2",
        "hf-hub:SchuefflerLab/pathryoshka-b": "Pathryoshka",
        "pathology/owkin_phikon_v2": "PhikonV2",
    }
    return short_name_map.get(model, model)
    
DEFAULT_AUGS = [
    "random_rotate90",
    "black_white",
    "jpeg_compression",
    "color_jitter",
    "pepper_noise",
    "gamma",
    "gaussian_blur",
]

DEFAULT_MODES = ["type1", "type2"]


# ----------------------------
# canonical metrics mapping
# ----------------------------
TARGET_METRICS: Dict[str, List[str]] = {
    "AUROC": ["BinaryAUROC", "MulticlassAUROC"],
    "BalancedAccuracy": ["BinaryBalancedAccuracy", "MulticlassBalancedAccuracy"],
    "F1Score": ["BinaryF1Score", "MulticlassF1Score"],
    "Recall": ["BinaryRecall", "MulticlassRecall"],
}


def wanted_canonical_metrics(dataset: str) -> List[str]:
    if dataset == "crc":
        return ["AUROC", "Recall", "F1Score"]
    return ["AUROC", "BalancedAccuracy", "F1Score"]


def pick_metric(metrics: Dict[str, Tuple[float, float]], canonical: str) -> Optional[Tuple[float, float]]:
    for raw in TARGET_METRICS[canonical]:
        if raw in metrics:
            return metrics[raw]
    return None

# ----------------------------
# json save helper
# ----------------------------
def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
# ----------------------------
# EVA filesystem readers
# ----------------------------
_LEVEL_RE = re.compile(r"^level-(\d+)$")


def discover_level_dirs(exp_root: Path) -> List[Path]:
    if not exp_root.exists():
        return []
    out: List[Path] = []
    try:
        with os.scandir(exp_root) as it:
            for entry in it:
                if not entry.is_dir():
                    continue
                if _LEVEL_RE.match(entry.name):
                    out.append(Path(entry.path))
    except FileNotFoundError:
        return []
    out.sort(key=lambda p: int(p.name.split("-")[1]))
    return out


def find_latest_results_json(level_dir: Path) -> Optional[Path]:
    if not level_dir.exists():
        return None

    best_res: Optional[Path] = None
    best_mtime: float = -1.0

    try:
        with os.scandir(level_dir) as it:
            for entry in it:
                if not entry.is_dir():
                    continue
                res = Path(entry.path) / "results.json"
                if not res.exists():
                    continue
                try:
                    mtime = entry.stat().st_mtime
                except OSError:
                    continue
                if mtime > best_mtime:
                    best_mtime = mtime
                    best_res = res
    except FileNotFoundError:
        return None

    return best_res


def load_metrics_test_else_val(results_json: Path) -> Dict[str, Tuple[float, float]]:
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

    out: Dict[str, Tuple[float, float]] = {}
    for full_name, stats in block.items():
        if not full_name.startswith(prefix):
            continue
        metric_name = full_name.split("/", 1)[1]
        mean = float(stats.get("mean", float("nan")))
        stdev = float(stats.get("stdev", 0.0))
        out[metric_name] = (mean, stdev)

    if not out:
        raise ValueError(f"No {prefix} metrics found in {results_json}")
    return out


# ----------------------------
# cached collection
# ----------------------------
CanonicalRows = Dict[str, List[Tuple[int, float, float]]]


def collect_one_experiment_canonical(exp_root: Path, dataset: str) -> CanonicalRows:
    wanted = wanted_canonical_metrics(dataset)
    rows: CanonicalRows = {m: [] for m in wanted}

    for level_dir in discover_level_dirs(exp_root):
        level_idx = int(level_dir.name.split("-")[1])
        res = find_latest_results_json(level_dir)
        if res is None:
            continue
        try:
            raw_metrics = load_metrics_test_else_val(res)
        except Exception:
            continue

        for canon in wanted:
            picked = pick_metric(raw_metrics, canon)
            if picked is None:
                continue
            mean, stdev = picked
            rows[canon].append((level_idx, float(mean), float(stdev)))

    rows = {k: v for k, v in rows.items() if v}
    for k in rows:
        rows[k].sort(key=lambda x: x[0])
    return rows


def collect_all_cached(
    results_base: Path,
    dataset: str,
    aug: str,
    modes: List[str],
    models: List[str],
) -> Dict[str, Dict[str, CanonicalRows]]:
    """
    Returns:
      cache[mode][model] = canonical rows
    """
    cache: Dict[str, Dict[str, CanonicalRows]] = {m: {} for m in modes}

    for mode in modes:
        for model in models:
            exp_root = results_base / dataset / model_to_dirname(model) / mode / aug
            if not exp_root.exists():
                continue
            rows = collect_one_experiment_canonical(exp_root, dataset)
            if rows:
                cache[mode][model] = rows

    return cache


# ----------------------------
# per (dataset, augmentation, metric) shared y-range for type1/type2
# ----------------------------
def bucket_ymin(ymin_raw: float, step: float = 0.01) -> float:
    ymin_raw = float(ymin_raw)
    ymin_raw = max(0.0, min(1.0, ymin_raw))
    n_steps = int(math.ceil((1.0 - ymin_raw) / step))
    ymin = 1.0 - n_steps * step
    return round(max(0.0, ymin), 2)


def bucket_ymax(ymax_raw: float, step: float = 0.01) -> float:
    ymax_raw = float(ymax_raw)
    ymax_raw = max(0.0, min(1.0, ymax_raw))
    n_steps = int(math.ceil(ymax_raw / step))
    ymax = n_steps * step
    return round(min(1.0, ymax), 2)


def scale_key(dataset: str, augmentation: str, metric: str) -> str:
    return f"{dataset}__{augmentation}__{metric}"


def update_scale_cache_from_cached(
    scale_cache: dict,
    dataset: str,
    aug: str,
    cached: Dict[str, Dict[str, CanonicalRows]],
) -> None:
    wanted = wanted_canonical_metrics(dataset)

    for metric in wanted:
        lows: List[float] = []
        highs: List[float] = []

        for _mode, per_model in cached.items():
            for _model, rows_by_metric in per_model.items():
                for (_lvl, mean, stdev) in rows_by_metric.get(metric, []):
                    lows.append(float(mean) - float(stdev))
                    highs.append(float(mean) + float(stdev))

        if not lows:
            continue

        k = scale_key(dataset, aug, metric)
        ymin = float(np.min(lows))
        ymax = float(np.max(highs))

        cur = scale_cache.get(k)
        if cur is None:
            scale_cache[k] = {"ymin_raw": ymin, "ymax_raw": ymax}
        else:
            scale_cache[k]["ymin_raw"] = min(scale_cache[k]["ymin_raw"], ymin)
            scale_cache[k]["ymax_raw"] = max(scale_cache[k]["ymax_raw"], ymax)


def get_y_limits(scale_cache: dict, dataset: str, augmentation: str, metric: str) -> Tuple[float, float]:
    k = scale_key(dataset, augmentation, metric)
    entry = scale_cache.get(k, {})

    ymin_raw = float(entry.get("ymin_raw", 0.95))
    ymax_raw = float(entry.get("ymax_raw", 1.0))

    ymin = bucket_ymin(ymin_raw, step=0.01)
    ymax = min(1.0, bucket_ymax(ymax_raw + 0.01, step=0.01))

    if ymax <= ymin:
        ymax = min(1.0, round(ymin + 0.03, 2))

    return (ymin, ymax)


# ----------------------------
# consistent colors across all figures
# ----------------------------
def build_model_style_map(models: List[str]) -> Dict[str, dict]:
    cmap = plt.get_cmap("tab10")
    out: Dict[str, dict] = {}
    for i, model in enumerate(models):
        out[model] = {
            "color": cmap(i % 10),
            "label": short_model_name(model),
        }
    return out


# ----------------------------
# combined plotting
# ----------------------------
def write_combined_figure(
    out_path: Path,
    dataset: str,
    augmentation: str,
    modes: List[str],
    metrics: List[str],
    model_order: List[str],
    cached: Dict[str, Dict[str, CanonicalRows]],
    scale_cache: dict,
    style_map: Dict[str, dict],
) -> None:
    n_rows = len(metrics)
    n_cols = len(modes)

    # slightly smaller per-panel size so it fits Word better
    fig_w = 6.8
    fig_h = 1.9 * n_rows + 0.7

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    legend_handles = []
    legend_labels = []

    for r, metric in enumerate(metrics):
        y_limits = get_y_limits(scale_cache, dataset, augmentation, metric)

        for c, mode in enumerate(modes):
            ax = axes[r][c]
            per_model_metrics = cached.get(mode, {})

            plotted_any = False
            all_levels = set()

            for model in model_order:
                vals = per_model_metrics.get(model, {}).get(metric, [])
                if not vals:
                    continue

                vals = sorted(vals, key=lambda x: x[0])
                levels = np.array([v[0] for v in vals], dtype=int)
                means = np.array([v[1] for v in vals], dtype=float)
                stdevs = np.array([v[2] for v in vals], dtype=float)

                style = style_map[model]
                line = ax.errorbar(
                    levels,
                    means,
                    yerr=stdevs,
                    marker="o",
                    capsize=3,
                    linewidth=1.3,
                    markersize=3.8,
                    color=style["color"],
                    label=style["label"],
                )
                plotted_any = True
                all_levels.update(levels.tolist())

                if r == 0 and c == 0:
                    legend_handles.append(line[0])
                    legend_labels.append(style["label"])

            ax.set_ylim(y_limits[0], y_limits[1])
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            if all_levels:
                ax.set_xticks(sorted(all_levels))

            ax.grid(alpha=0.3)

            if r == n_rows - 1:
                ax.set_xlabel(mode, fontsize=9)

            if c == 0:
                ax.set_ylabel(metric, fontsize=9)

            if r == 0:
                ax.set_title(mode, fontsize=10)

            ax.tick_params(axis="both", labelsize=8)

            if not plotted_any:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=8)

    fig.suptitle(f"{dataset} | {augmentation}\ncorruption level (0 = baseline)", fontsize=11, y=0.955)

    # single shared legend at bottom
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=3,
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, 0.015),
            columnspacing=1.2,
            handletextpad=0.5,
        )

    plt.tight_layout(rect=[0.005, 0.09, 0.999, 0.945])
    fig.subplots_adjust(left=0.08, right=0.992, wspace=0.15, hspace=0.28)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


# ----------------------------
# main
# ----------------------------
def main():
    results_base = Path(require_env("RESULTS_BASE"))

    datasets = parse_csv_env("DATASET_NAMES")
    if not datasets:
        datasets = [require_env("DATASET_NAME")]

    modes = parse_csv_env("MODES") or DEFAULT_MODES
    # if user gives more, still preserve order; combined layout expects typically type1,type2
    modes = [m for m in modes if m in ("type1", "type2")] or DEFAULT_MODES

    augs = parse_csv_env("AUGMENTATIONS") or DEFAULT_AUGS
    models = parse_model_names()

    style_map = build_model_style_map(models)

    scale_dir = results_base / "_robustness_scales"
    scale_file = scale_dir / "y_scales.json"
    master_csv = scale_dir / "all_plot_data.csv"

    master_rows: List[Tuple[object, ...]] = []
    scale_cache: dict = {}
    all_cached: Dict[str, Dict[str, Dict[str, Dict[str, CanonicalRows]]]] = {}

    # ----------------------------
    # PHASE 1: collect everything + compute final y-ranges
    # ----------------------------
    for dataset in datasets:
        all_cached[dataset] = {}
        for aug in augs:
            cached = collect_all_cached(results_base, dataset, aug, modes, models)
            all_cached[dataset][aug] = cached
            update_scale_cache_from_cached(scale_cache, dataset, aug, cached)

    save_json(scale_file, scale_cache)

    print("[INFO] Final y-ranges:")
    for dataset in datasets:
        for aug in augs:
            for metric in wanted_canonical_metrics(dataset):
                yr = get_y_limits(scale_cache, dataset, aug, metric)
                print(f"  {dataset} | {aug} | {metric}: {yr[0]:.2f} .. {yr[1]:.2f}")

    # ----------------------------
    # PHASE 2: build master CSV + combined figures
    # ----------------------------
    for dataset in datasets:
        metrics = wanted_canonical_metrics(dataset)

        for aug in augs:
            cached = all_cached[dataset][aug]

            # master CSV rows
            for mode in modes:
                per_model_metrics = cached.get(mode, {})
                for model in models:
                    rows_by_metric = per_model_metrics.get(model, {})
                    for metric in metrics:
                        for (lvl, mean, stdev) in rows_by_metric.get(metric, []):
                            master_rows.append((
                                dataset,
                                mode,
                                aug,
                                metric,
                                short_model_name(model),
                                int(lvl),
                                float(mean),
                                float(stdev),
                            ))

            # only include models that actually have any data in this dataset+aug
            model_order = []
            for model in models:
                present = False
                for mode in modes:
                    if model in cached.get(mode, {}):
                        present = True
                        break
                if present:
                    model_order.append(model)

            if not model_order:
                print(f"[WARN] No usable data: {dataset} | {aug}")
                continue

            out_dir = results_base / dataset / "multiplots_combined" / aug
            out_path = out_dir / f"combined__{safe_slug(dataset)}__{safe_slug(augmentation_display_name(aug))}.png"

            write_combined_figure(
                out_path=out_path,
                dataset=dataset,
                augmentation=aug,
                modes=modes,
                metrics=metrics,
                model_order=model_order,
                cached=cached,
                scale_cache=scale_cache,
                style_map=style_map,
            )

            print(f"[DONE] {dataset} | {aug} -> {out_path}")

    scale_dir.mkdir(parents=True, exist_ok=True)
    uniq = sorted(set(master_rows), key=lambda r: (r[0], r[1], r[2], r[3], r[4], r[5]))
    with open(master_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "mode", "augmentation", "metric", "model", "level", "mean", "stdev"])
        w.writerows(uniq)

    print(f"[INFO] Y-scale cache: {scale_file}")
    print(f"[INFO] Master CSV: {master_csv}")


def augmentation_display_name(aug: str) -> str:
    return aug


if __name__ == "__main__":
    main()