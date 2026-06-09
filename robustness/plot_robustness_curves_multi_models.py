#!/usr/bin/env python3
"""
plot_robustness_curves.py

One script, two modes:

    Single-model mode (default, same as old script):
        Uses MODEL_NAME
        Writes plots/CSVs into:
            RESULTS_BASE/DATASET_NAME/MODEL_NAME__/MODE/AUGMENTATION/plots

    Comparison mode (2+ models):
        Uses MODEL_NAMES (comma-separated)
        Writes plots/CSVs into:
            RESULTS_BASE/DATASET_NAME/_comparisons/MODE/AUGMENTATION/plots

Common behavior:
    Looks for level-XX directories
    For each level, selects the newest finished EVA run (latest mtime run folder containing results.json)
    Loads all test metrics from results.json (keys test/*)
    Skips missing models/levels gracefully

"""

import os
import json
import re
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# small helpers
# ----------------------------
def require_env(name: str) -> str:
    v = os.environ.get(name, "")
    if not v:
        raise RuntimeError(f"{name} must be set")
    return v


def model_to_dirname(model_name: str) -> str:
    # Matches your existing folder naming convention
    return model_name.replace("/", "__")


def parse_model_names() -> List[str]:
    """
    Priority:
      - If MODEL_NAMES is set: use it (comma-separated) => comparison mode if 2+ items
      - Else: fall back to MODEL_NAME => single-model mode
    """
    raw = os.environ.get("MODEL_NAMES", "").strip()
    if raw:
        models = [m.strip() for m in raw.split(",") if m.strip()]
        if not models:
            raise RuntimeError("MODEL_NAMES is set but empty after parsing")
        return models
    return [require_env("MODEL_NAME")]


# ----------------------------
# EVA results readers
# ----------------------------
def find_latest_results_json(level_dir: Path) -> Optional[Path]:
    """
    Find newest results.json under a level-XX directory.

    EVA creates subfolders per run (often timestamps/hydra runs). We pick the
    newest run folder by modification time, as your old script did.
    """
    if not level_dir.exists():
        return None

    candidates: List[Path] = []
    for run_dir in level_dir.iterdir():
        if not run_dir.is_dir():
            continue
        res = run_dir / "results.json"
        if res.exists():
            candidates.append(res)

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.parent.stat().st_mtime)
    return candidates[-1]


def load_metrics_test_else_val(results_json: Path) -> Dict[str, Tuple[float, float]]:
    """
    Returns: metric_name -> (mean, stdev)

    Rule:
      - Use test metrics if present (metrics["test"][0] with keys "test/<metric>")
      - Else use val metrics if present (metrics["val"][0] with keys "val/<metric>")
      - Never use train
    """
    with open(results_json, "r") as f:
        data = json.load(f)

    metrics_root = data.get("metrics", {})

    # Prefer test
    test_entries = metrics_root.get("test", [])
    if test_entries:
        block = test_entries[0]
        prefix = "test/"
    else:
        # Fallback to val
        val_entries = metrics_root.get("val", [])
        if not val_entries:
            raise ValueError(f"No test or val metrics found in {results_json}")
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
        raise ValueError(f"Found {prefix[:-1]} metrics block but no '{prefix}<metric>' keys in {results_json}")

    return out



def discover_level_dirs(exp_root: Path) -> List[Path]:
    LEVEL_RE = re.compile(r"level-(\d+)$")

    level_dirs: List[Path] = []
    if not exp_root.exists():
        return level_dirs

    for p in exp_root.iterdir():
        if not p.is_dir():
            continue
        if LEVEL_RE.match(p.name):
            level_dirs.append(p)

    level_dirs.sort(key=lambda p: int(p.name.split("-")[1]))
    return level_dirs


# ----------------------------
# plotting + csv
# ----------------------------
def safe_slug(s: str) -> str:
    # safe for filenames
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

def short_model_name(model: str) -> str:
    model = model.replace("__", "/")
    return model.split("/")[-1] if "/" in model else model

def write_metric_plot_and_csv_single(
    out_dir: Path,
    dataset: str,
    model_display: str,
    mode: str,
    augmentation: str,
    metric_name: str,
    values: List[Tuple[int, float, float]],
) -> None:
    values.sort(key=lambda x: x[0])
    levels = np.array([v[0] for v in values], dtype=int)
    means = np.array([v[1] for v in values], dtype=float)
    stdevs = np.array([v[2] for v in values], dtype=float)

    plt.figure(figsize=(4.6, 3.2))
    plt.errorbar(levels, means, yerr=stdevs, marker="o", capsize=4)
    plt.xticks(levels)
    plt.xlabel("Augmentation level (0 = baseline)")
    plt.ylabel(metric_name)
    plt.title(f"{dataset} | {model_display}\n{augmentation} ({mode})")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = out_dir / f"{safe_slug(metric_name)}_vs_level.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    csv_path = out_dir / f"{safe_slug(metric_name)}_vs_level.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "mean", "stdev"])
        for l, m, s in zip(levels, means, stdevs):
            w.writerow([int(l), float(m), float(s)])


def write_metric_plot_and_csv_multi(
    out_dir: Path,
    dataset: str,
    mode: str,
    augmentation: str,
    metric_name: str,
    per_model_values: Dict[str, List[Tuple[int, float, float]]],  # model_display -> values
    model_order: List[str],
) -> None:
    plt.figure(figsize=(5, 3.3))

    merged_rows: List[List[object]] = []
    plotted_any = False

    for model_display in model_order:
        values = per_model_values.get(model_display, [])
        if not values:
            continue

        values.sort(key=lambda x: x[0])
        levels = np.array([v[0] for v in values], dtype=int)
        means = np.array([v[1] for v in values], dtype=float)
        stdevs = np.array([v[2] for v in values], dtype=float)

        plt.errorbar(levels, means, yerr=stdevs, marker="o", capsize=4, label=short_model_name(model_display))
        plotted_any = True

        for l, m, s in zip(levels, means, stdevs):
            merged_rows.append([model_display, int(l), float(m), float(s)])

        # per-model csv for this metric
        per_model_csv = out_dir / f"{safe_slug(metric_name)}__{safe_slug(model_display)}_vs_level.csv"
        with open(per_model_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "level", "mean", "stdev"])
            for l, m, s in zip(levels, means, stdevs):
                w.writerow([model_display, int(l), float(m), float(s)])

    if not plotted_any:
        plt.close()
        return
        
    # Collect all levels that appear in any model for this metric
    all_levels = sorted({lvl for vals in per_model_values.values() for (lvl, _, _) in vals})
    if all_levels:
        plt.xticks(all_levels)

    plt.xlabel("Augmentation level (0 = baseline)")
    plt.ylabel(metric_name)
    plt.title(f"{dataset} | {augmentation} ({mode})\n{metric_name}")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    plot_path = out_dir / f"{safe_slug(metric_name)}__multi_model_vs_level.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    merged_csv = out_dir / f"{safe_slug(metric_name)}__multi_model_vs_level.csv"
    merged_rows.sort(key=lambda r: (r[0], r[1]))  # model, level
    with open(merged_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "level", "mean", "stdev"])
        w.writerows(merged_rows)


# ----------------------------
# data collection
# ----------------------------
def collect_metrics_for_one_model(exp_root: Path) -> Dict[str, List[Tuple[int, float, float]]]:
    """
    Returns: metric_name -> list[(level, mean, stdev)]
    Skips missing/invalid levels; returns {} if nothing found.
    """
    level_dirs = discover_level_dirs(exp_root)
    if not level_dirs:
        return {}

    all_metrics: Dict[str, List[Tuple[int, float, float]]] = {}

    for level_dir in level_dirs:
        level_idx = int(level_dir.name.split("-")[1])
        results_json = find_latest_results_json(level_dir)
        if results_json is None:
            print(f"[WARN] {level_dir}: no finished run -> skipping level")
            continue

        try:
            metrics = load_metrics_test_else_val(results_json)
        except Exception as e:
            print(f"[WARN] {level_dir}: failed to read metrics ({e}) -> skipping level")
            continue

        for metric_name, (mean, stdev) in metrics.items():
            all_metrics.setdefault(metric_name, []).append((level_idx, mean, stdev))

    # sort each metric by level
    for k in list(all_metrics.keys()):
        all_metrics[k].sort(key=lambda x: x[0])

    return all_metrics


# ----------------------------
# main
# ----------------------------
def main():
    RESULTS_BASE = Path(require_env("RESULTS_BASE"))
    DATASET_NAME = require_env("DATASET_NAME")
    MODE = require_env("MODE")
    AUGMENTATION = require_env("AUGMENTATION")

    models = parse_model_names()

    # Decide mode based on number of models.
    # - 1 model => behave like old script (write inside model folder)
    # - 2+ models => write dataset-level comparison folder
    if len(models) == 1:
        model = models[0]
        model_dir = model_to_dirname(model)

        exp_root = RESULTS_BASE / DATASET_NAME / model_dir / MODE / AUGMENTATION
        if not exp_root.exists():
            raise RuntimeError(f"Experiment root not found: {exp_root}")

        all_metrics = collect_metrics_for_one_model(exp_root)
        if not all_metrics:
            raise RuntimeError(f"No test metrics found for model '{model}' under {exp_root}")

        # OLD BEHAVIOR OUTPUT LOCATION (inside model folder)
        out_dir = exp_root / "plots"
        out_dir.mkdir(exist_ok=True)

        for metric_name, values in sorted(all_metrics.items()):
            write_metric_plot_and_csv_single(
                out_dir=out_dir,
                dataset=DATASET_NAME,
                model_display=model,
                mode=MODE,
                augmentation=AUGMENTATION,
                metric_name=metric_name,
                values=values,
            )
            print(f"[INFO] Saved {metric_name} plot and CSV")

        print("[DONE] Single-model plots written to:", out_dir)
        return

    # 2+ models => comparison mode
    per_model_metrics: Dict[str, Dict[str, List[Tuple[int, float, float]]]] = {}
    for model in models:
        model_dir = model_to_dirname(model)
        exp_root = RESULTS_BASE / DATASET_NAME / model_dir / MODE / AUGMENTATION
        if not exp_root.exists():
            print(f"[WARN] Missing exp root for model '{model}': {exp_root} -> skipping model")
            continue

        m = collect_metrics_for_one_model(exp_root)
        if not m:
            print(f"[WARN] No usable metrics for model '{model}' under {exp_root} -> skipping model")
            continue

        per_model_metrics[model] = m
        print(f"[INFO] Loaded model '{model}' from {exp_root}")

    if not per_model_metrics:
        raise RuntimeError("No models had usable data; nothing to plot")
        
    model_order = [m for m in models if m in per_model_metrics]
    # COMPARISON OUTPUT LOCATION (dataset-level)
    comp_root = RESULTS_BASE / DATASET_NAME / "_comparisons" / MODE / AUGMENTATION
    out_dir = comp_root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Union of metrics
    metric_names = set()
    for mm in per_model_metrics.values():
        metric_names.update(mm.keys())
    metric_names = sorted(metric_names)

    for metric_name in metric_names:
        per_model_values: Dict[str, List[Tuple[int, float, float]]] = {}
        for model_display, metrics_dict in per_model_metrics.items():
            if metric_name in metrics_dict:
                per_model_values[model_display] = metrics_dict[metric_name]

        if not per_model_values:
            continue

        write_metric_plot_and_csv_multi(
            out_dir=out_dir,
            dataset=DATASET_NAME,
            mode=MODE,
            augmentation=AUGMENTATION,
            metric_name=metric_name,
            per_model_values=per_model_values,
            model_order=model_order,
        )
        print(f"[INFO] Saved multi-model plot + CSV for metric '{metric_name}'")

    print("[DONE] Comparison plots written to:", out_dir)


if __name__ == "__main__":
    main()
