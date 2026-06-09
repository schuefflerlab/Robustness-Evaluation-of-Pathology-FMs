import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1. Load file
# =========================
file_path = "all_plot_data_reformatted.xlsx"   # change accorting to the respective result compilation file
df = pd.read_excel(file_path)

model_name_map = {
    "mahmood_uni2_h": "Uni2",
    "prov_gigapath": "Prov-Gigapath",
    "paige_virchow2": "Virchow2",
    "pathryoshka-b": "Pathryoshka",
    "owkin_phikon_v2": "PhikonV2",
}

df["model"] = df["model"].replace(model_name_map)
# =========================
# 2. Keep only AUROC
# =========================
required_cols = ["dataset", "mode", "augmentation", "metric", "model", "level", "mean"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.copy()
df["level"] = pd.to_numeric(df["level"], errors="coerce")
df["mean"] = pd.to_numeric(df["mean"], errors="coerce")

if df["level"].isna().any():
    raise ValueError("Some 'level' values could not be parsed as numeric.")
if df["mean"].isna().any():
    raise ValueError("Some 'mean' values could not be parsed as numeric.")

df = df[df["metric"] == "AUROC"].copy()

# =========================
# 3. Sanity checks
# =========================
expected_datasets = {"crc", "patch_camelyon"}
expected_modes = {"type1", "type2"}
expected_augs = {
    "black_white",
    "color_jitter",
    "gamma",
    "gaussian_blur",
    "jpeg_compression",
    "pepper_noise",
    "random_rotate90",
}

datasets_found = set(df["dataset"].unique())
modes_found = set(df["mode"].unique())
augs_found = set(df["augmentation"].unique())
models_found = sorted(df["model"].unique())

print("Datasets found:", datasets_found)
print("Modes found:", modes_found)
print("Augmentations found:", augs_found)
print("Models found:", models_found)

if datasets_found != expected_datasets:
    raise ValueError(f"Unexpected datasets: {datasets_found}")
if modes_found != expected_modes:
    raise ValueError(f"Unexpected modes: {modes_found}")
if augs_found != expected_augs:
    raise ValueError(f"Unexpected augmentations: {augs_found}")
if len(models_found) != 5:
    raise ValueError(f"Expected 5 models, found {len(models_found)}")

# =========================
# 4. Match each corrupted AUROC row to its exact clean baseline
# =========================
group_cols = ["dataset", "mode", "model", "augmentation"]

baseline_counts = (
    df.assign(is_baseline=(df["level"] == 0))
      .groupby(group_cols)["is_baseline"]
      .sum()
)

bad_baselines = baseline_counts[baseline_counts != 1]
if not bad_baselines.empty:
    raise ValueError(
        "Some AUROC groups do not have exactly one baseline row.\n"
        f"{bad_baselines}"
    )

baseline_df = (
    df[df["level"] == 0][group_cols + ["mean"]]
    .rename(columns={"mean": "baseline_auroc"})
)

merged = df.merge(
    baseline_df,
    on=group_cols,
    how="left",
    validate="many_to_one"
)

if merged["baseline_auroc"].isna().any():
    raise ValueError("Some rows could not be matched to a baseline AUROC row.")

# Only corrupted levels contribute to the robustness score
corrupted = merged[merged["level"] > 0].copy()

# Signed drop: positive = corruption hurts, negative = corruption helps
corrupted["auroc_drop"] = corrupted["baseline_auroc"] - corrupted["mean"]

# =========================
# 5. Average across levels within each perturbation
# =========================
per_augmentation = (
    corrupted.groupby(group_cols)
    .agg(
        baseline_auroc=("baseline_auroc", "first"),
        n_corrupted_levels=("level", "nunique"),
        levels_used=("level", lambda x: ",".join(map(str, sorted(set(x))))),
        mean_drop_levels=("auroc_drop", "mean"),
        median_drop_levels=("auroc_drop", "median"),
    )
    .reset_index()
)

# =========================
# 6. Average across perturbations within each dataset/mode/model
# =========================
per_dataset_model = (
    per_augmentation.groupby(["dataset", "mode", "model"])
    .agg(
        robustness_score=("mean_drop_levels", "mean"),
        median_robustness_score=("mean_drop_levels", "median"),
        n_perturbations=("mean_drop_levels", "size"),
    )
    .reset_index()
)

bad_pert_counts = per_dataset_model[per_dataset_model["n_perturbations"] != 7]
if not bad_pert_counts.empty:
    raise ValueError(
        "Some dataset/mode/model combinations do not have all 7 perturbations.\n"
        f"{bad_pert_counts}"
    )

# =========================
# 7. Build table for plotting: CRC, PatchCamelyon, Mean
# =========================
pivot = per_dataset_model.pivot_table(
    index=["mode", "model"],
    columns="dataset",
    values="robustness_score"
).reset_index()

pivot["mean_both"] = (pivot["crc"] + pivot["patch_camelyon"]) / 2

# Order models by combined Type2 robustness (lower is better)
type2_order = (
    pivot[pivot["mode"] == "type2"]
    .sort_values("mean_both", ascending=True)["model"]
    .tolist()
)

print("\nPlot values:")
print(pivot.sort_values(["mode", "model"]))

# =========================
# 8. Plot helper
# =========================
def make_plot(mode_name, out_file):
    plot_df = pivot[pivot["mode"] == mode_name].copy()
    plot_df["model"] = pd.Categorical(plot_df["model"], categories=type2_order, ordered=True)
    plot_df = plot_df.sort_values("model")

    x = np.arange(len(plot_df))
    width = 0.24

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, plot_df["crc"], width=width, label="CRC")
    plt.bar(x, plot_df["patch_camelyon"], width=width, label="PatchCamelyon")
    plt.bar(x + width, plot_df["mean_both"], width=width, label="Mean")

    plt.xticks(x, plot_df["model"], rotation=25, ha="right")
    plt.ylabel("Mean AUROC drop from clean baseline")
    plt.title(
        f"Corruption robustness by model ({mode_name})\n"
        "Lower is better. Score = mean AUROC drop across all perturbations and available corruption levels."
    )
    plt.axhline(0, linewidth=1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.show()

# =========================
# 9. Create exactly 2 plots
# =========================
make_plot("type1", "robustness_type1_auroc.png")
make_plot("type2", "robustness_type2_auroc.png")

# =========================
# 10. Save compact results
# =========================
out_excel = "robustness_auroc_two_plots.xlsx"
with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
    corrupted.to_excel(writer, sheet_name="per_level_auroc_drops", index=False)
    per_augmentation.to_excel(writer, sheet_name="per_augmentation_auroc", index=False)
    per_dataset_model.to_excel(writer, sheet_name="per_dataset_model_auroc", index=False)
    pivot.to_excel(writer, sheet_name="plot_values", index=False)

print("\nSaved:")
print("-", out_excel)
print("-", "robustness_type1_auroc.png")
print("-", "robustness_type2_auroc.png")
