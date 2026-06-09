#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

def require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"{name} must be set in the environment")
    return v

def mktemp_dir() -> str:
    return subprocess.check_output(["mktemp", "-d"]).decode().strip()

def parse_levels(s: str) -> list[int]:
    # "0,2" -> [0,2]
    out = []
    for part in s.split(","):
        part = part.strip()
        if part == "":
            continue
        out.append(int(part))
    return sorted(set(out))

AUG_ENV_VARS = [
    "BLUR_KERNEL", "BLUR_SIGMA",
    "CJ_BRIGHTNESS", "CJ_CONTRAST", "CJ_SATURATION", "CJ_HUE",
    "BW_ENABLED",
    "GAMMA",
    "JPEG_QUALITY",
    "ROTATE_ANGLES", "ROTATE_P",
    "PEPPER_P",
]

def clear_all_aug_env(env: dict) -> None:
    for k in AUG_ENV_VARS:
        env.pop(k, None)

def main():
    DATASET_NAME = require_env("DATASET_NAME")

    # -------- base config --------
    if DATASET_NAME == "patch_camelyon":
        BASE_CONFIG = "/yourpath/code/eva/configs/vision/pathology/offline/classification/patch_camelyon.yaml"
    elif DATASET_NAME == "crc":
        BASE_CONFIG = "/yourpath/code/eva/configs/vision/pathology/offline/classification/crc.yaml"
    else:
        raise RuntimeError(f"Unsupported DATASET_NAME={DATASET_NAME}")

    MAKE_EXAMPLE_SCRIPT = "/yourpath/code/make_example_image.py"

    # -------- required env --------
    MODEL_NAME = require_env("MODEL_NAME")
    DATA_ROOT = require_env("DATA_ROOT")
    RESULTS_BASE = Path(require_env("RESULTS_BASE"))

    MODE = require_env("MODE")  # keep type2 for these audits
    if MODE not in ("type1", "type2"):
        raise RuntimeError(f"Unknown MODE={MODE} (expected type1 or type2)")

    AUG = require_env("AUGMENTATION")  # color_jitter / jpeg_compression / pepper_noise
    NORMALIZE_MEAN = require_env("NORMALIZE_MEAN")
    NORMALIZE_STD = require_env("NORMALIZE_STD")
    IN_FEATURES = require_env("IN_FEATURES")

    # --- audit controls ---
    FORCE_OVERRIDE_BASELINE = os.environ.get("FORCE_OVERRIDE_BASELINE") == "1"
    MODE_TAG = os.environ.get("MODE_TAG", MODE)  # output folder tag
    OVERRIDE_VARIANT = os.environ.get("OVERRIDE_VARIANT", "old").lower()  # old|new
    LEVELS = parse_levels(os.environ.get("LEVELS", "0,2"))  # default audit: 0 and 2

    if OVERRIDE_VARIANT not in ("old", "new"):
        raise RuntimeError("OVERRIDE_VARIANT must be 'old' or 'new'")

    # -------- YAML path sets (old vs new) --------
    # Set these to your actual files:
    if DATASET_NAME == "patch_camelyon":
        PATHS_OLD = {
            "color_jitter": "/yourpath/code/robustness/patchcam_colorjitter_type2_env.yaml",
            "jpeg_compression": "/yourpath/code/robustness/patchcam_jpeg_type2_env.yaml",
            "pepper_noise": "/yourpath/code/robustness/patchcam_pepper_type2_env.yaml",
        }
        # NEW: point to your corrected versions (suggest naming *_preprocfix.yaml)
        PATHS_NEW = {
            "color_jitter": "/yourpath/code/robustness/patchcam_colorjitter_type2_preprocfix_env.yaml",
            "jpeg_compression": "/yourpath/code/robustness/patchcam_jpeg_type2_preprocfix_env.yaml",
            "pepper_noise": "/yourpath/code/robustness/patchcam_pepper_type2_preprocfix_env.yaml",
        }
    else:  # crc
        PATHS_OLD = {
            "color_jitter": "/yourpath/code/robustness/crc_colorjitter_type2_env.yaml",
            "jpeg_compression": "/yourpath/code/robustness/crc_jpeg_type2_env.yaml",
            "pepper_noise": "/yourpath/code/robustness/crc_pepper_type2_env.yaml",
        }
        PATHS_NEW = {
            "color_jitter": "/yourpath/code/robustness/crc_colorjitter_type2_preprocfix_env.yaml",
            "jpeg_compression": "/yourpath/code/robustness/crc_jpeg_type2_preprocfix_env.yaml",
            "pepper_noise": "/yourpath/code/robustness/crc_pepper_type2_preprocfix_env.yaml",
        }

    if AUG not in PATHS_OLD:
        raise RuntimeError(f"Unknown AUGMENTATION={AUG}. Expected one of: {list(PATHS_OLD)}")

    override_yaml = (PATHS_NEW if OVERRIDE_VARIANT == "new" else PATHS_OLD)[AUG]
    if not Path(override_yaml).exists():
        raise RuntimeError(f"Override YAML not found: {override_yaml}")

    # -------- audit levels (define "level-02" params) --------
    # We map requested levels to parameters. Level 0 = None baseline/no-aug.
    LEVEL_PARAMS = {
        "color_jitter": {0: None, 2: (0.4, 0.4, 0.4, 0.10)},
        "jpeg_compression": {0: None, 2: 50},
        "pepper_noise": {0: None, 2: 0.20},
    }
    params_map = LEVEL_PARAMS[AUG]

    def set_cj_env(env: dict, params):
        b, c, s, h = params
        env["CJ_BRIGHTNESS"] = str(b)
        env["CJ_CONTRAST"] = str(c)
        env["CJ_SATURATION"] = str(s)
        env["CJ_HUE"] = str(h)

    def set_jpeg_env(env: dict, q):
        env["JPEG_QUALITY"] = str(int(q))

    def set_pepper_env(env: dict, p):
        env["PEPPER_P"] = str(float(p))

    SET_ENV = {
        "color_jitter": set_cj_env,
        "jpeg_compression": set_jpeg_env,
        "pepper_noise": set_pepper_env,
    }[AUG]

    # -------- output structure (separated by MODE_TAG) --------
    # include override variant in folder to prevent mixing
    sweep_root = RESULTS_BASE / DATASET_NAME / MODEL_NAME.replace("/", "__") / MODE_TAG / OVERRIDE_VARIANT / AUG
    sweep_root.mkdir(parents=True, exist_ok=True)

    # -------- base env for subprocesses --------
    base_env = os.environ.copy()
    CODE_ROOT = "/yourpath/code"
    base_env["PYTHONPATH"] = CODE_ROOT + (":" + base_env["PYTHONPATH"] if "PYTHONPATH" in base_env else "")

    base_env["MODEL_NAME"] = MODEL_NAME
    base_env["DATASET_NAME"] = DATASET_NAME
    base_env["DATA_ROOT"] = DATA_ROOT
    base_env["RESULTS_BASE"] = str(RESULTS_BASE)
    base_env["MODE"] = MODE
    base_env["AUGMENTATION"] = AUG
    base_env["NORMALIZE_MEAN"] = NORMALIZE_MEAN
    base_env["NORMALIZE_STD"] = NORMALIZE_STD
    base_env["IN_FEATURES"] = IN_FEATURES
    base_env["N_DATA_WORKERS"] = os.environ.get("N_DATA_WORKERS", "8")

    for level in LEVELS:
        if level not in params_map:
            raise RuntimeError(f"Unsupported audit level {level} for {AUG}. Supported: {sorted(params_map)}")

        params = params_map[level]
        level_dir = sweep_root / f"level-{level:02d}"
        level_dir.mkdir(parents=True, exist_ok=True)

        run_env = base_env.copy()
        clear_all_aug_env(run_env)

        run_env["OUTPUT_ROOT"] = str(level_dir)
        run_env["LIGHTNING_ROOT"] = str(level_dir)
        run_env["EMBEDDINGS_ROOT"] = mktemp_dir()

        if params is None:
            print(f"\n=== EVA | {AUG} | {MODE_TAG} | {OVERRIDE_VARIANT} | level-{level:02d} | baseline(no-aug) ===")
            if FORCE_OVERRIDE_BASELINE:
                cmd = ["eva", "predict_fit", "--config", BASE_CONFIG, "--config", override_yaml]
            else:
                cmd = ["eva", "predict_fit", "--config", BASE_CONFIG]
        else:
            print(f"\n=== EVA | {AUG} | {MODE_TAG} | {OVERRIDE_VARIANT} | level-{level:02d} | params={params} ===")
            SET_ENV(run_env, params)
            cmd = ["eva", "predict_fit", "--config", BASE_CONFIG, "--config", override_yaml]

        subprocess.run(cmd, check=True, env=run_env)

        examples_dir = level_dir / "examples"
        examples_dir.mkdir(parents=True, exist_ok=True)
        run_env["EXAMPLES_DIR"] = str(examples_dir)
        try:
            subprocess.run(["python3", MAKE_EXAMPLE_SCRIPT], check=True, env=run_env)
        except subprocess.CalledProcessError as e:
            print(f"[WARN] Example image generation failed: {e}")

    print("\nDONE.")

if __name__ == "__main__":
    main()
