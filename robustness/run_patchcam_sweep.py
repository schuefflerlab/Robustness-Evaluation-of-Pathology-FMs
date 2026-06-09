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

AUG_ENV_VARS = [
    # blur
    "BLUR_KERNEL", "BLUR_SIGMA",
    # color jitter
    "CJ_BRIGHTNESS", "CJ_CONTRAST", "CJ_SATURATION", "CJ_HUE",
    # bw
    "BW_ENABLED",
    # gamma
    "GAMMA",
    # compression
    "JPEG_QUALITY",
    # rotation
    "ROTATE_ANGLES", "ROTATE_P",
    "PEPPER_P",
]

def clear_all_aug_env(env: dict) -> None:
    for k in AUG_ENV_VARS:
        env.pop(k, None)

def main():
    DATASET_NAME = require_env("DATASET_NAME")
    # -------- paths --------
    if DATASET_NAME == "patch_camelyon":
        BASE_CONFIG = "/yourpath/code/eva/configs/vision/pathology/offline/classification/patch_camelyon.yaml"
    elif DATASET_NAME == "crc":
        BASE_CONFIG = "/yourpath/code/eva/configs/vision/pathology/offline/classification/crc.yaml"
    else:
        raise RuntimeError(f"Unsupported DATASET_NAME={DATASET_NAME}")
    
    MAKE_EXAMPLE_SCRIPT = "/yourpath/code/make_example_image.py"
    
    PATHRYOSHKA_BACKBONE_OVERRIDE = "/yourpath/code/robustness/pathryoshka_timm_backbone.yaml"
    PATHRYOSHKA_OVERRIDE_CRC      = "/yourpath/code/robustness/pathryoshka_timm_backbone_crc.yaml"
    
    if DATASET_NAME == "patch_camelyon":
        BLUR_TYPE1 = "/yourpath/code/robustness/patchcam_blur_env.yaml"
        BLUR_TYPE2 = "/yourpath/code/robustness/patchcam_blur_type2_env.yaml"
        CJ_TYPE1 = "/yourpath/code/robustness/patchcam_colorjitter_env.yaml"
        CJ_TYPE2 = "/yourpath/code/robustness/patchcam_colorjitter_type2_env.yaml"
        BW_TYPE1 = "/yourpath/code/robustness/patchcam_bw_env.yaml"
        BW_TYPE2 = "/yourpath/code/robustness/patchcam_bw_type2_env.yaml"
        GAMMA_TYPE1 = "/yourpath/code/robustness/patchcam_gamma_env.yaml"
        GAMMA_TYPE2 = "/yourpath/code/robustness/patchcam_gamma_type2_env.yaml"
        JPEG_TYPE1  = "/yourpath/code/robustness/patchcam_jpeg_env.yaml"
        JPEG_TYPE2  = "/yourpath/code/robustness/patchcam_jpeg_type2_env.yaml"
        ROT_TYPE1 = "/yourpath/code/robustness/patchcam_rotate_env.yaml"
        ROT_TYPE2 = "/yourpath/code/robustness/patchcam_rotate_type2_env.yaml"
        PEPPER_TYPE1 = "/yourpath/code/robustness/patchcam_pepper_env.yaml"
        PEPPER_TYPE2 = "/yourpath/code/robustness/patchcam_pepper_type2_env.yaml"
    
    elif DATASET_NAME == "crc":
        BLUR_TYPE1 = "/yourpath/code/robustness/crc_blur_env.yaml"
        BLUR_TYPE2 = "/yourpath/code/robustness/crc_blur_type2_env.yaml"
        CJ_TYPE1 = "/yourpath/code/robustness/crc_colorjitter_env.yaml"
        CJ_TYPE2 = "/yourpath/code/robustness/crc_colorjitter_type2_env.yaml"
        BW_TYPE1 = "/yourpath/code/robustness/crc_bw_env.yaml"
        BW_TYPE2 = "/yourpath/code/robustness/crc_bw_type2_env.yaml"
        GAMMA_TYPE1 = "/yourpath/code/robustness/crc_gamma_env.yaml"
        GAMMA_TYPE2 = "/yourpath/code/robustness/crc_gamma_type2_env.yaml"
        JPEG_TYPE1  = "/yourpath/code/robustness/crc_jpeg_env.yaml"
        JPEG_TYPE2  = "/yourpath/code/robustness/crc_jpeg_type2_env.yaml"
        ROT_TYPE1 = "/yourpath/code/robustness/crc_rotate_env.yaml"
        ROT_TYPE2 = "/yourpath/code/robustness/crc_rotate_type2_env.yaml"
        PEPPER_TYPE1 = "/yourpath/code/robustness/crc_pepper_env.yaml"
        PEPPER_TYPE2 = "/yourpath/code/robustness/crc_pepper_type2_env.yaml"
    # -------- required env --------
    MODEL_NAME = require_env("MODEL_NAME")
    USE_PATHRYOSHKA_TIMM = (MODEL_NAME == "hf-hub:SchuefflerLab/pathryoshka-b")
    DATASET_NAME = require_env("DATASET_NAME")
    DATA_ROOT = require_env("DATA_ROOT")
    RESULTS_BASE = Path(require_env("RESULTS_BASE"))
 
    MODE = require_env("MODE")          # type1 or type2
    AUG = require_env("AUGMENTATION")   # gaussian_blur / color_jitter / black_white
 
    NORMALIZE_MEAN = require_env("NORMALIZE_MEAN")
    NORMALIZE_STD = require_env("NORMALIZE_STD")
    IN_FEATURES = require_env("IN_FEATURES")
 
    if MODE not in ("type1", "type2"):
        raise RuntimeError(f"Unknown MODE={MODE} (expected type1 or type2)")
 
    # -------- augmentation registry --------
    BLUR_LEVELS = {
        0: None,
        1: (3, 0.5),
        2: (5, 1.0),
        3: (7, 2.0),
        4: (9, 3.0),
        5: (11, 4.0),
    }
 
    CJ_LEVELS = {
        0: None,
        1: (0.2, 0.2, 0.2, 0.05),
        2: (0.4, 0.4, 0.4, 0.10),
        3: (0.6, 0.6, 0.6, 0.15),
    }
 
    BW_LEVELS = {0: None, 1: True}
    
    GAMMA_LEVELS = {0: None, 1: 0.4, 2: 0.8, 3: 1.2, 4: 1.6, 5: 2}
    
    JPEG_LEVELS = {0: None, 1: 75, 2: 50, 3: 30}
    
    ROT_LEVELS = {0: None, 1: {"angles": "[90,180,270]", "p": 1.0}}
    
    PEPPER_LEVELS = {0: None, 1: 0.10, 2: 0.20, 3: 0.30}
 
    def set_pepper_env(env: dict, p):
        env["PEPPER_P"] = str(float(p))
    
    def set_jpeg_env(env: dict, q):
        env["JPEG_QUALITY"] = str(int(q))
 
    def set_blur_env(env: dict, params):
        kernel, sigma = params
        env["BLUR_KERNEL"] = str(kernel)
        env["BLUR_SIGMA"] = str(sigma)
 
    def set_cj_env(env: dict, params):
        b, c, s, h = params
        env["CJ_BRIGHTNESS"] = str(b)
        env["CJ_CONTRAST"] = str(c)
        env["CJ_SATURATION"] = str(s)
        env["CJ_HUE"] = str(h)
 
    def set_bw_env(env: dict, _params):
        env["BW_ENABLED"] = "1"
        
    def set_gamma_env(env: dict, g):
        env["GAMMA"] = str(g)
 
    def set_rot_env(env: dict, params):
        env["ROTATE_ANGLES"] = params["angles"]
        env["ROTATE_P"] = str(params["p"])
 
    REGISTRY = {
        "gaussian_blur": {
            "levels": BLUR_LEVELS,
            "set_env": set_blur_env,
            "config": {"type1": BLUR_TYPE1, "type2": BLUR_TYPE2},
        },
        "color_jitter": {
            "levels": CJ_LEVELS,
            "set_env": set_cj_env,
            "config": {"type1": CJ_TYPE1, "type2": CJ_TYPE2},
        },
        "black_white": {
            "levels": BW_LEVELS,
            "set_env": set_bw_env,
            "config": {"type1": BW_TYPE1, "type2": BW_TYPE2},
        },
        "gamma": {
            "levels": GAMMA_LEVELS,
            "set_env": set_gamma_env,
            "config": {"type1": GAMMA_TYPE1, "type2": GAMMA_TYPE2},
        },
        "jpeg_compression": {
            "levels": JPEG_LEVELS,
            "set_env": set_jpeg_env,
            "config": {"type1": JPEG_TYPE1, "type2": JPEG_TYPE2},
        },
        "random_rotate90": {
            "levels": ROT_LEVELS,
            "set_env": set_rot_env,
            "config": {"type1": ROT_TYPE1, "type2": ROT_TYPE2},
        },
        "pepper_noise": {
            "levels": PEPPER_LEVELS,
            "set_env": set_pepper_env,
            "config": {"type1": PEPPER_TYPE1, "type2": PEPPER_TYPE2},
        },
    }
 
    if AUG not in REGISTRY:
        raise RuntimeError(f"Unknown AUGMENTATION={AUG}. Expected one of: {list(REGISTRY)}")
 
    entry = REGISTRY[AUG]
    override_yaml = entry["config"][MODE]
 
    # -------- output structure (plotter-compatible) --------
    sweep_root = RESULTS_BASE / DATASET_NAME / MODEL_NAME.replace("/", "__") / MODE / AUG
    sweep_root.mkdir(parents=True, exist_ok=True)
 
    # -------- base env passed to subprocesses (explicit, stable) --------
    base_env = os.environ.copy()
    CODE_ROOT = "/yourpath/code"
    base_env["PYTHONPATH"] = CODE_ROOT + (":" + base_env["PYTHONPATH"] if "PYTHONPATH" in base_env else "")

    # ensure required core vars are present/consistent in subprocess env
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

    for level, params in entry["levels"].items():
        level_dir = sweep_root / f"level-{level:02d}"
        level_dir.mkdir(parents=True, exist_ok=True)

        run_env = base_env.copy()
        clear_all_aug_env(run_env)  # <- critical: prevents leakage across levels/augs

        run_env["OUTPUT_ROOT"] = str(level_dir)
        run_env["LIGHTNING_ROOT"] = str(level_dir)
        run_env["EMBEDDINGS_ROOT"] = mktemp_dir()

        # build cmd in a single place to avoid ordering bugs
        cmd = ["eva", "predict_fit", "--config", BASE_CONFIG]

        # apply model override early
        if USE_PATHRYOSHKA_TIMM:
            if DATASET_NAME == "patch_camelyon":
                cmd += ["--config", PATHRYOSHKA_BACKBONE_OVERRIDE]
            elif DATASET_NAME == "crc":
                cmd += ["--config", PATHRYOSHKA_OVERRIDE_CRC]
            else:
                raise RuntimeError(f"Unsupported DATASET_NAME for Pathryoshka override: {DATASET_NAME}")

        if params is None:
            print(f"\n=== EVA | {AUG} | {MODE} | level-{level:02d} | baseline ===")
        else:
            print(f"\n=== EVA | {AUG} | {MODE} | level-{level:02d} | params={params} ===")
            entry["set_env"](run_env, params)
            # augmentation override LAST so it cannot be clobbered
            cmd += ["--config", override_yaml]

        subprocess.run(cmd, check=True, env=run_env)


        # Example image after EVA (uses same clean env)
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
