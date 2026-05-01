from pathlib import Path

# -----------------------------
# Project configuration
# -----------------------------
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16

BASE_EPOCHS = 8
BASE_LR = 1e-3

FINE_TUNE_EPOCHS = 5
FINE_TUNE_LR = 1e-5

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Update this manually if your folder is somewhere else.
CANDIDATE_DATA_DIRS = [
    Path.cwd() / "handbags_data",
    Path.cwd().parent / "handbags_data",
    Path.home() / "Desktop" / "handbags" / "handbags_data",
    Path("/mnt/data/handbags_data"),
]

DATA_DIR = next((p for p in CANDIDATE_DATA_DIRS if p.exists()), None)

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find handbags_data. Set DATA_DIR manually in config.py."
    )

OUTPUT_DIR = Path.home() / "Desktop" / "handbags" / "handbag_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_CHECKPOINT_PATH = OUTPUT_DIR / "best_resnet18_handbags.pt"
FINE_TUNE_CHECKPOINT_PATH = OUTPUT_DIR / "best_finetuned_resnet18_handbags.pt"

DISPLAY_NAME_MAP = {
    "bottega_veneta_mini_jodie": "Bottega Veneta Mini Jodie",
    "chanel_255": "Chanel 255",
    "fendi_baguette": "Fendi Baguette",
    "gucci_jackie_hobo": "Gucci Jackie Hobo",
    "lady_dior": "Lady Dior",
    "prada_cleo": "Prada Cleo",
    "prado_cleo": "Prada Cleo",  # Dataset folder uses this spelling.
    "negative": "Negative",
}
