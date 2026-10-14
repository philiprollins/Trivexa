import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Model files are shared with an existing ComfyUI install, but only read from disk.
MODELS_DIR = Path(os.environ.get("TRIVEXA_MODELS_DIR", Path.home() / "ComfyUI" / "models"))
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

DATA_DIR = Path(os.environ.get("TRIVEXA_DATA_DIR", ROOT / "data"))
OUTPUTS_DIR = DATA_DIR / "outputs"
DB_PATH = DATA_DIR / "trivexa.db"
# Small architecture configs/tokenizers diffusers needs to build a pipeline from a single checkpoint file.
HF_CONFIG_DIR = DATA_DIR / "hf-configs"
# Weights Trivexa downloads itself (e.g. MV-Adapter), as opposed to the shared MODELS_DIR.
MV_ADAPTER_DIR = DATA_DIR / "models" / "mv-adapter"

TEXTURE_WEIGHTS_DIR = DATA_DIR / "models" / "texture"  # RealESRGAN + LaMa for the upstream texture baker
MESHFLOW_DIR = DATA_DIR / "models" / "meshflow"  # config.yaml + model.pth (facebook/meshflow)
SKINTOKENS_DIR = DATA_DIR / "models" / "skintokens"  # experiments/*/…ckpt (VAST-AI/SkinTokens)

FRONTEND_DIST = ROOT / "frontend" / "dist"

# User-editable settings (see settings.toml at the repo root; TRIVEXA_SETTINGS points elsewhere).
SETTINGS_PATH = Path(os.environ.get("TRIVEXA_SETTINGS", ROOT / "settings.toml"))
_DEFAULTS = {"texture": {"baker": "upstream"}}


def _load_settings() -> dict:
    settings = {section: dict(values) for section, values in _DEFAULTS.items()}
    if SETTINGS_PATH.is_file():
        with SETTINGS_PATH.open("rb") as f:
            for section, values in tomllib.load(f).items():
                settings.setdefault(section, {}).update(values)
    return settings


SETTINGS = _load_settings()


def ensure_dirs() -> None:
    for d in (DATA_DIR, OUTPUTS_DIR, HF_CONFIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
