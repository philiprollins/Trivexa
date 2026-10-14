"""Download model weights Trivexa needs beyond the shared models folder.

    uv run python -m trivexa.download             # everything
    uv run python -m trivexa.download ig2mv lama  # specific items (see ITEMS; an item may be several files)

Downloads resume if interrupted and are checked against a known SHA-256.
"""

import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from .config import MESHFLOW_DIR, MV_ADAPTER_DIR, SKINTOKENS_DIR, TEXTURE_WEIGHTS_DIR, TRIPOSG_DIR, UNIMATE_DIR

MV_ADAPTER_REPO = "huanngzh/mv-adapter"
MV_ADAPTER_REVISION = "6de4033df6b53366f3c009d22f5ec434bb55e59f"
# Gated: accept the license on huggingface.co/facebook/meshflow, then `hf auth login`, before downloading.
MESHFLOW_REPO = "facebook/meshflow"
MESHFLOW_REVISION = "9249a90e4997e105e533d6f502453fa3b344676f"
SKINTOKENS_REPO = "VAST-AI/SkinTokens"
SKINTOKENS_REVISION = "79736cad0fd84de384d5eede659b4ebd24effe33"
TRIPOSG_REPO = "VAST-AI/TripoSG"
TRIPOSG_REVISION = "2c1c516d22d58db486a058d98d31bb6177344e06"
UNIMATE_REPO = "Linzhan/UniMate"
UNIMATE_REVISION = "174fc755ec248d0c98f27759b7a95648e7f05905"


@dataclass(frozen=True)
class HubFile:
    repo: str
    revision: str
    filename: str  # path in the repo; lands at dest/filename
    dest: Path
    sha256: str | None = None  # None: use the Hub's LFS metadata (small non-LFS files need it spelled out)


@dataclass(frozen=True)
class UrlFile:
    url: str
    sha256: str
    filename: str
    dest: Path


ITEMS = {
    "t2mv": HubFile(MV_ADAPTER_REPO, MV_ADAPTER_REVISION, "mvadapter_t2mv_sdxl.safetensors", MV_ADAPTER_DIR),
    "i2mv": HubFile(MV_ADAPTER_REPO, MV_ADAPTER_REVISION, "mvadapter_i2mv_sdxl.safetensors", MV_ADAPTER_DIR),
    # Geometry-guided variants, for texturing an existing mesh from an image (ig2mv) or a prompt (tg2mv).
    "ig2mv": HubFile(MV_ADAPTER_REPO, MV_ADAPTER_REVISION, "mvadapter_ig2mv_sdxl.safetensors", MV_ADAPTER_DIR),
    "tg2mv": HubFile(MV_ADAPTER_REPO, MV_ADAPTER_REVISION, "mvadapter_tg2mv_sdxl.safetensors", MV_ADAPTER_DIR),
    # The upstream texture baker's view upscaler and inpainter (the files MV-Adapter's README points to).
    "esrgan": UrlFile(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
        "RealESRGAN_x2plus.pth",
        TEXTURE_WEIGHTS_DIR,
    ),
    "lama": UrlFile(  # same file IOPaint uses (MD5 e3aa4aaa15225a33ec84f9f4bc47e500)
        "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
        "344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9",
        "big-lama.pt",
        TEXTURE_WEIGHTS_DIR,
    ),
    # Artist-style remeshing of generated models (both files form one bundle in data/models/meshflow/).
    "meshflow": HubFile(MESHFLOW_REPO, MESHFLOW_REVISION, "meshflow/model.pth", MESHFLOW_DIR.parent),
    "meshflow-config": HubFile(
        MESHFLOW_REPO,
        MESHFLOW_REVISION,
        "meshflow/config.yaml",
        MESHFLOW_DIR.parent,
        "8ad6180b747065775a1dd6f7d4949424b69d88eea298d43ba5599dbcd4d2a132",
    ),
    # Auto-rigging (MIT). The TokenRig checkpoint already holds the skin-token VAE, so upstream's separate
    # skin_vae_2_10_32768/last.ckpt (read only for its config, which engines/skintokens.py spells out) is skipped.
    "skintokens": HubFile(
        SKINTOKENS_REPO,
        SKINTOKENS_REVISION,
        "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
        SKINTOKENS_DIR,
    ),
    # Image -> 3D (MIT). The repo's diffusers layout is kept under data/models/triposg/; engines/triposg.py builds
    # the models itself, so model_index.json and the scheduler/preprocessor configs are not needed.
    "triposg": (
        HubFile(TRIPOSG_REPO, TRIPOSG_REVISION, "transformer/diffusion_pytorch_model.safetensors", TRIPOSG_DIR),
        HubFile(
            TRIPOSG_REPO,
            TRIPOSG_REVISION,
            "transformer/config.json",
            TRIPOSG_DIR,
            "54ad779e68f9bc7303122cdf291f6db48dec7c67e41d28c05ac359daf914128a",
        ),
        HubFile(TRIPOSG_REPO, TRIPOSG_REVISION, "vae/diffusion_pytorch_model.safetensors", TRIPOSG_DIR),
        HubFile(
            TRIPOSG_REPO,
            TRIPOSG_REVISION,
            "vae/config.json",
            TRIPOSG_DIR,
            "2fb6ba9b364d42c9681879dca0f7edcb978f2db974a1ca4b15d28f9b15a5baed",
        ),
        HubFile(TRIPOSG_REPO, TRIPOSG_REVISION, "image_encoder_dinov2/model.safetensors", TRIPOSG_DIR),
        HubFile(
            TRIPOSG_REPO,
            TRIPOSG_REVISION,
            "image_encoder_dinov2/config.json",
            TRIPOSG_DIR,
            "8ed81b0d5ffd2c8b2cecd7dc1326d3eefa409bc9d525537b6b2a3c70a992fd02",
        ),
    ),
    # Text -> motion on any skeleton (checkpoints CC BY-NC 4.0). The recommended UniML3D run, final step; its
    # flan-t5-base text encoder is not part of this item yet.
    "unimate": (
        HubFile(UNIMATE_REPO, UNIMATE_REVISION, "unimate_uniml3d_f60_v3/checkpoints/checkpoint_step_150000.pt", UNIMATE_DIR),
        HubFile(UNIMATE_REPO, UNIMATE_REVISION, "unimate_uniml3d_f60_v3/dataset_stats.npy", UNIMATE_DIR),
        HubFile(
            UNIMATE_REPO,
            UNIMATE_REVISION,
            "unimate_uniml3d_f60_v3/config.json",
            UNIMATE_DIR,
            "1ffbd379d3667de7d701ed6f3418f1c9d1a66f35d2653776255008c07c051aaa",
        ),
    ),
}


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _expected_sha256(item: HubFile | UrlFile) -> str:
    if item.sha256:
        return item.sha256
    info = HfApi().model_info(item.repo, revision=item.revision, files_metadata=True)
    return next(s.lfs.sha256 for s in info.siblings if s.rfilename == item.filename)


def _download_url(item: UrlFile, target: Path) -> None:
    """Stream to <target>.part, resuming from where a previous attempt stopped."""
    part = target.with_suffix(target.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(item.url, headers={"Range": f"bytes={done}-"} if done else {})
    with urllib.request.urlopen(req) as resp, part.open("ab" if resp.status == 206 else "wb") as out:
        while block := resp.read(1024 * 1024):
            out.write(block)
    part.replace(target)


def fetch(name: str) -> None:
    for item in ITEMS[name] if isinstance(ITEMS[name], tuple) else (ITEMS[name],):
        _fetch_file(name, item)


def _fetch_file(name: str, item: HubFile | UrlFile) -> None:
    target = item.dest / item.filename
    expected = _expected_sha256(item)
    if target.is_file() and _sha256(target) == expected:
        print(f"{name}: already present and verified")
        return
    print(f"{name}: downloading {item.filename} → {item.dest}")
    item.dest.mkdir(parents=True, exist_ok=True)
    if isinstance(item, HubFile):
        hf_hub_download(item.repo, item.filename, revision=item.revision, local_dir=item.dest)
    else:
        _download_url(item, target)
    actual = _sha256(target)
    if actual != expected:
        target.unlink()
        sys.exit(f"{name}: checksum mismatch (got {actual}, expected {expected}); deleted, please retry")
    print(f"{name}: verified")


if __name__ == "__main__":
    for item in sys.argv[1:] or ITEMS:
        if item not in ITEMS:
            sys.exit(f"Unknown item {item!r}; choose from {', '.join(ITEMS)}")
        fetch(item)
