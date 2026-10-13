"""Download model weights Trivexa needs beyond the shared models folder.

    uv run python -m trivexa.download             # everything
    uv run python -m trivexa.download ig2mv lama  # specific items (see ITEMS)

Downloads resume if interrupted and are checked against a known SHA-256.
"""

import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from .config import MV_ADAPTER_DIR, TEXTURE_WEIGHTS_DIR

MV_ADAPTER_REPO = "huanngzh/mv-adapter"
MV_ADAPTER_REVISION = "6de4033df6b53366f3c009d22f5ec434bb55e59f"


@dataclass(frozen=True)
class HubFile:
    repo: str
    revision: str
    filename: str
    dest: Path  # checksum comes from the Hub's LFS metadata


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
}


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _expected_sha256(item: HubFile | UrlFile) -> str:
    if isinstance(item, UrlFile):
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
    item = ITEMS[name]
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
