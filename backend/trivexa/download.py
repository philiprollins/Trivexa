"""Download model weights Trivexa needs beyond the shared models folder.

    uv run python -m trivexa.download            # everything missing
    uv run python -m trivexa.download t2mv i2mv  # specific items

Downloads resume if interrupted and are checked against the published SHA-256.
"""

import hashlib
import sys

from huggingface_hub import HfApi, hf_hub_download

from .config import MV_ADAPTER_DIR

MV_ADAPTER_REPO = "huanngzh/mv-adapter"
MV_ADAPTER_REVISION = "6de4033df6b53366f3c009d22f5ec434bb55e59f"

ITEMS = {
    "t2mv": (MV_ADAPTER_REPO, MV_ADAPTER_REVISION, "mvadapter_t2mv_sdxl.safetensors", MV_ADAPTER_DIR),
    "i2mv": (MV_ADAPTER_REPO, MV_ADAPTER_REVISION, "mvadapter_i2mv_sdxl.safetensors", MV_ADAPTER_DIR),
}


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name: str) -> None:
    repo, revision, filename, dest = ITEMS[name]
    target = dest / filename
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    expected = next(s.lfs.sha256 for s in info.siblings if s.rfilename == filename)
    if target.is_file() and _sha256(target) == expected:
        print(f"{name}: already present and verified")
        return
    print(f"{name}: downloading {filename} → {dest}")
    hf_hub_download(repo, filename, revision=revision, local_dir=dest)
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
