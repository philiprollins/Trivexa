"""Discovery of model files on disk and architecture detection from safetensors headers."""

import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import CHECKPOINTS_DIR

MODEL_EXTS = {".safetensors"}


@dataclass
class Checkpoint:
    id: str  # path relative to the checkpoints dir; used as the public identifier
    name: str
    arch: str  # "sdxl" | "sd15" | "unknown"
    size: int

    def to_dict(self) -> dict:
        return asdict(self)


_arch_cache: dict[tuple[str, float, int], str] = {}


def _read_safetensors_keys(path: Path) -> list[str]:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        if header_len > 100_000_000:
            return []
        header = json.loads(f.read(header_len))
    return [k for k in header if k != "__metadata__"]


def detect_arch(path: Path) -> str:
    st = path.stat()
    key = (str(path), st.st_mtime, st.st_size)
    if key not in _arch_cache:
        try:
            keys = _read_safetensors_keys(path)
        except (OSError, ValueError, struct.error):
            keys = []
        if any(k.startswith("conditioner.embedders.1.") for k in keys):
            arch = "sdxl"
        elif any(k.startswith("cond_stage_model.transformer.") for k in keys):
            arch = "sd15"
        else:
            arch = "unknown"
        _arch_cache[key] = arch
    return _arch_cache[key]


def list_checkpoints() -> list[Checkpoint]:
    if not CHECKPOINTS_DIR.is_dir():
        return []
    out = []
    for p in sorted(CHECKPOINTS_DIR.rglob("*"), key=lambda p: p.name.lower()):
        if p.suffix.lower() not in MODEL_EXTS or not p.is_file():
            continue
        out.append(
            Checkpoint(
                id=str(p.relative_to(CHECKPOINTS_DIR)),
                name=p.stem,
                arch=detect_arch(p),
                size=p.stat().st_size,
            )
        )
    return out


def resolve_checkpoint(model_id: str, arch: str) -> Path:
    """Map a public checkpoint id back to a file, refusing anything not in the listing."""
    for ckpt in list_checkpoints():
        if ckpt.id == model_id:
            if ckpt.arch != arch:
                raise ValueError(f"{ckpt.name} is a {ckpt.arch} model, expected {arch}")
            return CHECKPOINTS_DIR / ckpt.id
    raise ValueError(f"Unknown model: {model_id}")
