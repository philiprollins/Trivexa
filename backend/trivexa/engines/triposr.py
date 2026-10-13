"""Single image -> textured-by-vertex-colour mesh with TripoSR (vendored code, local weights)."""

import logging
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf
from PIL import Image

from ..config import MODELS_DIR
from ..vendor.tsr.system import TSR
from ..vendor.tsr.utils import resize_foreground, to_gradio_3d_orientation

log = logging.getLogger(__name__)

WEIGHTS_DIR = MODELS_DIR / "triposr"  # config.yaml, model.ckpt, dino/config.json (stabilityai/TripoSR)


@dataclass
class TripoSRParams:
    image: Image.Image  # RGBA with the background already removed, or RGB to use as-is
    foreground_ratio: float = 0.85
    mc_resolution: int = 256
    threshold: float = 25.0


# transformers 5 renamed ViT's per-layer modules; TripoSR's checkpoint uses the old names.
_VIT_RENAMES = (
    (r"\.encoder\.layer\.(\d+)\.attention\.attention\.query\.", r".layers.\1.attention.q_proj."),
    (r"\.encoder\.layer\.(\d+)\.attention\.attention\.key\.", r".layers.\1.attention.k_proj."),
    (r"\.encoder\.layer\.(\d+)\.attention\.attention\.value\.", r".layers.\1.attention.v_proj."),
    (r"\.encoder\.layer\.(\d+)\.attention\.output\.dense\.", r".layers.\1.attention.o_proj."),
    (r"\.encoder\.layer\.(\d+)\.intermediate\.dense\.", r".layers.\1.mlp.fc1."),
    (r"\.encoder\.layer\.(\d+)\.output\.dense\.", r".layers.\1.mlp.fc2."),
    (r"\.encoder\.layer\.(\d+)\.", r".layers.\1."),
)


def _adapt_state_dict(state: dict, model: torch.nn.Module) -> dict:
    expected = set(model.state_dict())
    if all(k in expected for k in state):
        return state
    adapted = {}
    for key, value in state.items():
        if key not in expected:
            for pattern, repl in _VIT_RENAMES:
                new = re.sub(pattern, repl, key)
                if new != key:
                    key = new
                    break
        adapted[key] = value
    return adapted


def available() -> bool:
    return all((WEIGHTS_DIR / f).is_file() for f in ("config.yaml", "model.ckpt", "dino/config.json"))


def prepare_input(image: Image.Image, foreground_ratio: float) -> Image.Image:
    """Centre the subject at the given fill ratio and flatten onto grey, as in the upstream demo."""
    rgba = image.convert("RGBA")
    if rgba.getextrema()[3][0] < 255:  # has transparency: crop to the subject
        rgba = resize_foreground(rgba, foreground_ratio)
    arr = np.asarray(rgba).astype(np.float32) / 255
    rgb = arr[..., :3] * arr[..., 3:] + (1 - arr[..., 3:]) * 0.5
    return Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8))


class TripoSREngine:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model: TSR | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        log.info("Loading TripoSR from %s", WEIGHTS_DIR)
        cfg = OmegaConf.load(WEIGHTS_DIR / "config.yaml")
        cfg.image_tokenizer.pretrained_model_name_or_path = str(WEIGHTS_DIR / "dino")
        OmegaConf.resolve(cfg)
        model = TSR(cfg)
        state = torch.load(WEIGHTS_DIR / "model.ckpt", map_location="cpu", weights_only=True)
        model.load_state_dict(_adapt_state_dict(state, model))
        model.renderer.set_chunk_size(8192)
        self.model = model.to(self.device).eval()

    def unload(self) -> None:
        if self.model is None:
            return
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate(self, p: TripoSRParams, on_step: Callable[[int, int, str | None], None]) -> tuple[trimesh.Trimesh, Image.Image]:
        """Return the mesh (Y-up, glTF convention) and the processed input image."""
        self.load()
        cond = prepare_input(p.image, p.foreground_ratio)
        on_step(1, 3, None)
        scene_codes = self.model([cond], device=self.device)
        on_step(2, 3, None)
        mesh = self.model.extract_mesh(scene_codes, True, resolution=p.mc_resolution, threshold=p.threshold)[0]
        if mesh.volume < 0:  # keep normals pointing outwards whatever the marching-cubes winding
            mesh.invert()
        mesh = to_gradio_3d_orientation(mesh)
        # TripoSR's subject then faces -Z; turn it to face +Z, the glTF "front" (and what MV-Adapter expects).
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))
        on_step(3, 3, None)
        return mesh, cond
