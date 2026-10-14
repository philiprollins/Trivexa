"""Single image -> untextured mesh with TripoSG (vendored code, weights in data/models/triposg/).

A 1.5B rectified-flow DiT turns DINOv2-L image features into 2048 shape latents; the VAE decodes them into an SDF
that is meshed hierarchically (dense grid, then a narrow band at twice the resolution) with marching cubes.
"""

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pymeshlab
import torch
import trimesh
from PIL import Image
from transformers import Dinov2Model

from . import Cancelled
from ..config import TRIPOSG_DIR
from ..vendor.triposg.inference_utils import hierarchical_extract_geometry
from ..vendor.triposg.models.autoencoders import TripoSGVAEModel
from ..vendor.triposg.models.transformers import TripoSGDiTModel
from ..vendor.triposg.schedulers import RectifiedFlowScheduler

log = logging.getLogger(__name__)

_FILES = (
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "image_encoder_dinov2/config.json",
    "image_encoder_dinov2/model.safetensors",
)
# Upstream's BitImageProcessor config: shortest edge to 256 (bicubic), centre crop 224, ImageNet normalisation.
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_BOUNDS = (-1.005, -1.005, -1.005, 1.005, 1.005, 1.005)  # pipeline_triposg.py default
# Upstream decodes a dense 256^3 grid before refining a narrow band. The band is so wide that the dense pass only
# settles empty space: starting at 128^3 (or even 64^3) gives the identical mesh, and 512^3 meshing takes 11 s, not 20.
_DENSE_DEPTH = 7


@dataclass
class TripoSGParams:
    image: Image.Image  # RGBA with the background already removed, or RGB to use as-is
    steps: int = 50
    guidance: float = 7.0
    seed: int = 0
    octree_depth: int = 9  # final grid is 2**depth per side
    faces: int = 0  # target face count for quadric decimation; 0 keeps the full marching-cubes mesh


def available() -> bool:
    return all((TRIPOSG_DIR / f).is_file() for f in _FILES)


def prepare_input(image: Image.Image, padding_ratio: float = 0.1) -> Image.Image:
    """Crop to the subject, flatten onto white and pad to a square with a 10% margin, as upstream's image_process."""
    rgba = np.asarray(image.convert("RGBA")).astype(np.float32) / 255
    alpha = rgba[..., 3:]
    rgb = rgba[..., :3] * alpha + (1 - alpha)
    ys, xs = np.where(alpha[..., 0] >= 0.5)  # BiRefNet's mask is soft; upstream thresholds RMBG's with Otsu
    if ys.size == 0:
        raise ValueError("Input image has no visible subject")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = rgb[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    pad = int(max(h, w) * padding_ratio)
    side = max(h, w) + 2 * pad
    canvas = np.ones((side, side, 3), dtype=np.float32)
    top, left = (side - h) // 2, (side - w) // 2
    canvas[top : top + h, left : left + w] = crop
    return Image.fromarray((canvas * 255).round().astype(np.uint8))


def _simplify(mesh: trimesh.Trimesh, faces: int) -> trimesh.Trimesh:
    """Quadric edge-collapse decimation, as upstream's --faces option but keeping the mesh watertight: upstream's
    merge_close_vertices welds across thin gaps and, without preservetopology, 857k -> 200k faces splits the
    marching-cubes mesh into extra open pieces."""
    if faces <= 0 or len(mesh.faces) <= faces:
        return mesh
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices.astype(np.float64), face_matrix=mesh.faces))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=faces, preservetopology=True)
    out = ms.current_mesh()
    return trimesh.Trimesh(out.vertex_matrix(), out.face_matrix())


class TripoSGEngine:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.dtype = torch.float16  # what upstream runs in
        self.transformer: TripoSGDiTModel | None = None
        self.vae: TripoSGVAEModel | None = None
        self.image_encoder: Dinov2Model | None = None

    def load(self) -> None:
        if self.transformer is not None:
            return
        log.info("Loading TripoSG from %s", TRIPOSG_DIR)
        kw = {"torch_dtype": self.dtype}
        # .to(dtype) as well, like upstream's pipe.to(device, dtype): from_pretrained leaves non-persistent buffers
        # (the VAE's positional-embedding frequencies) in fp32.
        load = lambda cls, sub: cls.from_pretrained(TRIPOSG_DIR / sub, **kw).to(self.device, self.dtype).eval()  # noqa: E731
        self.transformer = load(TripoSGDiTModel, "transformer")
        self.vae = load(TripoSGVAEModel, "vae")
        self.image_encoder = load(Dinov2Model, "image_encoder_dinov2")

    def unload(self) -> None:
        if self.transformer is None:
            return
        self.transformer = self.vae = self.image_encoder = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        w, h = image.size
        scale = 256 / min(w, h)
        image = image.resize((round(w * scale), round(h * scale)), Image.Resampling.BICUBIC)
        w, h = image.size
        left, top = (w - 224) // 2, (h - 224) // 2
        image = image.crop((left, top, left + 224, top + 224))
        x = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).unsqueeze(0).float() / 255
        x = ((x - _MEAN) / _STD).to(self.device, self.dtype)
        return self.image_encoder(x).last_hidden_state

    @torch.inference_mode()
    def generate(
        self,
        p: TripoSGParams,
        on_step: Callable[[int, int, str | None], None],
        is_cancelled: Callable[[], bool],
    ) -> tuple[trimesh.Trimesh, Image.Image, trimesh.Trimesh | None]:
        """Return the mesh (Y-up, facing +Z), the processed input image, and the undecimated mesh if the face
        limit reduced it (else None)."""
        self.load()
        cond = prepare_input(p.image)
        total = p.steps + 2  # + meshing + decimation/cleanup

        embeds = self._encode_image(cond)
        cfg = p.guidance > 1
        if cfg:  # upstream's unconditional branch is all-zero image features
            embeds = torch.cat([torch.zeros_like(embeds), embeds])

        # pipeline_triposg.TripoSGPipeline.__call__, stepped here for progress and cancel.
        scheduler = RectifiedFlowScheduler(num_train_timesteps=1000, shift=1.0)  # scheduler/scheduler_config.json
        scheduler.set_timesteps(p.steps, device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(p.seed)
        latents = torch.randn(
            (1, 2048, self.transformer.config.in_channels), generator=generator, device=self.device, dtype=self.dtype
        )
        for i, t in enumerate(scheduler.timesteps):
            model_in = torch.cat([latents] * 2) if cfg else latents
            pred = self.transformer(model_in, t.expand(model_in.shape[0]), encoder_hidden_states=embeds, return_dict=False)[0]
            if cfg:
                uncond, cond_pred = pred.chunk(2)
                pred = uncond + p.guidance * (cond_pred - uncond)
            latents = scheduler.step(pred, t, latents, return_dict=False)[0]
            if is_cancelled():
                raise Cancelled()
            on_step(i + 1, total, None)

        vertices, faces = hierarchical_extract_geometry(
            lambda x: self.vae.decode(latents, sampled_points=x).sample,
            self.device,
            bounds=_BOUNDS,
            dense_octree_depth=min(_DENSE_DEPTH, p.octree_depth - 1),
            hierarchical_octree_depth=p.octree_depth,
        )[0]
        if vertices is None:
            raise RuntimeError("TripoSG produced no surface")
        on_step(p.steps + 1, total, None)
        torch.cuda.empty_cache()

        full = trimesh.Trimesh(vertices, faces)
        if full.volume < 0:  # keep normals pointing outwards whatever the marching-cubes winding
            full.invert()
        mesh = _simplify(full, p.faces)
        on_step(total, total, None)
        return mesh, cond, (full if mesh is not full else None)
