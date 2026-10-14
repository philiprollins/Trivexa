"""Dense mesh -> artist-style low-poly mesh with MeshFlow (vendored code, local weights).

MeshFlow samples the input's surface, voxelizes it as RoPE positions for a flow-matching DiT, and decodes the
latents with its MeshVAE into vertices plus an adjacency graph whose triangles form the new mesh (~0.5-2k faces).
Reference-image conditioning needs DINOv3 (gated weights) and isn't wired up: the zero visual condition is used.
"""

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import trimesh
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

from . import Cancelled
from ..config import MESHFLOW_DIR
from ..vendor.meshflow.models.condition_encoder import make_empty_visual_embeds, resolve_visual_embed_dim
from ..vendor.meshflow.models.meshflow_dit import build_meshflow_dit
from ..vendor.meshflow.models.meshflow_vae import build_mesh_model
from ..vendor.meshflow.pipelines.meshflow_pipeline import MeshFlowPipeline
from ..vendor.meshflow.pipelines.utils import flow_sample

log = logging.getLogger(__name__)

DEFAULT_STEPS = 28  # the bundled config's num_inference_steps


@dataclass
class MeshFlowParams:
    mesh: trimesh.Trimesh
    steps: int = DEFAULT_STEPS
    seed: int = 0
    fill_holes: bool = False


def available() -> bool:
    return all((MESHFLOW_DIR / f).is_file() for f in ("config.yaml", "model.pth"))


def _subtree(state: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


class MeshFlowEngine:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.pipe: MeshFlowPipeline | None = None

    def load(self) -> None:
        if self.pipe is not None:
            return
        log.info("Loading MeshFlow from %s", MESHFLOW_DIR)
        cfg = OmegaConf.load(MESHFLOW_DIR / "config.yaml")
        s = cfg.system
        # Upstream's from_pretrained re-reads the 4.5 GB checkpoint once per sub-model; read it once (mmap) instead.
        state = torch.load(MESHFLOW_DIR / "model.pth", map_location="cpu", weights_only=True, mmap=True)
        vae = build_mesh_model(OmegaConf.to_container(s.mesh_model, resolve=True))
        vae.load_state_dict(_subtree(state, "mesh_model."))
        dit = build_meshflow_dit(OmegaConf.to_container(s.denoiser_model, resolve=True))
        dit.load_state_dict(_subtree(state, "denoiser_model.dit_model."))
        del state

        visual_cfg = OmegaConf.to_container(s.visual_condition, resolve=True)
        embed_dim = resolve_visual_embed_dim(visual_cfg, fallback=int(s.denoiser_model.visual_condition_dim))
        pipe = MeshFlowPipeline(
            models={"mesh_model": vae.to(self.device).eval(), "visual_encoder": None, "denoiser": dit.to(self.device).eval()},
            scheduler=FlowMatchEulerDiscreteScheduler(**OmegaConf.to_container(s.denoise_scheduler, resolve=True)),
            num_verts=int(cfg.data.n_verts),
            voxelize_resolution=int(s.voxelize_resolution),
            num_inference_steps=int(s.num_inference_steps),
            guidance_scale=1.0,  # CFG only applies to the (unused) image condition
            dtype=torch.float16,
        )
        pipe._device = self.device
        pipe._empty_visual_embeds = make_empty_visual_embeds(int(visual_cfg["image_size"]), embed_dim)
        self.pipe = pipe

    def unload(self) -> None:
        if self.pipe is None:
            return
        self.pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate(
        self,
        p: MeshFlowParams,
        on_step: Callable[[int, int, str | None], None],
        is_cancelled: Callable[[], bool],
    ) -> trimesh.Trimesh:
        """Return the remeshed model in the input's frame, with colours carried over from the input."""
        self.load()
        pipe = self.pipe
        src = p.mesh
        # Upstream's "bsphere" normalization: centre on the bounding box, scale into the unit sphere.
        center = (src.vertices.min(0) + src.vertices.max(0)) / 2
        radius = float(np.linalg.norm(src.vertices - center, axis=1).max())
        unit = trimesh.Trimesh((src.vertices - center) / radius, src.faces, process=False)
        points, _ = trimesh.sample.sample_surface(unit, pipe.num_verts, seed=p.seed)
        surface = torch.from_numpy(points.astype(np.float32)).unsqueeze(0).to(self.device)

        denoiser = pipe.models["denoiser"]
        latents = None
        # Same as MeshFlowPipeline.sample_latent, but stepping the sampler ourselves for progress and cancel.
        with torch.autocast(device_type=self.device.type, dtype=pipe.dtype):
            for i, (latents, _) in enumerate(
                flow_sample(
                    pipe.scheduler,
                    denoiser,
                    (denoiser.input_size, denoiser.in_channels),
                    steps=p.steps,
                    visual_cond=pipe.get_visual_cond(None, guidance_scale=1.0),
                    guidance_scale=1.0,
                    do_classifier_free_guidance=False,
                    generator=torch.Generator(device=self.device).manual_seed(p.seed),
                    device=self.device,
                    joint_attention_kwargs=pipe.get_rope_cond(surface),
                )
            ):
                if is_cancelled():
                    raise Cancelled()
                on_step(i + 1, p.steps + 1, None)
        out = pipe.decode_latent(latents, fill_holes=p.fill_holes).to_trimesh()
        on_step(p.steps + 1, p.steps + 1, None)

        out.vertices = out.vertices * radius + center
        colors = _vertex_colors(src)
        if colors is not None:
            _, nearest = cKDTree(src.vertices).query(out.vertices)
            out.visual = trimesh.visual.ColorVisuals(out, vertex_colors=colors[nearest])
        return out


def _vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray | None:
    """Per-vertex RGBA of a vertex-coloured or UV-textured mesh (textures are sampled at the vertices)."""
    visual = mesh.visual.to_color() if mesh.visual.kind == "texture" else mesh.visual
    return np.asarray(visual.vertex_colors) if visual.kind == "vertex" else None
