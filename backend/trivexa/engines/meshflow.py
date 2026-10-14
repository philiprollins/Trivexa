"""Dense mesh -> artist-style low-poly mesh with MeshFlow (vendored code, local weights).

MeshFlow samples the input's surface, voxelizes it as RoPE positions for a flow-matching DiT, and decodes the
latents with its MeshVAE into vertices plus an adjacency graph whose triangles form the new mesh (~0.5-2k faces).
An optional reference image is encoded with DINOv3 ViT-L/16 (weights from Meta, loaded on first use) and steers the
DiT through cross-attention with classifier-free guidance; without one the zero visual condition is used.
"""

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from scipy.spatial import cKDTree
from torchvision import transforms

from . import Cancelled
from ..config import MESHFLOW_DIR
from ..vendor.meshflow.models.condition_encoder import make_empty_visual_embeds, resolve_visual_embed_dim
from ..vendor.meshflow.models.meshflow_dit import build_meshflow_dit
from ..vendor.meshflow.models.meshflow_vae import build_mesh_model
from ..vendor.meshflow.pipelines.meshflow_pipeline import MeshFlowPipeline
from ..vendor.meshflow.pipelines.utils import flow_sample
from ..vendor.dinov3.models.vision_transformer import DinoVisionTransformer

log = logging.getLogger(__name__)

DEFAULT_STEPS = 28  # the bundled config's num_inference_steps
DEFAULT_GUIDANCE = 2.5  # the bundled config's guidance_scale (CFG on the image condition)
# The backbone MeshFlow's config names (visual_condition.hub_model: dinov3_vitl16), from Meta's DINOv3 downloads.
DINOV3_WEIGHTS = MESHFLOW_DIR / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"


@dataclass
class MeshFlowParams:
    mesh: trimesh.Trimesh
    steps: int = DEFAULT_STEPS
    seed: int = 0
    fill_holes: bool = False
    image: Image.Image | None = None  # reference, already cut out and flattened (see prepare_reference)
    guidance: float = DEFAULT_GUIDANCE


def available() -> bool:
    return all((MESHFLOW_DIR / f).is_file() for f in ("config.yaml", "model.pth"))


def image_available() -> bool:
    return DINOV3_WEIGHTS.is_file()


class _DinoEncoder(torch.nn.Module):
    """MeshFlow's DINOv3Encoder.encode_image on a locally built backbone (upstream loads it through torch.hub)."""

    def __init__(self, image_size: int):
        super().__init__()
        # dinov3_vitl16 (LVD-1689M) as built by dinov3/hub/backbones.py.
        self.dino_model = DinoVisionTransformer(
            img_size=224,
            patch_size=16,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            embed_dim=1024,
            depth=24,
            num_heads=16,
            ffn_ratio=4,
            qkv_bias=True,
            layerscale_init=1.0e-05,
            norm_layer="layernormbf16",
            ffn_layer="mlp",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
        )
        self.dino_model.load_state_dict(torch.load(DINOV3_WEIGHTS, map_location="cpu", weights_only=True))
        self.dino_model.eval().requires_grad_(False)
        self.transform = transforms.Compose([
            transforms.Resize(image_size, transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        ref = next(self.dino_model.parameters())
        x = self.transform(transforms.ToTensor()(image.convert("RGB")).unsqueeze(0)).to(ref.device, ref.dtype)
        with torch.autocast(device_type=ref.device.type, enabled=False):
            out = self.dino_model.forward_features(x)["x_prenorm"]
        return F.layer_norm(out, out.shape[-1:])


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
        self._image_size = int(visual_cfg["image_size"])

    def _load_image_encoder(self) -> None:
        if self.pipe.models["visual_encoder"] is None:
            log.info("Loading DINOv3 ViT-L/16 from %s", DINOV3_WEIGHTS)
            self.pipe.models["visual_encoder"] = _DinoEncoder(self._image_size).to(self.device)

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

        guidance = p.guidance if p.image is not None else 1.0  # CFG only mixes in the image condition
        if p.image is not None:
            self._load_image_encoder()
        visual_cond = pipe.get_visual_cond(p.image, guidance_scale=guidance, preprocess_image=False)

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
                    visual_cond=visual_cond,
                    guidance_scale=guidance,
                    do_classifier_free_guidance=guidance != 1.0,
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
            out.visual = trimesh.visual.ColorVisuals(out, vertex_colors=_transfer_colors(src.vertices, colors, out))
        return out


def _transfer_colors(src_vertices: np.ndarray, src_colors: np.ndarray, out: trimesh.Trimesh) -> np.ndarray:
    """Each new vertex gets the mean colour of the source vertices within half its average edge length.

    A low-poly vertex stands for a patch of the dense surface; copying only the nearest source vertex lets a dark
    crevice (eye socket, joint gap) blacken every large triangle around it.
    """
    edge_len = out.edges_unique_length
    total = np.bincount(out.edges_unique.ravel(), np.repeat(edge_len, 2), minlength=len(out.vertices))
    count = np.bincount(out.edges_unique.ravel(), minlength=len(out.vertices))
    radius = 0.5 * total / np.maximum(count, 1)
    tree = cKDTree(src_vertices)
    _, nearest = tree.query(out.vertices)
    result = src_colors[nearest].astype(np.float64)
    for i, idx in enumerate(tree.query_ball_point(out.vertices, radius)):
        if idx:
            result[i] = src_colors[idx].mean(0)
    return result.round().astype(np.uint8)


def _vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray | None:
    """Per-vertex RGBA of a vertex-coloured or UV-textured mesh (textures are sampled at the vertices)."""
    visual = mesh.visual.to_color() if mesh.visual.kind == "texture" else mesh.visual
    return np.asarray(visual.vertex_colors) if visual.kind == "vertex" else None
