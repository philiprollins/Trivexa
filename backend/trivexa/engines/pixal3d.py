"""Single image -> PBR-textured mesh with Pixal3D (vendored code; weights read from the shared models dir).

Pixal3D is TRELLIS.2 with pixel-aligned conditioning: DINOv3 features (plus NAF-upsampled ones) are back-projected
onto the voxel grid through the camera that took the photo, whose field of view MoGe-2 estimates. A cascade of
flow models then samples a 16^3 sparse structure, 512 and 1024/1536 shape latents and a texture latent; the decoded
mesh is remeshed, UV-unwrapped and its PBR attributes baked into textures by o-voxel.

Every stage runs one 1.3B flow model, so the models wait in system RAM (~14 GB) and only the active one is moved to
the GPU, as upstream's low-VRAM mode does.
"""

import gc
import json
import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import trimesh
from PIL import Image
from safetensors.torch import load_file

from . import Cancelled
from ..config import DINOV3_NAF_PATH, MOGE_PATH, PIXAL3D_DIR

log = logging.getLogger(__name__)

_MODELS = (
    "sparse_structure_flow_model",
    "sparse_structure_decoder",
    "shape_slat_flow_model_512",
    "shape_slat_flow_model_1024",
    "shape_slat_decoder",
    "tex_slat_flow_model_1024",
    "tex_slat_decoder",
)
# inference.py's IMAGE_COND_CONFIGS: (image size, projection grid resolution, NAF output size or None)
_COND = {
    "ss": (512, 16, None),
    "shape_512": (512, 32, 512),
    "shape_1024": (1024, 64, 512),
    "tex_1024": (1024, 64, 1024),
}
# facebook/dinov3-vitl16-pretrain-lvd1689m config.json
_DINOV3_CONFIG = dict(
    hidden_size=1024, intermediate_size=4096, num_hidden_layers=24, num_attention_heads=16, patch_size=16,
    num_register_tokens=4, layer_norm_eps=1e-5, layerscale_value=1.0, rope_theta=100.0, pos_embed_rescale=2.0,
    query_bias=True, key_bias=False, value_bias=True, proj_bias=True, mlp_bias=True, use_gated_mlp=False,
    hidden_act="gelu", image_size=224,
)
# MoGe configs/train/v2.json "model" (the published checkpoints carry this in model.pt; the safetensors doesn't)
_HEAD = dict(
    dim_in=[1024, 256, 128, 64, 32], dim_res_blocks=[1024, 256, 128, 64, 32], num_res_blocks=[0, 1, 1, 1, 0],
    res_block_in_norm="none", res_block_hidden_norm="none",
    resamplers=["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"],
)
_MOGE_CONFIG = dict(
    encoder=dict(backbone="dinov2_vitl14", intermediate_layers=[5, 11, 17, 23], dim_out=1024),
    neck=dict(_HEAD, dim_in=[1026, 2, 2, 2, 2], dim_out=None, num_res_blocks=[0, 2, 2, 2, 0]),
    points_head=dict(_HEAD, dim_out=[None, None, None, None, 3]),
    normal_head=dict(_HEAD, dim_out=[None, None, None, None, 3]),
    mask_head=dict(_HEAD, dim_out=[None, None, None, None, 1]),
    scale_head=dict(dims=[1024, 1024, 1024, 1]),
    remap_output="exp",
    num_tokens_range=[1200, 3600],
)
_PBR_LAYOUT = {"base_color": slice(0, 3), "metallic": slice(3, 4), "roughness": slice(4, 5), "alpha": slice(5, 6)}
# inference.py's final rotation ([[-1,0,0],[0,0,-1],[0,-1,0]]) leaves the subject Y-up but facing -Z; this is that
# rotation followed by a half turn about Y, so it faces +Z like Trivexa's other meshes.
_TO_GLTF = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
MAX_TOKENS = 49152  # upstream's cap on high-res shape tokens; past it the resolution drops in steps of 128


@dataclass
class Pixal3DParams:
    image: Image.Image  # RGBA with the background already removed, or RGB to use as-is
    seed: int = 42
    resolution: int = 1536  # 1024 or 1536 (upstream's default; "1536_cascade")
    steps: int = 12  # per sampling stage (structure, shape 512, shape HR, texture)
    guidance: float = 7.5  # structure and shape stages; texture stays at upstream's 1.0
    fov: float | None = None  # horizontal camera FOV in degrees; None estimates it with MoGe-2
    faces: int = 500_000  # decimation target after remeshing
    texture_size: int = 2048


def unavailable_reason() -> str | None:
    """Why Pixal3D can't run (missing weights or compiled extensions), or None."""
    pipeline = PIXAL3D_DIR / "pipeline.json"
    if not pipeline.is_file():
        return f"weights not found: {pipeline}"
    spec = json.loads(pipeline.read_text())["args"]["models"]
    files = [PIXAL3D_DIR / f"{spec[name]}{ext}" for name in _MODELS for ext in (".json", ".safetensors")]
    missing = [p for p in (*files, DINOV3_NAF_PATH, MOGE_PATH) if not p.is_file()]
    if missing:
        return "weights not found: " + ", ".join(str(p) for p in missing[:3])
    try:
        import cumesh  # noqa: F401
        import flex_gemm  # noqa: F401
        import nvdiffrast.torch  # noqa: F401
        import o_voxel  # noqa: F401
    except ImportError as e:
        return f"the pixal3d extra is not installed ({e.name}); see run.sh"
    return None


def available() -> bool:
    return unavailable_reason() is None


def prepare_input(image: Image.Image) -> Image.Image:
    """Upstream's preprocess_image for an image that already has its alpha: at most 1024 px, cropped to a square
    1.1x the subject's bounding box (alpha > 0.8) and flattened onto black."""
    image = image.convert("RGBA")
    scale = min(1, 1024 / max(image.size))
    if scale < 1:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    alpha = np.array(image)[:, :, 3]
    ys, xs = np.nonzero(alpha > 0.8 * 255)
    if ys.size == 0:
        raise ValueError("Input image has no visible subject")
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    size = int(max(x1 - x0, y1 - y0) * 1.1)
    image = image.crop((cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2))
    rgba = np.array(image).astype(np.float32) / 255
    rgb = rgba[:, :, :3] * rgba[:, :, 3:4]  # black background
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))


def camera_from_fov(camera_angle_x: float, mesh_scale: float = 1.0, image_resolution: int = 512) -> dict:
    """inference.py's distance_from_fov: the camera distance at which the unit cube's left edge (grid point
    (-1, 0, 0)) lands on the image's left border, for a camera with this horizontal FOV."""
    f_pixels = (16.0 / math.tan(camera_angle_x / 2.0)) * image_resolution / 32.0
    xw, yw = -1.0 / mesh_scale / 2, 0.0  # (-1, 0, 0) turned into Blender's frame (x, -z, y), then scaled
    x_ndc = 0 - image_resolution / 2.0
    distance = f_pixels * xw / x_ndc - yw
    return {"camera_angle_x": camera_angle_x, "distance": distance, "mesh_scale": mesh_scale}


def _flow_sampler_class():
    from ..vendor.pixal3d.pipelines.samplers import FlowEulerGuidanceIntervalSampler

    class Sampler(FlowEulerGuidanceIntervalSampler):
        """Upstream's sampler with a callback after every Euler step (progress and cancel)."""

        on_step: Callable[[], None] = staticmethod(lambda: None)

        def sample_once(self, *args, **kwargs):
            out = super().sample_once(*args, **kwargs)
            self.on_step()
            return out

    return Sampler


class Pixal3DEngine:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.models: dict | None = None  # name -> module, kept on the CPU between stages
        self.spec: dict | None = None  # pipeline.json "args"
        self.dino = self.naf = self.moge = None
        self.cond: dict | None = None  # stage -> DinoV3ProjFeatureExtractor (sharing dino and naf)

    def load(self) -> None:
        if self.models is not None:
            return
        from transformers import DINOv3ViTConfig, DINOv3ViTModel

        from ..vendor.moge.model.v2 import MoGeModel
        from ..vendor.naf.naf import NAF
        from ..vendor.pixal3d import models as p3d_models
        from ..vendor.pixal3d.image_cond import DinoV3ProjFeatureExtractor

        log.info("Loading Pixal3D from %s", PIXAL3D_DIR)
        with open(PIXAL3D_DIR / "pipeline.json") as f:
            self.spec = json.load(f)["args"]
        models = {}
        for name in _MODELS:
            models[name] = p3d_models.from_pretrained(str(PIXAL3D_DIR / self.spec["models"][name])).eval()
            gc.collect()

        weights = load_file(DINOV3_NAF_PATH)
        dino = DINOv3ViTModel(DINOv3ViTConfig(**_DINOV3_CONFIG)).eval()
        # transformers 5 nests the blocks under .model; the file has transformers 4 names
        layer_prefix = "model.layer." if hasattr(dino, "model") else "layer."
        dino.load_state_dict(
            {(layer_prefix + k[6:] if k.startswith("layer.") else k): v for k, v in weights.items() if not k.startswith("naf.")},
            strict=True,
        )
        naf = NAF().eval()
        naf.load_state_dict({k[4:]: v for k, v in weights.items() if k.startswith("naf.")}, strict=True)
        dino.requires_grad_(False)
        naf.requires_grad_(False)

        moge = MoGeModel(**_MOGE_CONFIG).eval()
        moge.load_state_dict({k: v.float() for k, v in load_file(MOGE_PATH).items()}, strict=True)

        self.cond = {
            stage: DinoV3ProjFeatureExtractor(dino, naf, image_size=size, grid_resolution=grid,
                                              use_naf_upsample=naf_size is not None, naf_target_size=naf_size)
            for stage, (size, grid, naf_size) in _COND.items()
        }
        self.models, self.dino, self.naf, self.moge = models, dino, naf, moge

    def unload(self) -> None:
        if self.models is None:
            return
        self.models = self.spec = self.dino = self.naf = self.moge = self.cond = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @contextmanager
    def _on_gpu(self, *modules):
        for m in modules:
            m.to(self.device)
        try:
            yield
        finally:
            for m in modules:
                m.cpu()
            torch.cuda.empty_cache()

    def estimate_fov(self, image: Image.Image) -> float:
        """Horizontal FOV (radians) MoGe-2 estimates for the prepared image, as inference.py's
        get_camera_params_wild_moge."""
        x = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)
        with self._on_gpu(self.moge):
            out = self.moge.infer(x.to(self.device))
        fx = out["intrinsics"].squeeze()[0, 0].item() * image.width
        return 2 * math.atan(image.width / (2 * fx))

    def _proj_cond(self, stage: str, image: Image.Image, camera: dict, coords: torch.Tensor | None = None,
                   grid_resolution: int | None = None) -> dict:
        """get_proj_cond_ss / get_proj_cond_shape: global + projected features and their all-zero negatives."""
        from ..vendor.pixal3d.image_cond import ProjGrid
        from ..vendor.pixal3d.modules.sparse import SparseTensor

        extractor = self.cond[stage]
        default_grid = extractor.proj_grid
        if grid_resolution is not None and grid_resolution != extractor.grid_resolution:
            extractor.proj_grid = ProjGrid(grid_resolution=grid_resolution, image_resolution=default_grid.image_resolution)
        grid = extractor.proj_grid.grid_resolution
        args = {k: torch.tensor([camera[k]], device=self.device) for k in ("camera_angle_x", "distance", "mesh_scale")}
        # Upstream projects the whole grid and then picks the occupied voxels; projecting only those is the same.
        index = None if coords is None else (coords[:, 1].long() * grid + coords[:, 2].long()) * grid + coords[:, 3].long()
        try:
            z_global, z_proj = extractor([image], **args, point_index=index)
        finally:
            extractor.proj_grid = default_grid
        if coords is None:
            proj, neg_proj = z_proj, torch.zeros_like(z_proj)
        else:  # batch of one: every coordinate's batch index is 0
            feats = z_proj[0]
            proj = SparseTensor(feats=feats, coords=coords)
            neg_proj = SparseTensor(feats=torch.zeros_like(feats), coords=coords)
        return {
            "cond": {"global": z_global, "proj": proj},
            "neg_cond": {"global": torch.zeros_like(z_global), "proj": neg_proj},
        }

    def _sample(self, which: str, model, noise, cond: dict, p: Pixal3DParams, sampler, **extra):
        spec = self.spec[f"{which}_sampler"]
        params = dict(spec["params"], steps=p.steps)
        if which != "tex_slat":
            params["guidance_strength"] = p.guidance
        with self._on_gpu(model):
            return sampler.sample(model, noise, **cond, **params, **extra, verbose=False).samples

    @torch.no_grad()
    def generate(
        self,
        p: Pixal3DParams,
        on_step: Callable[[int, int, str | None], None],
        is_cancelled: Callable[[], bool],
    ) -> tuple[trimesh.Trimesh, Image.Image, dict]:
        """Return the textured mesh (Y-up, facing +Z), the prepared input image, and facts worth recording
        (the FOV used and the resolution the cascade ran at)."""
        import o_voxel

        from ..vendor.pixal3d.modules.sparse import SparseTensor
        from ..vendor.pixal3d.representations import MeshWithVoxel

        self.load()
        image = prepare_input(p.image)
        total = 4 * p.steps + 4  # four sampling stages + camera, decode, GLB bake, export
        done = 0

        def step():
            nonlocal done
            if is_cancelled():
                raise Cancelled()
            done += 1
            on_step(done, total, None)

        Sampler = _flow_sampler_class()
        sampler = Sampler(**self.spec["sparse_structure_sampler"]["args"])
        sampler.on_step = step

        fov = math.radians(p.fov) if p.fov else self.estimate_fov(image)
        camera = camera_from_fov(fov)
        step()

        norm = {k: {s: torch.tensor(v)[None].to(self.device) for s, v in self.spec[f"{k}_slat_normalization"].items()}
                for k in ("shape", "tex")}
        m = self.models
        torch.manual_seed(p.seed)  # upstream draws every noise tensor from the global CPU generator
        self.dino.to(self.device)
        self.naf.to(self.device)
        try:
            # Stage 1: sparse structure (16^3 latent decoded to 64^3 occupancy, pooled to 32^3)
            cond = self._proj_cond("ss", image, camera)
            flow = m["sparse_structure_flow_model"]
            noise = torch.randn(1, flow.in_channels, flow.resolution, flow.resolution, flow.resolution).to(self.device)
            z_s = self._sample("sparse_structure", flow, noise, cond, p, sampler)
            del cond
            with self._on_gpu(m["sparse_structure_decoder"]):
                decoded = m["sparse_structure_decoder"](z_s) > 0
            if decoded.shape[2] != 32:
                ratio = decoded.shape[2] // 32
                decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
            coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
            if coords.shape[0] == 0:
                raise RuntimeError("Pixal3D produced an empty structure")

            # Stage 2: shape latent at 512
            cond = self._proj_cond("shape_512", image, camera, coords)
            flow = m["shape_slat_flow_model_512"]
            noise = SparseTensor(feats=torch.randn(coords.shape[0], flow.in_channels).to(self.device), coords=coords)
            lr_slat = self._sample("shape_slat", flow, noise, cond, p, sampler)
            lr_slat = lr_slat * norm["shape"]["std"] + norm["shape"]["mean"]
            del cond

            # Stage 3: upsample the structure and sample the shape latent at the high resolution
            decoder = m["shape_slat_decoder"]
            with self._on_gpu(decoder):
                decoder.low_vram = True
                try:
                    hr_coords = decoder.upsample(lr_slat, upsample_times=4)
                finally:
                    decoder.low_vram = False
            resolution = p.resolution
            while True:
                grid = resolution // 16
                quant = torch.cat([hr_coords[:, :1], ((hr_coords[:, 1:] + 0.5) / 512 * (grid - 1)).round().int()], dim=1)
                coords = quant.unique(dim=0)
                if coords.shape[0] < MAX_TOKENS or resolution == 1024:
                    break
                resolution -= 128
            del lr_slat, hr_coords, quant
            cond = self._proj_cond("shape_1024", image, camera, coords, grid_resolution=resolution // 16)
            flow = m["shape_slat_flow_model_1024"]
            noise = SparseTensor(feats=torch.randn(coords.shape[0], flow.in_channels).to(self.device), coords=coords)
            shape_slat = self._sample("shape_slat", flow, noise, cond, p, sampler)
            shape_slat = shape_slat * norm["shape"]["std"] + norm["shape"]["mean"]
            del cond, noise

            # Stage 4: texture latent, conditioned on the (normalised) shape latent
            cond = self._proj_cond("tex_1024", image, camera, shape_slat.coords, grid_resolution=resolution // 16)
            flow = m["tex_slat_flow_model_1024"]
            normed_shape = (shape_slat - norm["shape"]["mean"]) / norm["shape"]["std"]
            noise = normed_shape.replace(feats=torch.randn(normed_shape.coords.shape[0], flow.in_channels - normed_shape.feats.shape[1]).to(self.device))
            tex_slat = self._sample("tex_slat", flow, noise, cond, p, sampler, concat_cond=normed_shape)
            tex_slat = tex_slat * norm["tex"]["std"] + norm["tex"]["mean"]
            del cond, noise, normed_shape
        finally:
            self.dino.cpu()
            self.naf.cpu()
            torch.cuda.empty_cache()

        # Stage 5: decode the mesh and its PBR voxels
        with self._on_gpu(decoder):
            decoder.set_resolution(resolution)
            decoder.low_vram = True
            try:
                meshes, subs = decoder(shape_slat, return_subs=True)
            finally:
                decoder.low_vram = False
        with self._on_gpu(m["tex_slat_decoder"]):
            tex_voxels = m["tex_slat_decoder"](tex_slat, guide_subs=subs) * 0.5 + 0.5
        del subs, shape_slat, tex_slat
        mesh, voxels = meshes[0], tex_voxels[0]
        mesh.fill_holes()
        out = MeshWithVoxel(
            mesh.vertices, mesh.faces, origin=[-0.5, -0.5, -0.5], voxel_size=1 / resolution,
            coords=voxels.coords[:, 1:], attrs=voxels.feats,
            voxel_shape=torch.Size([*voxels.shape, *voxels.spatial_shape]), layout=_PBR_LAYOUT,
        )
        del meshes, tex_voxels, mesh, voxels
        torch.cuda.empty_cache()
        step()

        glb = o_voxel.postprocess.to_glb(
            vertices=out.vertices, faces=out.faces, attr_volume=out.attrs, coords=out.coords,
            attr_layout=_PBR_LAYOUT, grid_size=resolution, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=p.faces, texture_size=p.texture_size,
            remesh=True, remesh_band=1, remesh_project=0,
        )
        del out
        torch.cuda.empty_cache()
        step()
        glb.apply_transform(_TO_GLTF)
        step()
        return glb, image, {"fov": round(math.degrees(fov), 2), "resolution": resolution}
