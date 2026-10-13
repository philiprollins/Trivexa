"""Multi-view generation with MV-Adapter on top of a local SDXL checkpoint.

Two modes, each with its own adapter weights:
  t2mv  text -> 6 orthographic views
  i2mv  reference image (+ optional text) -> 6 views consistent with the reference
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image

from ..config import MV_ADAPTER_DIR
from ..vendor.mvadapter.models.attention_processor import DecoupledMVRowColSelfAttnProcessor2_0
from ..vendor.mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
from ..vendor.mvadapter.pipelines.pipeline_mvadapter_t2mv_sdxl import MVAdapterT2MVSDXLPipeline
from ..vendor.mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from ..vendor.mvadapter.utils.geometry import get_plucker_embeds_from_cameras_ortho
from ..vendor.mvadapter.utils.mesh_utils.camera import get_orthogonal_camera
from .diffusion import ARCHS, SAMPLERS, config_dir, decode_latents, make_step_callback

log = logging.getLogger(__name__)

# mode -> (pipeline class, adapter weights, self-attention processor or None for the default row attention)
MODES = {
    "t2mv": (MVAdapterT2MVSDXLPipeline, "mvadapter_t2mv_sdxl.safetensors", None),
    "i2mv": (MVAdapterI2MVSDXLPipeline, "mvadapter_i2mv_sdxl.safetensors", None),
    # Geometry-guided: conditioned on position/normal maps rendered from a mesh (incl. top/bottom views),
    # so it needs row+column attention. Used for texturing; the caller supplies the condition maps.
    "ig2mv": (MVAdapterI2MVSDXLPipeline, "mvadapter_ig2mv_sdxl.safetensors", DecoupledMVRowColSelfAttnProcessor2_0),
}
# Front, front-right, right, back, left, front-left (degrees around the object).
AZIMUTHS = [0, 45, 90, 180, 270, 315]
VIEW_SIZE = 768


def adapter_available(mode: str) -> bool:
    return (MV_ADAPTER_DIR / MODES[mode][1]).is_file()


@dataclass
class MVParams:
    mode: str
    model_path: Path
    prompt: str
    negative: str
    steps: int
    cfg: float
    sampler: str
    seed: int
    reference: Image.Image | None = None  # i2mv/ig2mv, already preprocessed to VIEW_SIZE
    reference_scale: float = 1.0
    control_images: torch.Tensor | None = None  # ig2mv: (6, 6, H, W) position+normal maps in [0, 1]


def prepare_reference(rgba: Image.Image, size: int = VIEW_SIZE) -> Image.Image:
    """Crop to the subject, fit it to 90% of a square canvas and flatten onto grey (what i2mv was trained on)."""
    arr = np.asarray(rgba.convert("RGBA"))
    alpha = arr[..., 3] > 0
    if not alpha.any():
        raise ValueError("Reference image has no visible subject")
    ys, xs = np.where(alpha)
    h_img, w_img = alpha.shape
    y0, y1 = max(ys.min() - 1, 0), min(ys.max() + 1, h_img)
    x0, x1 = max(xs.min() - 1, 0), min(xs.max() + 1, w_img)
    crop = arr[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    scale = size * 0.9 / max(h, w)
    h, w = max(1, int(h * scale)), max(1, int(w * scale))
    crop = np.asarray(Image.fromarray(crop).resize((w, h), Image.Resampling.LANCZOS))
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    top, left = (size - h) // 2, (size - w) // 2
    canvas[top : top + h, left : left + w] = crop
    rgba_f = canvas.astype(np.float32) / 255
    rgb = rgba_f[..., :3] * rgba_f[..., 3:] + (1 - rgba_f[..., 3:]) * 0.5
    return Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8))


class MultiviewEngine:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.pipe = None
        self.mode: str | None = None
        self.model_path: Path | None = None
        self._base_scheduler_config = None
        self.vae_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        self._control_images: torch.Tensor | None = None

    def load(self, mode: str, model_path: Path) -> None:
        if self.pipe is not None and self.mode == mode and self.model_path == model_path:
            return
        self.unload()
        pipeline_cls, weight_name, attn_processor = MODES[mode]
        log.info("Loading MV-Adapter %s on %s", mode, model_path.name)
        pipe = pipeline_cls.from_single_file(
            str(model_path),
            config=str(config_dir(ARCHS["sdxl"])),
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        # Swap the UNet's attention processors for multi-view ones, then load the adapter weights into them.
        if attn_processor is None:
            pipe.init_custom_adapter(num_views=len(AZIMUTHS))
        else:
            pipe.init_custom_adapter(num_views=len(AZIMUTHS), self_attn_processor=attn_processor)
        pipe.load_custom_adapter(str(MV_ADAPTER_DIR), weight_name=weight_name, local_files_only=True)
        pipe.to(device=self.device, dtype=torch.float16)
        pipe.cond_encoder.to(device=self.device, dtype=torch.float16)
        pipe.vae.to(dtype=self.vae_dtype)
        pipe.set_progress_bar_config(disable=True)
        self.pipe, self.mode, self.model_path = pipe, mode, model_path
        self._base_scheduler_config = pipe.scheduler.config
        self._control_images = self._camera_conditioning()

    def unload(self) -> None:
        if self.pipe is None:
            return
        self.pipe = self.mode = self.model_path = None
        self._control_images = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _camera_conditioning(self) -> torch.Tensor:
        """Plücker ray embeddings for the fixed orthographic camera ring (from the upstream inference scripts)."""
        n = len(AZIMUTHS)
        cameras = get_orthogonal_camera(
            elevation_deg=[0] * n,
            distance=[1.8] * n,
            left=-0.55,
            right=0.55,
            bottom=-0.55,
            top=0.55,
            azimuth_deg=[a - 90 for a in AZIMUTHS],
            device=self.device,
        )
        plucker = get_plucker_embeds_from_cameras_ortho(cameras.c2w, [1.1] * n, VIEW_SIZE)
        return ((plucker + 1.0) / 2.0).clamp(0, 1)

    def generate(
        self,
        p: MVParams,
        on_step: Callable[[int, int, str | None], None],
        is_cancelled: Callable[[], bool],
    ) -> list[Image.Image]:
        self.load(p.mode, p.model_path)
        pipe = self.pipe

        _, scheduler_cls, overrides = SAMPLERS[p.sampler]
        # MV-Adapter is trained with a shifted noise schedule (higher noise for the larger multi-view batch).
        sched = ShiftSNRScheduler.from_scheduler(
            scheduler_cls.from_config(self._base_scheduler_config, **overrides),
            shift_mode="interpolated",
            shift_scale=8.0,
            scheduler_class=scheduler_cls,
        )
        pipe.scheduler = sched

        callback = make_step_callback(sched, ARCHS["sdxl"], p.steps, on_step, is_cancelled, grid_cols=3)
        kwargs = {}
        if p.mode in ("i2mv", "ig2mv"):
            kwargs = {"reference_image": p.reference, "reference_conditioning_scale": p.reference_scale}
        control = self._control_images if p.control_images is None else p.control_images.to(self.device)

        # Encode the prompt, then park the text encoders (~1.6 GB) in system RAM while the UNet runs:
        # i2mv's cached reference attention doesn't fit in 16 GB alongside them.
        encoders = (pipe.text_encoder, pipe.text_encoder_2)
        for enc in encoders:
            enc.to(self.device)
        with torch.inference_mode():
            pe, ne, ppe, npe = pipe.encode_prompt(
                p.prompt, device=self.device, do_classifier_free_guidance=True, negative_prompt=p.negative
            )
        for enc in encoders:
            enc.to("cpu")
        torch.cuda.empty_cache()

        result = pipe(
            prompt_embeds=pe,
            negative_prompt_embeds=ne,
            pooled_prompt_embeds=ppe,
            negative_pooled_prompt_embeds=npe,
            height=VIEW_SIZE,
            width=VIEW_SIZE,
            num_inference_steps=p.steps,
            guidance_scale=p.cfg,
            num_images_per_prompt=len(AZIMUTHS),
            control_image=control,
            control_conditioning_scale=1.0,
            generator=torch.Generator("cpu").manual_seed(p.seed),
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
            output_type="latent",
            **kwargs,
        )
        # i2mv temporarily moves the VAE to fp32/fp16 to encode the reference; decode_latents restores bf16.
        return decode_latents(pipe, result.images, self.vae_dtype)
