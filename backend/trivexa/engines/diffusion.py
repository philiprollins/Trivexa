"""Text-to-image for Stable Diffusion family checkpoints (SDXL, SD 1.5), loaded from single files with diffusers.

Only one pipeline is kept on the GPU at a time; switching to a model of another architecture swaps it out.
"""

import base64
import io
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UniPCMultistepScheduler,
)
from huggingface_hub import snapshot_download
from PIL import Image

from ..config import HF_CONFIG_DIR
from . import Cancelled

log = logging.getLogger(__name__)

# id -> (label, scheduler class, scheduler overrides)
SAMPLERS: dict[str, tuple[str, type, dict]] = {
    "euler_a": ("Euler a", EulerAncestralDiscreteScheduler, {}),
    "euler": ("Euler", EulerDiscreteScheduler, {}),
    "dpmpp_2m": ("DPM++ 2M", DPMSolverMultistepScheduler, {}),
    "dpmpp_2m_karras": ("DPM++ 2M Karras", DPMSolverMultistepScheduler, {"use_karras_sigmas": True}),
    "dpmpp_2m_sde_karras": (
        "DPM++ 2M SDE Karras",
        DPMSolverMultistepScheduler,
        {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True},
    ),
    "unipc": ("UniPC", UniPCMultistepScheduler, {}),
    "ddim": ("DDIM", DDIMScheduler, {}),
}

CHUNK = 75  # CLIP context is 77 including BOS/EOS


@dataclass(frozen=True)
class Arch:
    id: str
    label: str
    pipeline_cls: type
    config_repo: str  # architecture configs + tokenizers only; weights come from the local checkpoint
    config_name: str  # folder under data/hf-configs
    # Cheap linear latent -> RGB projection for live previews (same factors ComfyUI uses).
    latent_rgb: tuple[tuple[float, float, float], ...]
    latent_rgb_bias: tuple[float, float, float]


ARCHS: dict[str, Arch] = {
    "sdxl": Arch(
        id="sdxl",
        label="SDXL",
        pipeline_cls=StableDiffusionXLPipeline,
        config_repo="stabilityai/stable-diffusion-xl-base-1.0",
        config_name="sdxl-base-1.0",
        latent_rgb=(
            (0.3651, 0.4232, 0.4341),
            (-0.2533, -0.0042, 0.1068),
            (0.1076, 0.1111, -0.0362),
            (-0.3165, -0.2492, -0.2188),
        ),
        latent_rgb_bias=(0.1084, -0.0175, -0.0011),
    ),
    "sd15": Arch(
        id="sd15",
        label="SD 1.5",
        pipeline_cls=StableDiffusionPipeline,
        config_repo="stable-diffusion-v1-5/stable-diffusion-v1-5",
        config_name="sd-v1-5",
        latent_rgb=(
            (0.3512, 0.2297, 0.3227),
            (0.3250, 0.4974, 0.2350),
            (-0.2829, 0.1762, 0.2721),
            (-0.2120, -0.2616, -0.7177),
        ),
        latent_rgb_bias=(0.0, 0.0, 0.0),
    ),
}


@dataclass
class GenParams:
    arch: str
    model_path: Path
    prompt: str
    negative: str
    width: int
    height: int
    steps: int
    cfg: float
    sampler: str
    seed: int
    # 1 = the architecture's standard text-encoder layer (last for SD 1.5, penultimate for SDXL);
    # each step above that goes one layer further back.
    clip_skip: int = 1


def config_dir(arch: Arch) -> Path:
    """Architecture configs + tokenizers (a few MB, no weights), fetched once and then used offline."""
    local = HF_CONFIG_DIR / arch.config_name
    if not (local / "model_index.json").exists():
        log.info("Fetching %s config files (one-time, no weights)…", arch.label)
        snapshot_download(
            arch.config_repo,
            local_dir=local,
            allow_patterns=["model_index.json", "*/config.json", "*/*_config.json", "tokenizer*/*"],
        )
    return local


def latent_preview(arch: Arch, latents: torch.Tensor, sigma: float | None) -> str:
    x = latents[0].float()
    if sigma:
        x = x / math.sqrt(sigma**2 + 1)
    factors = torch.tensor(arch.latent_rgb, device=x.device)
    bias = torch.tensor(arch.latent_rgb_bias, device=x.device)
    rgb = torch.einsum("chw,cr->rhw", x, factors) + bias[:, None, None]
    rgb = ((rgb + 1) / 2).clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class DiffusionEngine:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.pipe: StableDiffusionPipeline | StableDiffusionXLPipeline | None = None
        self.arch: Arch | None = None
        self.model_path: Path | None = None
        self._base_scheduler_config = None
        self.vae_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    def load(self, arch: Arch, model_path: Path) -> None:
        if self.pipe is not None and self.model_path == model_path:
            return
        self.unload()
        log.info("Loading %s checkpoint %s", arch.label, model_path.name)
        kwargs = {}
        if arch.id == "sd15":
            kwargs = {"safety_checker": None, "feature_extractor": None, "requires_safety_checker": False}
        pipe = arch.pipeline_cls.from_single_file(
            str(model_path),
            config=str(config_dir(arch)),
            torch_dtype=torch.float16,
            local_files_only=True,
            **kwargs,
        )
        pipe.to(self.device)
        # The SDXL VAE overflows in fp16; bf16 has fp32's range, so decode there instead of upcasting to fp32.
        pipe.vae.to(dtype=self.vae_dtype)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.arch = arch
        self.model_path = model_path
        self._base_scheduler_config = pipe.scheduler.config

    def unload(self) -> None:
        if self.pipe is None:
            return
        self.pipe = None
        self.arch = None
        self.model_path = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- prompt encoding -------------------------------------------------------
    # Prompts are encoded in 75-token chunks and concatenated, so long tag-style prompts aren't
    # truncated at CLIP's 77-token limit. Positive and negative are padded to the same chunk count.

    @staticmethod
    def _chunk_count(tokenizers, texts: list[str]) -> tuple[list[list[list[int]]], int]:
        ids = [[tok(t, truncation=False, add_special_tokens=False).input_ids for t in texts] for tok in tokenizers]
        return ids, max(1, *(math.ceil(len(i) / CHUNK) for per_tok in ids for i in per_tok))

    def _chunk_rows(self, tok, ids: list[int], n_chunks: int):
        for c in range(n_chunks):
            part = ids[c * CHUNK : (c + 1) * CHUNK]
            row = [tok.bos_token_id, *part, tok.eos_token_id]
            row += [tok.pad_token_id] * (CHUNK + 2 - len(row))
            yield c, torch.tensor([row], device=self.device)

    @torch.inference_mode()
    def _encode_sdxl(self, prompt: str, negative: str, clip_skip: int) -> dict:
        pipe = self.pipe
        encoders = [(pipe.tokenizer, pipe.text_encoder), (pipe.tokenizer_2, pipe.text_encoder_2)]
        token_ids, n_chunks = self._chunk_count([tok for tok, _ in encoders], [prompt, negative])

        embeds = [[], []]  # per text: list of per-encoder hidden states
        pooled = [None, None]
        for (tok, enc), per_text_ids in zip(encoders, token_ids):
            for ti, ids in enumerate(per_text_ids):
                chunks = []
                for c, row in self._chunk_rows(tok, ids, n_chunks):
                    out = enc(row, output_hidden_states=True)
                    # SDXL conditions on the penultimate layer; clip_skip > 1 goes further back.
                    chunks.append(out.hidden_states[-(clip_skip + 1)])
                    if enc is pipe.text_encoder_2 and c == 0:
                        pooled[ti] = out[0]
                embeds[ti].append(torch.cat(chunks, dim=1))

        negative_embeds = torch.cat(embeds[1], dim=-1)
        negative_pooled = pooled[1]
        if not negative.strip() and pipe.config.force_zeros_for_empty_prompt:
            negative_embeds = torch.zeros_like(negative_embeds)
            negative_pooled = torch.zeros_like(negative_pooled)
        return {
            "prompt_embeds": torch.cat(embeds[0], dim=-1),
            "pooled_prompt_embeds": pooled[0],
            "negative_prompt_embeds": negative_embeds,
            "negative_pooled_prompt_embeds": negative_pooled,
        }

    @torch.inference_mode()
    def _encode_sd15(self, prompt: str, negative: str, clip_skip: int) -> dict:
        tok, enc = self.pipe.tokenizer, self.pipe.text_encoder
        (token_ids,), n_chunks = self._chunk_count([tok], [prompt, negative])
        # transformers 5 flattened CLIPTextModel; older versions nest it under .text_model.
        final_norm = getattr(enc, "text_model", enc).final_layer_norm

        embeds = []
        for ids in token_ids:
            chunks = []
            for _, row in self._chunk_rows(tok, ids, n_chunks):
                out = enc(row, output_hidden_states=True)
                if clip_skip == 1:
                    chunks.append(out.last_hidden_state)
                else:
                    # A1111-style CLIP skip: take an earlier layer, then apply the final layer norm.
                    chunks.append(final_norm(out.hidden_states[-clip_skip]))
            embeds.append(torch.cat(chunks, dim=1))
        return {"prompt_embeds": embeds[0], "negative_prompt_embeds": embeds[1]}

    # -- generation ------------------------------------------------------------

    def generate(
        self,
        p: GenParams,
        on_step: Callable[[int, int, str | None], None],
        is_cancelled: Callable[[], bool],
    ) -> Image.Image:
        arch = ARCHS[p.arch]
        self.load(arch, p.model_path)
        pipe = self.pipe

        _, scheduler_cls, overrides = SAMPLERS[p.sampler]
        sched = scheduler_cls.from_config(self._base_scheduler_config, **overrides)
        pipe.scheduler = sched

        # Preview the model's predicted final image (x0) rather than the noisy latent, which stays unreadable
        # until late in sampling. Euler/DDIM return x0 from step(); multistep solvers keep it in model_outputs.
        x0: dict[str, torch.Tensor | None] = {"pred": None}
        scheduler_step = sched.step

        def step_capturing_x0(*args, **kwargs):
            out = scheduler_step(*args, **kwargs)
            x0["pred"] = out[1] if isinstance(out, tuple) and len(out) > 1 else getattr(out, "pred_original_sample", None)
            return out

        sched.step = step_capturing_x0

        encode = self._encode_sdxl if arch.id == "sdxl" else self._encode_sd15
        embeds = encode(p.prompt, p.negative, p.clip_skip)
        generator = torch.Generator("cpu").manual_seed(p.seed)

        def callback(pipe_, i, t, kwargs):
            if is_cancelled():
                raise Cancelled()
            pred = x0["pred"]
            if pred is None and getattr(sched, "model_outputs", None) and sched.model_outputs[-1] is not None:
                pred = sched.model_outputs[-1]
            if pred is not None:
                preview = latent_preview(arch, pred, None)
            else:
                sigma = None
                if sched.init_noise_sigma > 1 and getattr(sched, "step_index", None) is not None:
                    sigma = float(sched.sigmas[min(sched.step_index, len(sched.sigmas) - 1)])
                preview = latent_preview(arch, kwargs["latents"], sigma)
            on_step(i + 1, p.steps, preview)
            return kwargs

        result = pipe(
            **embeds,
            width=p.width,
            height=p.height,
            num_inference_steps=p.steps,
            guidance_scale=p.cfg,
            generator=generator,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
            output_type="latent",
        )
        return self._decode(result.images)

    @torch.inference_mode()
    def _decode(self, latents: torch.Tensor) -> Image.Image:
        vae = self.pipe.vae
        latents = latents.to(self.vae_dtype) / vae.config.scaling_factor
        image = vae.decode(latents, return_dict=False)[0]
        return self.pipe.image_processor.postprocess(image.float(), output_type="pil")[0]
