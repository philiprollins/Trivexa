"""Mesh -> skeleton + skinning weights with SkinTokens (VAST-AI, MIT; vendored code, local weights).

TokenRig, a Qwen3-0.6B-shaped transformer, reads a point cloud of the mesh (54k surface samples with normals, encoded
by a Michelangelo shape encoder) and writes one token sequence: the skeleton as quantised joint positions in a
depth-first tree, then four "skin tokens" per bone. An FSQ-CVAE decodes each bone's tokens into its weights at the
sampled points, and those are interpolated onto the mesh's vertices.

Upstream loads and saves meshes through Blender; this engine works on trimesh meshes directly. Blender's glTF importer
turns Y-up into Z-up, which is the frame the model was trained in, so meshes are rotated the same way going in and
back coming out.
"""

import logging
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from transformers import StoppingCriteria, StoppingCriteriaList

from . import Cancelled
from ..config import SKINTOKENS_DIR

log = logging.getLogger(__name__)

CHECKPOINT = SKINTOKENS_DIR / "experiments" / "articulation_xl_quantization_256_token_4" / "grpo_1400.ckpt"
# The model_config of upstream's skin_vae_2_10_32768/last.ckpt, which upstream reads only for this (the weights are
# part of the TokenRig checkpoint).
VAE_CONFIG = {
    "model": {
        "in_channels": 4,
        "cond_channels": 3,
        "latent_channels": 512,
        "num_attention_heads": 12,
        "width_encoder": 768,
        "width_decoder": 768,
        "num_layers_encoder": 2,
        "num_layers_decoder": 10,
        "embedding_type": "frequency",
        "embed_frequency": 8,
        "embed_include_pi": True,
        "is_learned_queries": True,
        "use_pmpe": True,
        "FSQ_dict": {"levels": [8, 8, 8, 8, 8], "dim": 512},
    },
    "sample": {"cond_tokens": 384, "sample_tokens": 32, "compress_tokens": 32},
}
# The class token steers the skeleton style. "articulation" (Articulation-XL: any kind of object) is what upstream's
# demo uses; "vroid" asks for a skeleton like the VRoid humanoids in its training data (on a robot: ~40-46 bones with
# finger chains, against ~22). This checkpoint emits no body/hand part tokens, so the bones cannot be given VRoid names.
STYLES = {"auto": "articulation", "humanoid": "vroid"}
# upstream demo defaults
DEFAULT_BEAMS = 10
GROUP_PER_VERTEX = 4  # weights kept per vertex, as upstream exports them
MAX_LENGTH = 2048
# Z-up (Blender, the model's frame) <-> Y-up (glTF, Trivexa's frame), as Blender's glTF importer converts.
_Y_TO_Z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


@dataclass
class RigParams:
    mesh: trimesh.Trimesh  # Y-up
    style: str = "auto"  # see STYLES
    seed: int = 0
    beams: int = DEFAULT_BEAMS
    temperature: float = 1.0
    top_k: int = 5
    top_p: float = 0.95
    repetition_penalty: float = 2.0


@dataclass
class Rig:
    joints: np.ndarray  # (J, 3) joint (bone head) positions in the mesh's frame
    parents: list[int]  # parent index per joint, -1 for the root
    names: list[str]
    weights: np.ndarray  # (N, J) per-vertex weights, at most GROUP_PER_VERTEX non-zero per row, rows sum to 1


def available() -> bool:
    return CHECKPOINT.is_file()


class _Progress(StoppingCriteria):
    """Cancel and progress hook, called once per generated token.

    The skeleton's length is unknown until its end token, so that phase moves asymptotically through the first 40%;
    after it, the bone count fixes how many skin tokens follow.
    """

    def __init__(self, tokenizer, start_tokens: list[int], tokens_per_skin: int, on_step, is_cancelled):
        self.tokenizer = tokenizer
        self.start = start_tokens
        self.tokens_per_skin = tokens_per_skin
        self.on_step = on_step
        self.is_cancelled = is_cancelled
        self.skin_start: int | None = None
        self.skin_total = 0

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> torch.BoolTensor:
        if self.is_cancelled():
            raise Cancelled()
        n = input_ids.shape[1]
        if self.skin_start is None:
            seq = input_ids[0].tolist()
            if self.tokenizer.eos in seq:
                self.skin_start = n
                bones = self.tokenizer.bones_in_sequence(ids=np.array(self.start + seq))
                self.skin_total = max(bones * self.tokens_per_skin, 1)
        if self.skin_start is None:
            pct = 40 * (1 - np.exp(-n / 250))
        else:
            pct = 40 + 58 * min((n - self.skin_start) / self.skin_total, 1)
        self.on_step(int(pct), 100, None)
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


class SkinTokensEngine:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.model = None
        self._transform_config = None

    def load(self) -> None:
        if self.model is not None:
            return
        from ..vendor.skintokens.model.tokenrig import TokenRig

        log.info("Loading SkinTokens from %s", CHECKPOINT)
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False, mmap=True)
        hp = ckpt["hyper_parameters"]
        model = TokenRig(
            model_config=hp["model_config"] | {"vae_model_config": VAE_CONFIG},
            transform_config=hp["transform_config"],
            tokenizer_config=hp["tokenizer_config"],
        )
        state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["state_dict"].items()}
        model.load_state_dict(state)
        del ckpt, state
        self.model = model.to(self.device).eval().requires_grad_(False)
        self._transform_config = hp["transform_config"]["predict_transform"]

    def unload(self) -> None:
        if self.model is None:
            return
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate(
        self,
        p: RigParams,
        on_step: Callable[[int, int, str | None], None],
        is_cancelled: Callable[[], bool],
    ) -> Rig:
        from ..vendor.skintokens.data.sampler import get_sampler
        from ..vendor.skintokens.model.tokenrig import decode
        from ..vendor.skintokens.rig_package.info.asset import Asset

        self.load()
        model = self.model
        random.seed(p.seed)
        np.random.seed(p.seed % 2**32)
        torch.manual_seed(p.seed)

        # Upstream's predict transform: the "affine" augment (centre the bounding box, scale its longest side to
        # [-1, 1]) and unit normals, then 54k area-weighted surface samples. ("trim" needs an existing skeleton.)
        vertices = p.mesh.vertices @ _Y_TO_Z.T
        center = (vertices.min(0) + vertices.max(0)) / 2
        scale = float((vertices.max(0) - vertices.min(0)).max()) / 2
        unit = trimesh.Trimesh((vertices - center) / scale, p.mesh.faces, process=False)
        asset = Asset(
            vertices=np.asarray(unit.vertices, dtype=np.float32),
            faces=np.asarray(unit.faces),
            vertex_normals=np.asarray(unit.vertex_normals, dtype=np.float32),
            face_normals=np.asarray(unit.face_normals, dtype=np.float32),
        )
        sampled = get_sampler(**dict(self._transform_config["sampler"])).sample(asset=asset)
        points = torch.from_numpy(sampled.sampled_vertices).float().to(self.device)
        normals = torch.from_numpy(sampled.sampled_normals).float().to(self.device)
        on_step(0, 100, None)

        cls = STYLES[p.style]
        start = model.make_start_tokens(cls=[cls])[0]
        progress = _Progress(model.tokenizer, start, model.tokens_per_skin, on_step, is_cancelled)
        res = model.generate(
            vertices=points,
            normals=normals,
            cls=cls,
            return_decode_dict=True,
            max_length=MAX_LENGTH,
            top_k=p.top_k,
            top_p=p.top_p,
            temperature=p.temperature,
            repetition_penalty=p.repetition_penalty,
            num_return_sequences=1,
            num_beams=p.beams,
            do_sample=True,
            stopping_criteria=StoppingCriteriaList([progress]),
        )
        if is_cancelled():
            raise Cancelled()
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            out = decode(
                cond=res.cond,
                cond_latents=res.cond_latents,
                inputs_ids=res.output_ids,
                tokenizer=model.tokenizer,
                tokens_per_skin=model.tokens_per_skin,
                vae=model.vae,
            )
        skeleton = out["detokenize_output"]
        if out["skin_pred"] is None:
            raise RuntimeError("SkinTokens ran out of tokens before finishing the skin weights; try another seed")
        on_step(99, 100, None)

        # Weights at the samples -> at the vertices: inverse-distance mix of the 8 nearest samples (upstream's
        # Asset.from_data), then the GROUP_PER_VERTEX largest per vertex, renormalised (upstream's export).
        sample_skin = out["skin_pred"].float().cpu().numpy()
        k = min(8, len(sampled.sampled_vertices))
        dist, idx = cKDTree(sampled.sampled_vertices).query(asset.vertices, k=k)
        mix = 1.0 / (dist + 1e-8)
        mix /= mix.sum(1, keepdims=True)
        weights = _sparsify(np.einsum("nk,nkj->nj", mix, sample_skin[idx]), GROUP_PER_VERTEX)

        joints = (np.asarray(skeleton.joints, dtype=np.float64) * scale + center) @ _Y_TO_Z  # back to Y-up
        return Rig(
            joints=joints,
            parents=[int(i) for i in skeleton.parents],
            names=[f"bone_{i}" for i in range(skeleton.J)],  # upstream's names
            weights=weights.astype(np.float32),
        )


def _sparsify(skin: np.ndarray, keep: int) -> np.ndarray:
    """Keep each row's `keep` largest weights (negatives dropped) and make the row sum to 1."""
    skin = np.clip(skin, 0, None)
    keep = min(keep, skin.shape[1])
    top = np.argpartition(-skin, keep - 1, axis=1)[:, :keep]
    out = np.zeros_like(skin)
    rows = np.arange(len(skin))[:, None]
    out[rows, top] = skin[rows, top]
    total = out.sum(1, keepdims=True)
    empty = total[:, 0] <= 1e-8
    out[empty, skin[empty].argmax(1)] = 1.0  # all-zero row: the strongest bone takes it
    total[empty] = 1.0
    return out / total
