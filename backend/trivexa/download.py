"""Download model weights Trivexa needs beyond the shared models folder.

    uv run python -m trivexa.download             # everything
    uv run python -m trivexa.download ig2mv lama  # specific items (see ITEMS; an item may be several files)

Downloads resume if interrupted and are checked against a known SHA-256. If a Hub download stalls (seen with hf_xet
on HoloPart's transformer, stuck at 64 MiB), rerun with HF_HUB_DISABLE_XET=1.
"""

import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from .config import (
    HOLOPART_DIR,
    KIMODO_DIR,
    MESHFLOW_DIR,
    MV_ADAPTER_DIR,
    P3SAM_DIR,
    PARTFIELD_DIR,
    SKINTOKENS_DIR,
    TEXTURE_WEIGHTS_DIR,
    TRIPOSG_DIR,
    UNIMATE_DIR,
)

MV_ADAPTER_REPO = "huanngzh/mv-adapter"
MV_ADAPTER_REVISION = "6de4033df6b53366f3c009d22f5ec434bb55e59f"
# Gated: accept the license on huggingface.co/facebook/meshflow, then `hf auth login`, before downloading.
MESHFLOW_REPO = "facebook/meshflow"
MESHFLOW_REVISION = "9249a90e4997e105e533d6f502453fa3b344676f"
SKINTOKENS_REPO = "VAST-AI/SkinTokens"
SKINTOKENS_REVISION = "79736cad0fd84de384d5eede659b4ebd24effe33"
TRIPOSG_REPO = "VAST-AI/TripoSG"
TRIPOSG_REVISION = "2c1c516d22d58db486a058d98d31bb6177344e06"
UNIMATE_REPO = "Linzhan/UniMate"
UNIMATE_REVISION = "174fc755ec248d0c98f27759b7a95648e7f05905"
PARTFIELD_REPO = "mikaelaangel/partfield-ckpt"  # the checkpoint linked from nv-tlabs/PartField's README
PARTFIELD_REVISION = "90b9b1e08b6a12fdcb6ee26b4854a26235e1765f"
HUNYUAN3D_PART_REPO = "tencent/Hunyuan3D-Part"
HUNYUAN3D_PART_REVISION = "677174466c53571e8bacd5050dff5948734a1a4d"
HOLOPART_REPO = "VAST-AI/HoloPart"
HOLOPART_REVISION = "29cdb1417121842c4c4d59b9d283cc734f8c3aec"
KIMODO_REPO = "nvidia/Kimodo-SOMA-RP-v1.1"
KIMODO_REVISION = "6c9233af1180b8151e3c4703477104af5dce9dd5"
KIMODO_SOMA_DIR = KIMODO_DIR / "Kimodo-SOMA-RP-v1.1"
KIMODO_TEXT_DIR = KIMODO_DIR / "text-encoder"
LLAMA3_REPO = "meta-llama/Meta-Llama-3-8B-Instruct"  # gated; its file hashes are public, its files are not
LLAMA3_REVISION = "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
# An ungated copy whose files are byte-identical to Meta's (checked against Meta's hashes on download).
LLAMA3_MIRROR = ("NousResearch/Meta-Llama-3-8B-Instruct", "53346005fb0ef11d3b6a83b12c895cca40156b6c")
LLM2VEC_REPO = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
LLM2VEC_REVISION = "31474e395ada192e8ed1586db6be79fb3b70c9c0"
LLM2VEC_SUPERVISED_REPO = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
LLM2VEC_SUPERVISED_REVISION = "baa8ebf04a1c2500e61288e7dad65e8ae42601a7"
FLAN_T5_REPO = "google/flan-t5-base"
FLAN_T5_REVISION = "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"


@dataclass(frozen=True)
class HubFile:
    repo: str
    revision: str
    filename: str  # path in the repo; lands at dest/filename
    dest: Path
    sha256: str | None = None  # None: the Hub's LFS SHA-256, or the git blob id of a small non-LFS file
    mirror: tuple[str, str] | None = None  # (repo, revision) to download from instead, still checked against repo


@dataclass(frozen=True)
class UrlFile:
    url: str
    sha256: str
    filename: str
    dest: Path

def _hub_files(repo: str, revision: str, dest: Path, *filenames: str, mirror=None) -> tuple[HubFile, ...]:
    return tuple(HubFile(repo, revision, f, dest, mirror=mirror) for f in filenames)


# Kimodo's normalisation stats: small non-LFS files, so their checksums are spelled out.
KIMODO_STATS = tuple(
    HubFile(KIMODO_REPO, KIMODO_REVISION, f"stats/motion/{name}", KIMODO_SOMA_DIR, sha256)
    for name, sha256 in {
        "body/mean.npy": "d42f35f8bb7ef52fdec66ae29c5cf590bd038f4f1aabd2d560d1d0f78663ed6f",
        "body/std.npy": "965bbad497726ef4e6c46b519827e2cbcb29f974214b25718d20a725b82279ed",
        "global_root/mean.npy": "6c110980f9a0386e57a8c186f8eac7f55dc05bade92f44c4b391450ada598da1",
        "global_root/std.npy": "fb15aeed7143f5f0754611f0acf5fa5f55d1f6492a49c23e490fc62f7bf11fb0",
        "local_root/mean.npy": "4c36e8e33d6d51e6c8206ac796a52b7b3a4da7f0dc49ebdbb4f130178e3e9d31",
        "local_root/std.npy": "5f57b41f51ced7832cffa81702f6e04d8694e16475c184baec6c8be0e98c8dba",
    }.items()
)


ITEMS = {
    "t2mv": HubFile(MV_ADAPTER_REPO, MV_ADAPTER_REVISION, "mvadapter_t2mv_sdxl.safetensors", MV_ADAPTER_DIR),
    "i2mv": HubFile(MV_ADAPTER_REPO, MV_ADAPTER_REVISION, "mvadapter_i2mv_sdxl.safetensors", MV_ADAPTER_DIR),
    # Geometry-guided variants, for texturing an existing mesh from an image (ig2mv) or a prompt (tg2mv).
    "ig2mv": HubFile(MV_ADAPTER_REPO, MV_ADAPTER_REVISION, "mvadapter_ig2mv_sdxl.safetensors", MV_ADAPTER_DIR),
    "tg2mv": HubFile(MV_ADAPTER_REPO, MV_ADAPTER_REVISION, "mvadapter_tg2mv_sdxl.safetensors", MV_ADAPTER_DIR),
    # The upstream texture baker's view upscaler and inpainter (the files MV-Adapter's README points to).
    "esrgan": UrlFile(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
        "RealESRGAN_x2plus.pth",
        TEXTURE_WEIGHTS_DIR,
    ),
    "lama": UrlFile(  # same file IOPaint uses (MD5 e3aa4aaa15225a33ec84f9f4bc47e500)
        "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
        "344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9",
        "big-lama.pt",
        TEXTURE_WEIGHTS_DIR,
    ),
    # Artist-style remeshing of generated models (both files form one bundle in data/models/meshflow/).
    "meshflow": HubFile(MESHFLOW_REPO, MESHFLOW_REVISION, "meshflow/model.pth", MESHFLOW_DIR.parent),
    "meshflow-config": HubFile(
        MESHFLOW_REPO,
        MESHFLOW_REVISION,
        "meshflow/config.yaml",
        MESHFLOW_DIR.parent,
        "8ad6180b747065775a1dd6f7d4949424b69d88eea298d43ba5599dbcd4d2a132",
    ),
    # Auto-rigging (MIT). The TokenRig checkpoint already holds the skin-token VAE, so upstream's separate
    # skin_vae_2_10_32768/last.ckpt (read only for its config, which engines/skintokens.py spells out) is skipped.
    "skintokens": HubFile(
        SKINTOKENS_REPO,
        SKINTOKENS_REVISION,
        "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
        SKINTOKENS_DIR,
    ),
    # Image -> 3D (MIT). The repo's diffusers layout is kept under data/models/triposg/; engines/triposg.py builds
    # the models itself, so model_index.json and the scheduler/preprocessor configs are not needed.
    "triposg": (
        HubFile(TRIPOSG_REPO, TRIPOSG_REVISION, "transformer/diffusion_pytorch_model.safetensors", TRIPOSG_DIR),
        HubFile(
            TRIPOSG_REPO,
            TRIPOSG_REVISION,
            "transformer/config.json",
            TRIPOSG_DIR,
            "54ad779e68f9bc7303122cdf291f6db48dec7c67e41d28c05ac359daf914128a",
        ),
        HubFile(TRIPOSG_REPO, TRIPOSG_REVISION, "vae/diffusion_pytorch_model.safetensors", TRIPOSG_DIR),
        HubFile(
            TRIPOSG_REPO,
            TRIPOSG_REVISION,
            "vae/config.json",
            TRIPOSG_DIR,
            "2fb6ba9b364d42c9681879dca0f7edcb978f2db974a1ca4b15d28f9b15a5baed",
        ),
        HubFile(TRIPOSG_REPO, TRIPOSG_REVISION, "image_encoder_dinov2/model.safetensors", TRIPOSG_DIR),
        HubFile(
            TRIPOSG_REPO,
            TRIPOSG_REVISION,
            "image_encoder_dinov2/config.json",
            TRIPOSG_DIR,
            "8ed81b0d5ffd2c8b2cecd7dc1326d3eefa409bc9d525537b6b2a3c70a992fd02",
        ),
    ),
    # Text -> motion on any skeleton (checkpoints CC BY-NC 4.0). The recommended UniML3D run, final step, and its
    # frozen flan-t5-base text encoder (Apache-2.0; read with T5Tokenizer + T5EncoderModel).
    "unimate": (
        HubFile(UNIMATE_REPO, UNIMATE_REVISION, "unimate_uniml3d_f60_v3/checkpoints/checkpoint_step_150000.pt", UNIMATE_DIR),
        HubFile(UNIMATE_REPO, UNIMATE_REVISION, "unimate_uniml3d_f60_v3/dataset_stats.npy", UNIMATE_DIR),
        HubFile(
            UNIMATE_REPO,
            UNIMATE_REVISION,
            "unimate_uniml3d_f60_v3/config.json",
            UNIMATE_DIR,
            "1ffbd379d3667de7d701ed6f3418f1c9d1a66f35d2653776255008c07c051aaa",
        ),
        *_hub_files(
            FLAN_T5_REPO,
            FLAN_T5_REVISION,
            UNIMATE_DIR / "flan-t5-base",
            "config.json",
            "model.safetensors",
            "spiece.model",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ),
    ),
    # Text -> motion on the SOMA human skeleton (NVIDIA Open Model License, commercial use allowed); its text encoder
    # is the separate "kimodo-text" item.
    "kimodo": (
        HubFile(KIMODO_REPO, KIMODO_REVISION, "model.safetensors", KIMODO_SOMA_DIR),
        HubFile(
            KIMODO_REPO,
            KIMODO_REVISION,
            "config.yaml",
            KIMODO_SOMA_DIR,
            "905664ad05779b0e28c391b85dc81c9de166418bd5f471ef605f75ab746ce391",
        ),
        *KIMODO_STATS,
    ),
    # Kimodo's text encoder, LLM2Vec (MIT): Llama-3-8B-Instruct (Llama 3 Community License; ~16 GB bf16) made
    # bidirectional, with LLM2Vec's MNTP LoRA merged in and its supervised LoRA on top.
    "kimodo-text": (
        *_hub_files(
            LLM2VEC_REPO,
            LLM2VEC_REVISION,
            KIMODO_TEXT_DIR / "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            "config.json",
            "adapter_config.json",
            "adapter_model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ),
        *_hub_files(
            LLM2VEC_SUPERVISED_REPO,
            LLM2VEC_SUPERVISED_REVISION,
            KIMODO_TEXT_DIR / "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
            "adapter_config.json",
            "adapter_model.safetensors",
        ),
        *_hub_files(
            LLAMA3_REPO,
            LLAMA3_REVISION,
            KIMODO_TEXT_DIR / "Meta-Llama-3-8B-Instruct",
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            *(f"model-0000{i}-of-00004.safetensors" for i in range(1, 5)),
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            mirror=LLAMA3_MIRROR,
        ),
    ),
    # Mesh part segmentation, two alternatives. PartField (NVIDIA License: non-commercial) predicts a per-face
    # feature field that is clustered into any number of parts; the Objaverse-trained checkpoint is the only one
    # released. P3-SAM (Tencent Hunyuan 3D-Part license, not valid in the EU/UK/South Korea) segments into
    # parts automatically on a Sonata point-transformer backbone. p3sam.safetensors holds the fine-tuned Sonata
    # weights too; upstream reads facebook/sonata's sonata.pth only for its architecture config (in_channels 9,
    # enc_depths (3, 3, 3, 12, 3), enc_channels (48, 96, 192, 384, 512), enc_num_head (3, 6, 12, 24, 32),
    # enc_patch_size 1024, enc_mode, mask_token), so that file is skipped.
    "partfield": HubFile(PARTFIELD_REPO, PARTFIELD_REVISION, "model_objaverse.ckpt", PARTFIELD_DIR),
    "p3sam": HubFile(HUNYUAN3D_PART_REPO, HUNYUAN3D_PART_REVISION, "p3sam/p3sam.safetensors", P3SAM_DIR),
    # Part completion ("fill parts", MIT): a TripoSG-architecture DiT conditioned on the whole shape and one
    # segmented part, generating that part's occluded geometry. Its VAE is TripoSG's, stored in fp16.
    "holopart": (
        HubFile(HOLOPART_REPO, HOLOPART_REVISION, "transformer/diffusion_pytorch_model.safetensors", HOLOPART_DIR),
        HubFile(
            HOLOPART_REPO,
            HOLOPART_REVISION,
            "transformer/config.json",
            HOLOPART_DIR,
            "a91478087ef0ac7c4b27b4fc4962a9e32af397f9b02b9e19673ef5c1e207bbc8",
        ),
        HubFile(HOLOPART_REPO, HOLOPART_REVISION, "vae/diffusion_pytorch_model.safetensors", HOLOPART_DIR),
        HubFile(
            HOLOPART_REPO,
            HOLOPART_REVISION,
            "vae/config.json",
            HOLOPART_DIR,
            "9e1b0a706fcc5da974367e1377907337e98be3291df56742105a1a5c6e80d16d",
        ),
        HubFile(HOLOPART_REPO, HOLOPART_REVISION, "part_encoder/diffusion_pytorch_model.safetensors", HOLOPART_DIR),
        HubFile(
            HOLOPART_REPO,
            HOLOPART_REVISION,
            "part_encoder/config.json",
            HOLOPART_DIR,
            "ab139241a8e72e39f082cd53e4e6c91b6d941c66b587fdc65aec24cfcfc2ce83",
        ),
    ),
}


def _digest(path, algorithm: str) -> str:
    """SHA-256 of the file, or for "git" its git blob id (SHA-1 over a "blob <size>\\0" header and the content)."""
    if algorithm == "git":
        h = hashlib.sha1(b"blob %d\0" % Path(path).stat().st_size)
    else:
        h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _expected_digest(item: HubFile | UrlFile) -> tuple[str, str]:
    """(algorithm, digest): a spelled-out SHA-256, the Hub's LFS SHA-256, or for a small non-LFS file the git
    blob id the Hub lists (which it also does for gated repos one cannot download from yet)."""
    if item.sha256:
        return "sha256", item.sha256
    info = HfApi().model_info(item.repo, revision=item.revision, files_metadata=True)
    sibling = next(s for s in info.siblings if s.rfilename == item.filename)
    return ("sha256", sibling.lfs.sha256) if sibling.lfs else ("git", sibling.blob_id)


def _download_url(item: UrlFile, target: Path) -> None:
    """Stream to <target>.part, resuming from where a previous attempt stopped."""
    part = target.with_suffix(target.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(item.url, headers={"Range": f"bytes={done}-"} if done else {})
    with urllib.request.urlopen(req) as resp, part.open("ab" if resp.status == 206 else "wb") as out:
        while block := resp.read(1024 * 1024):
            out.write(block)
    part.replace(target)


def fetch(name: str) -> None:
    for item in ITEMS[name] if isinstance(ITEMS[name], tuple) else (ITEMS[name],):
        _fetch_file(name, item)


def _fetch_file(name: str, item: HubFile | UrlFile) -> None:
    target = item.dest / item.filename
    algorithm, expected = _expected_digest(item)
    if target.is_file() and _digest(target, algorithm) == expected:
        print(f"{name}: already present and verified")
        return
    print(f"{name}: downloading {item.filename} → {item.dest}")
    item.dest.mkdir(parents=True, exist_ok=True)
    if isinstance(item, HubFile):
        repo, revision = item.mirror or (item.repo, item.revision)
        hf_hub_download(repo, item.filename, revision=revision, local_dir=item.dest)
    else:
        _download_url(item, target)
    actual = _digest(target, algorithm)
    if actual != expected:
        target.unlink()
        sys.exit(f"{name}: checksum mismatch (got {actual}, expected {expected}); deleted, please retry")
    print(f"{name}: verified")


if __name__ == "__main__":
    for item in sys.argv[1:] or ITEMS:
        if item not in ITEMS:
            sys.exit(f"Unknown item {item!r}; choose from {', '.join(ITEMS)}")
        fetch(item)
