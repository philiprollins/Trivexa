# Vendored third-party code

Copied (not pip-installed) to avoid each project's heavy, unrelated dependencies.
Only the files needed for inference are included; local edits are marked `# Trivexa:`.

| Package | Source | Commit | License |
|---|---|---|---|
| `mvadapter/` | https://github.com/huanngzh/MV-Adapter | `4277e0018232bac82bb2c103caf0893cedb711be` | Apache-2.0 (`mvadapter/LICENSE`) |
| `birefnet/` | https://huggingface.co/ZhengPeng7/BiRefNet | `e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4` | MIT |
| `tsr/` | https://github.com/VAST-AI-Research/TripoSR | `107cefdc244c39106fa830359024f6a2f1c78871` | MIT (`tsr/LICENSE`) |
| `meshflow/` | https://github.com/facebookresearch/meshflow | `55f56f60e1bbf98d1c1991670ac998094d5f59ae` | FAIR Noncommercial Research (`meshflow/LICENSE`) |
| `dinov3/` | https://github.com/facebookresearch/dinov3 | `6876159a11b4df116f30f667f8c9888617df0751` | DINOv3 License (`dinov3/LICENSE.md`) |
| `skintokens/` | https://github.com/VAST-AI-Research/SkinTokens | `273b691d35989d71cd17ff2895fdc735097b92d1` | MIT (`skintokens/LICENSE`) |

Weights are not vendored: MV-Adapter adapters are downloaded to `data/models/mv-adapter/`,
BiRefNet weights are read from `<models dir>/background_removal/birefnet.safetensors`.
TripoSR weights (`config.yaml`, `model.ckpt`, `dino/config.json` from `stabilityai/TripoSR`) are read from `<models dir>/triposr/`.
MeshFlow weights (`config.yaml`, `model.pth` from the gated `facebook/meshflow`) are read from `data/models/meshflow/`.

`meshflow/` keeps only `models/`, `pipelines/` and `utils/` (no training, evaluation or Gradio code). Trivexa builds the
models itself (`engines/meshflow.py`) to read the checkpoint once, and builds the DINOv3 image encoder from the
vendored `dinov3/` instead of torch.hub. The
FAIR Noncommercial Research License covers the code, the weights *and their outputs*: remeshed models are for
noncommercial research use only.

`dinov3/` keeps only the ViT backbone (`models/vision_transformer.py`, `layers/`, `utils/`); the ViT-L/16 arguments
from `hub/backbones.py` live in `engines/meshflow.py`. The training-only fp8 layer is dropped. Its weights
(`dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`) are read from `data/models/meshflow/`.

`skintokens/` is upstream's `src/` without the Blender (bpy) parser, the bpy/HTTP server, and the Lightning
dataset/datapath modules; `configs/skeleton/` is kept for the bone-order configs. Trivexa loads and saves meshes
itself (`engines/skintokens.py`, `engines/gltf_skin.py`). Local edits: `ModelSpec` is a plain `nn.Module` (no
Lightning), attention runs on PyTorch SDPA instead of flash-attn, the Qwen3-0.6B config is spelled out instead of
fetched from the Hub, and the skin VAE is built from its config because its weights are part of the TokenRig
checkpoint (`data/models/skintokens/experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt`).

`mvadapter/` includes the texture pipeline (`pipelines/pipeline_texture.py`, `utils/mesh_utils/`) for the upstream
texture baker. It needs the optional `texture` dependencies; `mesh_utils/__init__.py` is kept empty so the camera
helpers used elsewhere import without them. nvdiffrast (installed, not vendored) is under the NVIDIA Source Code
License, which allows non-commercial use only.
