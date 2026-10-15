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
| `triposg/` | https://github.com/VAST-AI-Research/TripoSG | `fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c` | MIT (`triposg/LICENSE`); the DiT is Tencent Hunyuan Community License (see below) |
| `pixal3d/` | https://github.com/TencentARC/Pixal3D | `f7cf38429b0bd264f1995f0f8743a88b1c728b94` | MIT (`pixal3d/LICENSE`, third parties in `pixal3d/NOTICE`) |
| `naf/` | https://github.com/valeoai/NAF | `37f2dfc180f2de53d98bd601109c0da0dd6b0f43` | Apache-2.0 (`naf/LICENSE`) |
| `moge/` | https://github.com/microsoft/MoGe | `74fbce054ebed49800de42d0ad0e83495065719a` | MIT (`moge/LICENSE`); its DINOv2 is Apache-2.0 |

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

`triposg/` keeps `models/`, `schedulers/` and `inference_utils.py` (no pipelines, scripts or RMBG; Trivexa runs the
sampling loop itself in `engines/triposg.py` and cuts images out with BiRefNet). Upstream's `triposg/LICENSE` is the
Tencent FlashVDM Community License, which covers only the FlashVDM-derived fast decoder: that code (the flash
attention processor, `flash_extract_geometry`) is removed, together with its diso dependency (CC BY-NC, compiled
CUDA), so the repo's MIT `LICENSE` is the one kept. Meshing uses upstream's other path,
`hierarchical_extract_geometry`, with PyMCubes in place of scikit-image. `models/transformers/triposg_transformer.py`
is adapted from HunyuanDiT and keeps its Tencent Hunyuan Community License header: no license in the EU, and outputs
must not be used to improve other large language models. Weights (`VAST-AI/TripoSG`, MIT) are read from
`data/models/triposg/` in the Hub's diffusers layout.

`pixal3d/` keeps the inference half of upstream's `pixal3d/` package: `models/` (flow models and VAE decoders),
`modules/` (without the spconv and torchsparse convolution backends), `pipelines/samplers/` and `representations/`.
No pipelines, trainers, renderers or data toolkit: Trivexa runs the cascade itself (`engines/pixal3d.py`, after
`pipelines/pixal3d_image_to_3d.py` and `inference.py`), cuts images out with BiRefNet instead of RMBG-2.0, and keeps
the flow models in system RAM between stages. `image_cond.py` is `DinoV3ProjFeatureExtractor` and its projection
utilities from `trainers/flow_matching/mixins/image_conditioned_proj.py`, changed to share one DINOv3 and one NAF
across the four stages (upstream loads DINOv3 per stage and NAF through torch.hub), to iterate DINOv3's blocks where
transformers 5 keeps them, and to project only the occupied voxels instead of the whole grid (the same features; at
1536 the full 96^3 grid alone needs 7 GB). Local edits: attention defaults to PyTorch SDPA instead of flash-attn.
The compiled TRELLIS.2 libraries (FlexGEMM, CuMesh, o-voxel) are installed by the `pixal3d` extra, not vendored.
Weights (`TencentARC/Pixal3D`, MIT, but not offered in the EU) are read from `<models dir>/TencentARC/Pixal3D/`.

`naf/` is NAF's `src/model/naf.py` and `src/layers/`, Pixal3D's DINOv3 feature upsampler. Its cross-attention used
NATTEN's dilated neighbourhood attention (a compiled CUDA extension); `layers/attentions.py` computes the same thing in
PyTorch directly on the low-resolution keys and values, which matches NATTEN's border and dilation behaviour exactly
when the upsampling factor is an integer (always, in Pixal3D). The weights are the `naf.*` tensors of
`<models dir>/clip_vision/dino_v3_L_naf_fp32.safetensors`, next to DINOv3 ViT-L/16 itself.

`moge/` is MoGe's v2 model (`model/v2.py`, `model/modules/` with its DINOv2) for Pixal3D's camera field-of-view
estimate. The DINOv2 training utilities and `moge/utils` are dropped except the focal/shift solver
(`utils/geometry.py`, scipy); `infer()` builds the intrinsics itself instead of through utils3d, and its
`force_projection` (which needs utils3d) is off. The weights (`<models dir>/geometry_estimation/
moge_2_vitl_normal_fp16.safetensors`) carry no config; `engines/pixal3d.py` spells out `configs/train/v2.json`'s.
