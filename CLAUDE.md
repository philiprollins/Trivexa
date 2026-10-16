# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Trivexa is a local, Tripo-inspired web studio for image → multi-view → 3D → texture generation on one 16 GB GPU
(RTX 4090 Laptop). A FastAPI backend (`backend/trivexa`) runs all inference itself with PyTorch/diffusers and serves a
React/Vite UI (`frontend/`). Everything generated is saved locally under `data/` (gitignored).

**Hard constraint:** never use ComfyUI as an engine or call its API — it won't be running. Trivexa only *reads model
files* from `~/ComfyUI/models` (`TRIVEXA_MODELS_DIR`). Reading ComfyUI source for reference is fine.

## Commands

```bash
./run.sh                                    # build UI if stale, uv sync (+ "texture", "pixal3d" extras), serve on 127.0.0.1:8000
cd backend && .venv/bin/uvicorn trivexa.main:app --reload --port 8000   # backend dev
cd frontend && npm run dev                  # UI dev on :5173, proxies /api (incl. websocket) and /files to :8000
cd frontend && npm run build                # type-check (tsc -b) + build; the server serves frontend/dist
cd backend && uv run python -m trivexa.download [t2mv i2mv ig2mv tg2mv esrgan lama meshflow meshflow-config skintokens triposg unimate kimodo kimodo-text partfield p3sam holopart]  # resumable, SHA-256 checked
cd backend && uv sync --extra texture       # compiles nvdiffrast; needs CUDA_HOME=/usr/local/cuda on PATH
CXX=$PWD/backend/tools/cxx20 NVCC_APPEND_FLAGS=-std=c++20 uv sync --extra texture --extra pixal3d  # as run.sh does
```

- There are no automated tests or linters. Changes have been verified by driving the real API (POST a job, watch
  the `/api/ws` websocket for `job`/`progress` events) and by checking the UI in Chrome. `npm run build` is the
  frontend type check.
- The `pixal3d` extra compiles FlexGEMM, CuMesh and o-voxel from pinned git commits. Their `setup.py` pins
  `-std=c++17`, which torch 2.14's headers reject (`#error C++20 or later`): `tools/cxx20` swaps the flag for `.cpp`
  files and `NVCC_APPEND_FLAGS` overrides it for `.cu` (nvcc takes the last `-std`). `uv lock` clones TRELLIS.2 and
  its Eigen submodule from GitLab, which can take 15+ minutes.
- `uv sync` removes packages not in `pyproject.toml`/`uv.lock`, so anything installed ad hoc with `uv pip install`
  disappears on the next `./run.sh`. Add real dependencies with `uv add` (torch/torchvision come from the cu130 index).
- Stop the server with `kill $(pgrep -f "[u]vicorn trivexa")` — a bare `pkill -f "uvicorn trivexa"` matches and
  kills the shell running it.
- `settings.toml` (repo root, or `TRIVEXA_SETTINGS`) is read at startup: restart after editing.

## Models in use

From `~/ComfyUI/models` (architecture is detected from safetensors headers, not filenames):

| Use | Model |
|---|---|
| Image, SDXL | `juggernautXL_ragnarok` (photoreal — best base for multiview/texture), `waiIllustriousSDXL_v140` (anime — gives poor multiview results) |
| Image, SD 1.5 | `aZovyaRPGArtistTools_v4Pruned`, `revAnimated_v2Rebirth`, `v1-5-pruned-emaonly-fp16` |
| Hidden (not image models) | `ace_step*`, `stable-audio-open-1.0`, `yue2_3b_int8_convrot` (audio) |
| Background removal | `background_removal/birefnet.safetensors` (= ZhengPeng7/BiRefNet) |
| Image → 3D | `triposr/` (`config.yaml`, `model.ckpt`, `dino/config.json` = stabilityai/TripoSR) |
| Image → textured 3D | `TencentARC/Pixal3D/` (`pipeline.json` + `ckpts/`; the four `*_mv` flow checkpoints serve multiview input) |
| Pixal3D image encoder | `clip_vision/dino_v3_L_naf_fp32.safetensors` (DINOv3 ViT-L/16, transformers-4 key names, + `naf.*` NAF upsampler) |
| Pixal3D camera FOV | `geometry_estimation/moge_2_vitl_normal_fp16.safetensors` (MoGe-2 ViT-L normal; no config inside) |

Downloaded by Trivexa into `data/models/` (via `trivexa.download`):

| Use | Model |
|---|---|
| Multiview | MV-Adapter SDXL `t2mv` (text) and `i2mv` (image) |
| Texture | MV-Adapter SDXL `ig2mv` (image + geometry) and `tg2mv` (prompt + geometry) |
| Upstream texture baker | `RealESRGAN_x2plus.pth`, `big-lama.pt` |
| Remesh | MeshFlow `meshflow/config.yaml` + `model.pth` (gated `facebook/meshflow`, FAIR noncommercial, outputs too) |
| Remesh image guidance | `meshflow/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` (from Meta's DINOv3 form, fetched by hand) |
| Rig | SkinTokens `skintokens/experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt` (VAST-AI, MIT; includes the skin VAE) |
| Image → 3D | TripoSG `triposg/{transformer,vae,image_encoder_dinov2}/` (VAST-AI/TripoSG, MIT weights; ~7.9 GB, fp32 on disk) |
| Segment | PartField `partfield/model_objaverse.ckpt` (NVIDIA License, non-commercial) |
| Animate | UniMate `unimate/unimate_uniml3d_f60_v3/` (`checkpoints/checkpoint_step_150000.pt`, `config.json`, `dataset_stats.npy`; CC BY-NC 4.0) + `unimate/flan-t5-base/` |
| Animate (not wired up yet) | Kimodo `kimodo/Kimodo-SOMA-RP-v1.1/` (NVIDIA Open Model License; SOMA human skeleton) + `kimodo/text-encoder/` (LLM2Vec: Llama-3-8B-Instruct from NousResearch's byte-identical mirror, checked against Meta's hashes, + McGill-NLP's two LoRAs) |
| Segment (not wired up yet) | P3-SAM `p3sam/p3sam/p3sam.safetensors` (Hunyuan 3D-Part license, not EU/UK/KR; includes its fine-tuned Sonata backbone) |
| Fill parts (not wired up yet) | HoloPart `holopart/{transformer,vae,part_encoder}/` (VAST-AI/HoloPart, MIT; ~3.8 GB; the VAE is TripoSG's in fp16) |

Still unused on disk: TRELLIS.2 itself (`diffusion_models/trellis_2_*`, `vae/trellis_2_*`, `microsoft/TRELLIS.2-4B`);
Pixal3D's vendored code is TRELLIS.2's, so it would be a small step from there. SD 2.x / MV-Adapter SD2.1 variants are deliberately not supported.

## Built so far

Image tool (SDXL + SD 1.5, long-prompt chunking, live x0 previews, cancel) → Multiview tool (MV-Adapter t2mv/i2mv,
6 views at 768 px) → 3D Model tool (TripoSG by default, Pixal3D for shape + PBR texture in one pass, from one image or several views of a multiview set, or TripoSR; orbit viewer with colour/clay/wire, GLB/OBJ) → Segment tool (PartField,
model → N parts, one GLB node / OBJ object each, keeping colours or texture; the viewer's "Parts" mode colours them) → Remesh tool (MeshFlow,
dense mesh → ~1k-face artist-style mesh, colours transferred as vertex colours) → Texture tool (ig2mv or tg2mv
views baked into a UV texture by MV-Adapter's own pipeline, with Trivexa's lean baker as a fallback) → Rig tool (SkinTokens,
skeleton + skin weights in a skinned GLB; the viewer shows bones and runs a pose test) → Animate tool (UniMate,
prompt → 2 s of motion appended to the rigged GLB as a glTF animation; the viewer plays it, in place or travelling). Gallery, favorites,
"Reuse" settings, and per-asset actions chain the tools (image → Multiview / 3D, multiview set → 3D, model → Segment / Remesh / Texture / Rig, rigged model → Animate).
A side-by-side baker benchmark is in `data/benchmarks/texture-2026-10-12/`.

## Architecture

**Request flow.** Each `POST /api/generate/{image,multiview,3d,segment,remesh,texture,rig,animate}` in `main.py` validates input, resolves model
files (`models.resolve_checkpoint` only accepts ids from the on-disk listing), builds a `params` dict and submits one
`Job` per output with a `run` closure. `JobManager` (`jobs.py`) runs jobs one at a time on a worker thread under a GPU
lock and pushes `jobs` / `job` / `progress` (with preview data URLs) events over `/api/ws`. The closure calls an engine,
then `store.save_*`, and the saved asset rides back on the final `job` event. Endpoints that touch the asyncio queue
must be `async def`.

**One big pipeline on the GPU.** `_exclusive(engine)` unloads the other engines before a job. Image-to-multiview and
ig2mv peak ~14.4 GB even with the text encoders parked in system RAM after prompt encoding (`multiview.py`); the
upstream texture baker peaks ~12 GB, so the MV pipeline is unloaded before baking. Keep this in mind before keeping
anything else resident.

**Engines** (`backend/trivexa/engines/`):
- `diffusion.py` — `DiffusionEngine` with an `Arch` spec per architecture (pipeline class, config repo, latent→RGB
  preview factors). Single-file checkpoints load with configs/tokenizers cached in `data/hf-configs/` and
  `local_files_only=True`. Shared helpers used by other engines: `SAMPLERS`, `make_step_callback` (cancel + x0
  previews, optionally tiled as a grid), `decode_latents` (VAE in bf16, one latent at a time — avoids fp16 NaNs and
  the fp32 spike).
- `multiview.py` — MV-Adapter modes `t2mv`/`i2mv` (Plücker camera conditioning) and `ig2mv`/`tg2mv`
  (caller-supplied position+normal maps, row+column attention). Uses MV-Adapter's shifted-SNR scheduler around any sampler.
- `bg_removal.py` (BiRefNet, kept in RAM, moved to GPU only while running), `triposr.py` (image → vertex-coloured mesh).
- `triposg.py` — TripoSG: 1.5B rectified-flow DiT (fp16, CFG 7 with all-zero image features as the negative, 50
  steps ≈ 13 s) on DINOv2-L features of the cut-out image, flattened on white and padded square (upstream's
  `image_process`, with BiRefNet instead of RMBG-1.4). The VAE's SDF is meshed with upstream's hierarchical extractor
  (narrow-band refinement, PyMCubes); its dense pass starts at 128³ instead of 256³, which gives the identical mesh
  in half the time (512³: ~11 s, ~850k faces, watertight). Untextured; output is already Y-up facing +Z (no
  rotation). "Face limit" decimates with pymeshlab `preservetopology=True` and without upstream's
  `merge_close_vertices`, both of which otherwise break watertightness (~9 s for 850k → 200k). ~3.7 GB resident,
  ~5.3 GB peak. `params.face_limit` is the request; `params.faces` stays the saved mesh's count. When the limit
  reduced the mesh, the raw one is kept as `model_full.glb` (`model_urls.glb_full`, `params.full_faces`, ~15 MB at
  850k faces; no OBJ). Remesh reads it in preference to `model.glb`; Texture and Rig use the decimated mesh.
- `pixal3d.py` — Pixal3D (TRELLIS.2 backbone): DINOv3 (+ NAF-upsampled) features projected onto the voxel grid
  through the input's camera, whose FOV MoGe-2 estimates (`camera_from_fov` = upstream's `distance_from_fov`). Runs
  upstream's cascade itself: sparse structure → shape latent at 512 → upsampled shape latent at 1024/1536 (capped at
  `MAX_TOKENS`, lowering the resolution by 128 until it fits) → texture latent → decode → `o_voxel.postprocess.to_glb`
  (CuMesh remesh, UV unwrap, nvdiffrast bake of base colour/metallic/roughness). The four 1.3B flow models (bf16
  blocks, fp32 on disk) stay on the CPU (~14 GB RAM) and each moves to the GPU for its stage; DINOv3/NAF stay on the
  GPU through sampling. Load ~50 s; 1024 ~55 s, 1536 ~140 s; peak ~9 GB. Noise comes from the global CPU generator
  seeded once, as upstream. Images are cropped/flattened on **black** (upstream's preprocess). Output is a PBR
  `trimesh` turned to Y-up facing +Z (upstream's own final rotation faces −Z); `params.textured`, `fov_used`,
  `resolution_used`. Texture/Remesh/Rig accept it like any model.
  **Multi-view** (`Pixal3DParams.views`, API `source_views`: azimuths of a multiview set, front included): upstream's
  `inference_mv.py` path. The four flow models are swapped for the `*_mv` fine-tunes (`_swap_flows`, ~50 s; the
  decoders are shared, and the missing `ss_flow…_mv.json` falls back to the identical single-view config).
  `DinoV3ProjMultiViewFeatureExtractor` projects each view through `calc_mat_i = F·C₀⁻¹·Cᵢ` and averages them.
  MV-Adapter's orbit already is upstream's rig (Z-up, front camera on −Y, azimuth 90 on +X, unit cube at 1/1.1 of
  the frame), so views are cut out with BiRefNet, kept uncropped, and given perspective cameras at 20° FOV
  (`orbit_cameras`, distance 3.119; MV-Adapter is orthographic, which the narrow FOV approximates). MoGe isn't used.
  Four views at 1536 ~150 s, peak ~10.6 GB (six views at 1024 ~10.3 GB); the side profile follows the side views
  instead of being guessed.
- `meshflow.py` — MeshFlow DiT + MeshVAE decoder, optionally guided by a reference image encoded with DINOv3
  ViT-L/16 (built locally instead of via torch.hub; CFG 2.5 on the image condition, ~10 s; shape-only ~5 s, no CFG).
  The image is cut out with BiRefNet and flattened on white (`prepare_reference(..., background=1.0)`). The checkpoint
  has no vertex-count input (`use_proj_cond_on_temb` is off; forcing it on fails to load `t_embedder.cond_proj`), so
  "Detail" instead varies the latent sequence length (`DETAIL_LENGTHS`, 3072–8192; trained at 4096). That length is
  the most vertices the decoder can emit, and every layer is length-agnostic, so one loaded model serves all levels.
  Below ~2048 meshes tear; 8192 is markedly more non-manifold. Colours are
  area-averaged from the source. ~4.2 GB resident (+1.2 GB DINOv3), ~6.3 GB peak. Output is in the source's frame but is
  **non-manifold** in places (every 3-clique of the predicted edge graph becomes a triangle), which crashes UVAtlas:
  `texture_upstream` detaches faces on non-manifold edges before unwrapping.
- `skintokens.py` — SkinTokens: TokenRig (Qwen3-0.6B-shaped, SDPA) writes the skeleton as tokens, then 4 skin tokens
  per bone that its FSQ-CVAE decodes into weights at 54k surface samples, interpolated to the vertices (8-NN inverse
  distance), top 4 kept. The model was trained on Blender-imported meshes, so input is turned Y-up → Z-up and joints
  back. Style "auto"/"humanoid" is the `articulation`/`vroid` class token; the checkpoint emits no part tokens, so
  bones are always `bone_<i>`. Progress/cancel come from a `StoppingCriteria`. ~4 GB peak, ~4–6 s per rig once loaded.
  `gltf_skin.py` appends joints (translation-only nodes), inverse bind matrices and JOINTS_0/WEIGHTS_0 to trimesh's
  GLB export; `store.save_model3d(..., glb=...)` writes it (the OBJ stays unrigged). Rigs have `params.rigged`, `bones`.
- `unimate.py` — UniMate: a 74M flow-matching transformer (graph attention, adaLN text) that animates any skeleton
  of 2–71 joints. It is conditioned on the skeleton, not trained per rig: the rest pose canonicalised as upstream's
  stage 4 does (BFS joint order, a "facing pair" turned to +Z, root XZ at the origin, leaf-to-leaf diameter 2,
  grounded), graph distances / edge types / depths / Laplacian eigenvectors, and a flan-t5-base embedding of each
  joint's *label* ("Left Thigh"). Labels and the facing pair come from upstream's offline rule annotator, run on a
  temporary export NPZ; SkinTokens' `bone_<i>` rigs get geometric labels (hips, spine, limbs, fingers on humanoids;
  facing comes out +Z), and with no pair the rest pose is taken to face +Z (Trivexa's convention). Captions are
  sent as "An object <prompt>." (every training caption reads so). Sampling: dopri5 over the velocity field, CFG 3,
  60 frames at 30 fps; ~15 s, <1 GB VRAM. Decoding inverts the canonical transform: with identity rest rotations,
  the glTF rotation of every joint is q⁻¹·r·q (r the T-pose-relative rotation, q the facing turn), and the root
  path maps back through scale/centre/ground. `params.motion_consistency` is the mean gap between the rotations'
  FK and the joint positions the model also predicts (~0.01 of the diameter). `gltf_skin.add_animation` appends the
  clip to the source GLB (the OBJ stays unanimated); `read_skeleton` reads a skin's joints back. Contact-heavy motions
  can float (some jump seeds rise two body heights) or slide, as upstream warns.
- `partfield.py` — PartField: a PVCNN point encoder (100k surface samples) + triplane transformer give 448-channel
  part features (fp16, ~1 s); each face averages a few dozen samples (`FACE_SAMPLE_BUDGET`). Faces are clustered as
  upstream does for meshes, Ward agglomerative under face adjacency, so each part is one connected region and the part
  count is where the merge tree is cut ("Simple/Balanced/Detailed" = 5/10/18, as Tripo's presets; or 2–32).
  Adjacency is built on a position-welded copy: textured GLBs load split at UV seams, which otherwise turns a Pixal3D
  mesh into thousands of components. Loose pieces under 0.5% of the faces (Pixal3D/MeshFlow floaters and inner shells)
  are not clustered but take the nearest clustered face's label; otherwise they gather into one scattered part.
  200k faces ~8 s, ~950k ~70 s (the clustering, CPU); ~6–7 GB peak. The checkpoint is a Lightning file whose config is
  a pickled yacs `CfgNode`: it is unpickled with a whitelist instead of installing yacs. Output: `split_parts` makes a
  `trimesh.Scene` with nodes `part_01` (largest by area) …, which `store.save_model3d` accepts in place of a mesh;
  `params.segmented`, `parts`. A rig does not survive segmentation. The other tools load segmented GLBs with
  `force="mesh"`, i.e. as one merged mesh.
- `texture.py` — renders the ig2mv condition maps with headless OpenGL (moderngl/EGL) using MV-Adapter's own camera
  matrices, plus the **lean** baker (xatlas + projection, unseen texels keep vertex colours).
- `texture_upstream.py` — wraps the vendored MV-Adapter `TexturePipeline` (UVAtlas, RealESRGAN ×2, LaMa view
  inpainting) as the **upstream** baker. `main._texture_baker()` picks it from `settings.toml` and falls back to lean
  if its packages or weights are missing.

**Mesh conventions.** Meshes are Y-up and face +Z (glTF front). TripoSR output is rotated 180° to satisfy this and
records `params.front = "+z"`; older models lack it and face −Z (remeshing turns those to +Z). `params.engine` chains
the steps, e.g. `triposr + meshflow + mv-adapter`; remeshes also set `params.remeshed`. The texture tool's `front_yaw` turns the mesh in the
lean path and is folded into the camera azimuths in the upstream path.

**Storage** (`store.py`). One SQLite table `assets` with `kind` = `image` | `multiview` | `model3d` (textured models
are `model3d` with `params.textured`), `parent_id` for lineage, and an `extra_files` JSON column (multiview views,
references, model files, extra images). Multiview sets and 3D models each get a folder under `data/outputs/…/<day>/<id>/`;
delete removes the folder. The schema migrates in place (`ALTER TABLE` when a column is missing). 3D thumbnails start as
the input image and are replaced by a render the browser uploads the first time the model is opened.

**Vendored code** (`backend/trivexa/vendor/`, see its README). MV-Adapter, BiRefNet, TripoSR, TripoSG, Pixal3D, NAF, MoGe, MeshFlow, DINOv3, SkinTokens, PartField and UniMate inference code is
copied at pinned commits instead of pip-installed, to avoid their heavy unrelated dependencies and
`trust_remote_code`. Local edits are marked `# Trivexa:`. TripoSG's DiT is HunyuanDiT-derived (Tencent Hunyuan
Community License: not licensed in the EU); its FlashVDM-licensed fast decoder is removed. NAF's NATTEN attention
is reimplemented in PyTorch (exact for integer upsampling factors); Pixal3D's image conditioning
(`vendor/pixal3d/image_cond.py`) projects only occupied voxels (`point_index`), identical to upstream's full-grid
projection but without its 7 GB grid at 1536. PartField uses torch's `scatter_reduce` instead of torch_scatter. `mvadapter/utils/mesh_utils/__init__.py` is intentionally
empty so the camera helpers import without nvdiffrast/Triton/CV-CUDA. nvdiffrast is non-commercial licensed.

**Library-version quirks already handled.** transformers 5 moved DINOv3's blocks to `.model.layer` (Pixal3D's
DINOv3 file has 4.x names, remapped on load). transformers 5 flattened `CLIPTextModel` (no `.text_model`; diffusers'
SD1.5 clip-skip path still assumes it) and renamed ViT layers (TripoSR's checkpoint keys are remapped on load).
Current pymeshlab renamed `Percentage` → `PercentageValue`.

**Frontend** (`frontend/src`). `App.tsx` owns state: the active tool (rail), per-tool forms persisted in
localStorage, assets, selection, and the job map from `useJobs` (websocket with auto-reconnect). Panels per tool
(`ImagePanel`, `MultiviewPanel`, `Model3DPanel`, `SegmentPanel`, `RemeshPanel`, `TexturePanel`, `RigPanel`, `AnimatePanel`) share `controls.tsx`. `Viewer` renders images,
3×2 multiview grids, pending jobs, and 3D models via `ModelViewer` (three.js, lazy-loaded). Plain CSS with tokens
in `styles.css`; no CDN assets (the app must work offline).

## Working conventions

- Commit finished work in logical, meaningful commits (several small commits are preferred over one big one),
  ending messages with the Co-Authored-By line.
- Ask before downloading model weights; record new downloads in `trivexa/download.py` with a pinned revision/URL
  and checksum.
