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
./run.sh                                    # build UI if stale, uv sync (+ "texture" extra), serve on 127.0.0.1:8000
cd backend && .venv/bin/uvicorn trivexa.main:app --reload --port 8000   # backend dev
cd frontend && npm run dev                  # UI dev on :5173, proxies /api (incl. websocket) and /files to :8000
cd frontend && npm run build                # type-check (tsc -b) + build; the server serves frontend/dist
cd backend && uv run python -m trivexa.download [t2mv i2mv ig2mv tg2mv esrgan lama meshflow meshflow-config skintokens]  # resumable, SHA-256 checked
cd backend && uv sync --extra texture       # compiles nvdiffrast; needs CUDA_HOME=/usr/local/cuda on PATH
```

- There are no automated tests or linters. Changes have been verified by driving the real API (POST a job, watch
  the `/api/ws` websocket for `job`/`progress` events) and by checking the UI in Chrome. `npm run build` is the
  frontend type check.
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

Downloaded by Trivexa into `data/models/` (via `trivexa.download`):

| Use | Model |
|---|---|
| Multiview | MV-Adapter SDXL `t2mv` (text) and `i2mv` (image) |
| Texture | MV-Adapter SDXL `ig2mv` (image + geometry) and `tg2mv` (prompt + geometry) |
| Upstream texture baker | `RealESRGAN_x2plus.pth`, `big-lama.pt` |
| Remesh | MeshFlow `meshflow/config.yaml` + `model.pth` (gated `facebook/meshflow`, FAIR noncommercial, outputs too) |
| Remesh image guidance | `meshflow/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` (from Meta's DINOv3 form, fetched by hand) |
| Rig | SkinTokens `skintokens/experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt` (VAST-AI, MIT; includes the skin VAE) |

Still unused on disk, candidates for better 3D engines: TRELLIS.2 (`diffusion_models/trellis_2_*`, `vae/trellis_2_*`,
`microsoft/TRELLIS.2-4B`) and Pixal3D. SD 2.x / MV-Adapter SD2.1 variants are deliberately not supported.

## Built so far

Image tool (SDXL + SD 1.5, long-prompt chunking, live x0 previews, cancel) → Multiview tool (MV-Adapter t2mv/i2mv,
6 views at 768 px) → 3D Model tool (TripoSR, orbit viewer with colour/clay/wire, GLB/OBJ) → Remesh tool (MeshFlow,
dense mesh → ~1k-face artist-style mesh, colours transferred as vertex colours) → Texture tool (ig2mv or tg2mv
views baked into a UV texture by MV-Adapter's own pipeline, with Trivexa's lean baker as a fallback) → Rig tool (SkinTokens,
skeleton + skin weights in a skinned GLB; the viewer shows bones and runs a pose test). Gallery, favorites,
"Reuse" settings, and per-asset actions chain the tools (image → Multiview / 3D, multiview set → 3D, model → Remesh / Texture / Rig).
A side-by-side baker benchmark is in `data/benchmarks/texture-2026-10-12/`.

## Architecture

**Request flow.** Each `POST /api/generate/{image,multiview,3d,remesh,texture,rig}` in `main.py` validates input, resolves model
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

**Vendored code** (`backend/trivexa/vendor/`, see its README). MV-Adapter, BiRefNet, TripoSR, MeshFlow, DINOv3 and SkinTokens inference code is
copied at pinned commits instead of pip-installed, to avoid their heavy unrelated dependencies and
`trust_remote_code`. Local edits are marked `# Trivexa:`. `mvadapter/utils/mesh_utils/__init__.py` is intentionally
empty so the camera helpers import without nvdiffrast/Triton/CV-CUDA. nvdiffrast is non-commercial licensed.

**Library-version quirks already handled.** transformers 5 flattened `CLIPTextModel` (no `.text_model`; diffusers'
SD1.5 clip-skip path still assumes it) and renamed ViT layers (TripoSR's checkpoint keys are remapped on load).
Current pymeshlab renamed `Percentage` → `PercentageValue`.

**Frontend** (`frontend/src`). `App.tsx` owns state: the active tool (rail), per-tool forms persisted in
localStorage, assets, selection, and the job map from `useJobs` (websocket with auto-reconnect). Panels per tool
(`ImagePanel`, `MultiviewPanel`, `Model3DPanel`, `RemeshPanel`, `TexturePanel`, `RigPanel`) share `controls.tsx`. `Viewer` renders images,
3×2 multiview grids, pending jobs, and 3D models via `ModelViewer` (three.js, lazy-loaded). Plain CSS with tokens
in `styles.css`; no CDN assets (the app must work offline).

## Working conventions

- Commit finished work in logical, meaningful commits (several small commits are preferred over one big one),
  ending messages with the Co-Authored-By line.
- Ask before downloading model weights; record new downloads in `trivexa/download.py` with a pinned revision/URL
  and checksum.
