# Trivexa

A local image, multi-view and 3D generation studio. Runs entirely on this machine: a FastAPI server does inference with PyTorch/diffusers and serves a React UI.

## Tools

- **Image**: text-to-image with SDXL and SD 1.5 checkpoints.
- **Multiview**: six consistent 768 px views (front, 45°, right, back, left, 315°) of one subject using
  [MV-Adapter](https://github.com/huanngzh/MV-Adapter) on an SDXL checkpoint, either from text or from a
  reference image (background removed with BiRefNet). Any gallery image has a *Multiview* button.
- **3D Model**: single-image reconstruction with [TripoSR](https://github.com/VAST-AI-Research/TripoSR)
  (~2 s per mesh, vertex colours) from an upload, a gallery image or any view of a multiview set. Models open
  in an orbit viewer (colour / clay / wireframe) and download as GLB or OBJ. Rough, but fine for prototypes
  and small props.
- **Texture**: gives a 3D model a proper UV texture from a reference image (by default the image it was made from):
  MV-Adapter's geometry-guided `ig2mv` generates six views (incl. top/bottom) conditioned on the mesh's position
  and normal maps, which are baked into a 1K–4K texture (~75 s). Output: textured GLB, OBJ zip (OBJ + MTL + PNG)
  and the texture PNG. Two bakers, chosen in `settings.toml`: MV-Adapter's own pipeline (default) or Trivexa's
  lean fallback — see below.

## Run

```bash
./run.sh            # → http://127.0.0.1:8000
```

`run.sh` builds the UI if needed, syncs the Python env (uv), and starts the server. `TRIVEXA_HOST` / `TRIVEXA_PORT` override the bind address.

### Extra weights

Multiview needs the MV-Adapter SDXL weights (t2mv 1.96 GB, i2mv 3.6 GB) and Texture needs ig2mv (3.6 GB;
tg2mv is optional, for prompt-only texturing later). Download or resume them with:

```bash
cd backend && uv run python -m trivexa.download        # or: ... download t2mv i2mv ig2mv tg2mv
```

Files are pinned to a specific revision and checked against their published SHA-256. `esrgan` and `lama`
(RealESRGAN x2 and LaMa, 261 MB) are the upstream texture baker's view upscaler and inpainter.

### Settings

`settings.toml` (repo root; `TRIVEXA_SETTINGS` points elsewhere) holds options that need a restart:

- `[texture] baker = "upstream" | "lean"` — MV-Adapter's texture pipeline (UVAtlas, RealESRGAN, LaMa view
  inpainting; best quality, ~5 s, ~12 GB VRAM while baking) or Trivexa's lean baker (xatlas + OpenGL projection;
  ~15–25 s, < 3 GB, no extra dependencies). "upstream" needs the `texture` extra (`uv sync --extra texture`,
  which compiles nvdiffrast and so needs the CUDA toolkit; `run.sh` does this) and the `esrgan`/`lama` weights;
  if either is missing Trivexa falls back to "lean" and the Texture panel says why. A side-by-side benchmark
  of both is in `data/benchmarks/texture-2026-10-12/` (local only).
Background removal reads `background_removal/birefnet.safetensors` from the models folder, and TripoSR reads
`triposr/` (`config.yaml`, `model.ckpt`, `dino/config.json` from `stabilityai/TripoSR`).

### Dev mode (hot reload)

```bash
cd backend && .venv/bin/uvicorn trivexa.main:app --reload --port 8000
cd frontend && npm run dev     # → http://localhost:5173 (proxies /api and /files to :8000)
```

## Where things live

| What | Where |
|---|---|
| Model files (read-only) | `~/ComfyUI/models` — override with `TRIVEXA_MODELS_DIR` |
| Generated images | `data/outputs/images/YYYY-MM-DD/*.png` (settings embedded as PNG text chunks) |
| Multi-view sets | `data/outputs/multiview/YYYY-MM-DD/<id>/` (`view_000.png` … `view_315.png`, `grid.png`, `reference.png`) |
| 3D models | `data/outputs/models3d/YYYY-MM-DD/<id>/` (`model.glb`, `model.obj`, `input.png`; textured models also `material.mtl`, `texture.png`, `views.png`) |
| Downloaded weights | `data/models/` (MV-Adapter) |
| Thumbnails | `data/outputs/thumbs/` |
| Asset database | `data/trivexa.db` (SQLite) |
| Model configs/tokenizers | `data/hf-configs/` (fetched once per architecture on first load, a few MB, no weights) |

`TRIVEXA_DATA_DIR` moves the whole `data/` folder.

## Layout

```
backend/trivexa/
  main.py          API routes, websocket, static serving
  jobs.py          single-GPU job queue + progress broadcast
  store.py         saving images/thumbnails + SQLite records
  models.py        checkpoint discovery, arch detection from safetensors headers
  engines/diffusion.py  SDXL + SD 1.5 pipelines (single-file checkpoints, long-prompt chunking, live previews)
  engines/multiview.py  MV-Adapter text/image-to-multiview on an SDXL checkpoint
  engines/bg_removal.py BiRefNet foreground extraction
  engines/triposr.py    TripoSR image-to-mesh
  engines/texture.py    ig2mv condition maps (headless OpenGL via moderngl/EGL) + the lean baker (xatlas, projection)
  engines/texture_upstream.py  MV-Adapter's texture pipeline (vendored) as the default baker
  download.py      resumable, checksum-verified weight downloads
  vendor/          trimmed third-party inference code (MV-Adapter, BiRefNet, TripoSR); see vendor/README.md
frontend/src/      React UI (Vite)
```

## Notes

- SDXL (incl. Illustrious/Pony derivatives) and SD 1.5 checkpoints are supported. Architecture is detected from the
  safetensors header; inpainting variants, SD 2.x and non-image checkpoints are hidden. One model is kept in VRAM
  at a time, so switching models reloads (~3s SD 1.5, ~15s SDXL).
- CLIP skip 1 is each architecture's standard layer (last for SD 1.5, penultimate for SDXL); 2 matches A1111's
  "CLIP skip 2" on SD 1.5.
- Texturing: meshes are expected to face +Z (glTF front);
  the Texture panel can turn a model that faces another way. TripoSR models made before this was fixed face -Z
  and default to a half turn.
- Prompts longer than 77 tokens are encoded in chunks rather than truncated. `(word:1.2)` weighting syntax is not parsed yet.
- The SDXL VAE decodes in bf16 to avoid the fp16 NaN issue without the fp32 memory spike.
- Only one large pipeline (image, multiview or 3D) is on the GPU at a time; switching tools reloads
  (~8 s for MV-Adapter, ~3 s for TripoSR).
  Image-to-multiview peaks around 14.4 GB of VRAM (the text encoders are parked in system RAM while it denoises),
  so it is tight on a 16 GB card if anything else is using the GPU.
