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

## Run

```bash
./run.sh            # → http://127.0.0.1:8000
```

`run.sh` builds the UI if needed, syncs the Python env (uv), and starts the server. `TRIVEXA_HOST` / `TRIVEXA_PORT` override the bind address.

### Extra weights

Multiview needs the MV-Adapter SDXL weights (t2mv 1.96 GB, i2mv 3.6 GB). Download or resume them with:

```bash
cd backend && uv run python -m trivexa.download        # or: ... download t2mv / i2mv
```

Files are pinned to a specific revision and checked against their published SHA-256.
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
| 3D models | `data/outputs/models3d/YYYY-MM-DD/<id>/` (`model.glb`, `model.obj`, `input.png`) |
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
- Prompts longer than 77 tokens are encoded in chunks rather than truncated. `(word:1.2)` weighting syntax is not parsed yet.
- The SDXL VAE decodes in bf16 to avoid the fp16 NaN issue without the fp32 memory spike.
- Only one large pipeline (image, multiview or 3D) is on the GPU at a time; switching tools reloads
  (~8 s for MV-Adapter, ~3 s for TripoSR).
  Image-to-multiview peaks around 14.4 GB of VRAM (the text encoders are parked in system RAM while it denoises),
  so it is tight on a 16 GB card if anything else is using the GPU.
