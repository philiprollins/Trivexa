# Trivexa

A local image (and, soon, 3D) generation studio. Runs entirely on this machine: a FastAPI server does inference with PyTorch/diffusers and serves a React UI.

## Run

```bash
./run.sh            # → http://127.0.0.1:8000
```

`run.sh` builds the UI if needed, syncs the Python env (uv), and starts the server. `TRIVEXA_HOST` / `TRIVEXA_PORT` override the bind address.

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
