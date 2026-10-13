import logging
import random
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import models
from .config import CHECKPOINTS_DIR, FRONTEND_DIST, OUTPUTS_DIR, ensure_dirs
from .engines.diffusion import ARCHS, SAMPLERS, DiffusionEngine, GenParams
from .jobs import Job, JobManager
from .store import Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
torch.backends.cuda.matmul.allow_tf32 = True

ensure_dirs()
store = Store()
jobs = JobManager()
diffusion = DiffusionEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    jobs.start()
    yield
    await jobs.stop()


app = FastAPI(title="Trivexa", lifespan=lifespan)


# -- system ---------------------------------------------------------------------


@app.get("/api/status")
def status():
    gpu = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        gpu = {"name": torch.cuda.get_device_name(0), "vram_free": free, "vram_total": total}
    return {
        "gpu": gpu,
        "loaded_model": diffusion.model_path.name if diffusion.model_path else None,
        "loaded_arch": diffusion.arch.id if diffusion.arch else None,
        "pending_jobs": jobs.pending,
        "models_dir": str(CHECKPOINTS_DIR.parent),
        "outputs_dir": str(OUTPUTS_DIR),
    }


@app.post("/api/unload")
def unload():
    with jobs.gpu_lock:
        diffusion.unload()
    return {"ok": True}


@app.get("/api/models")
def list_models():
    return {
        "checkpoints": [c.to_dict() for c in models.list_checkpoints()],
        "archs": [{"id": a.id, "label": a.label} for a in ARCHS.values()],
    }


@app.get("/api/samplers")
def list_samplers():
    return {"samplers": [{"id": k, "label": v[0]} for k, v in SAMPLERS.items()]}


# -- generation -----------------------------------------------------------------


class ImageGenRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10_000)
    negative: str = Field("", max_length=10_000)
    model: str
    width: int = Field(1024, ge=256, le=2048, multiple_of=8)
    height: int = Field(1024, ge=256, le=2048, multiple_of=8)
    steps: int = Field(28, ge=1, le=150)
    cfg: float = Field(6.0, ge=1.0, le=30.0)
    sampler: str = "euler_a"
    seed: int = Field(-1, ge=-1, le=2**32 - 1)
    clip_skip: int = Field(1, ge=1, le=4)
    count: int = Field(1, ge=1, le=16)


@app.post("/api/generate/image")
async def generate_image(req: ImageGenRequest):
    try:
        model_path, arch = models.resolve_checkpoint(req.model, set(ARCHS))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if req.sampler not in SAMPLERS:
        raise HTTPException(400, f"Unknown sampler: {req.sampler}")

    base_seed = random.randint(0, 2**32 - 1) if req.seed == -1 else req.seed
    created = []
    # Each image is its own job with its own seed, so any single result can be reproduced exactly.
    for i in range(req.count):
        seed = (base_seed + i) % 2**32
        params = req.model_dump(exclude={"count"}) | {
            "seed": seed,
            "model_name": model_path.stem,
            "sampler_label": SAMPLERS[req.sampler][0],
            "arch": arch,
        }
        created.append(jobs.submit(Job(kind="image", params=params, run=_image_runner(model_path, params))).public())
    return {"jobs": created}


def _image_runner(model_path, params: dict):
    def run(job: Job) -> dict:
        image = diffusion.generate(
            GenParams(
                arch=params["arch"],
                model_path=model_path,
                prompt=params["prompt"],
                negative=params["negative"],
                width=params["width"],
                height=params["height"],
                steps=params["steps"],
                cfg=params["cfg"],
                sampler=params["sampler"],
                seed=params["seed"],
                clip_skip=params["clip_skip"],
            ),
            on_step=lambda step, total, preview: jobs.report_step(job, step, total, preview),
            is_cancelled=job.cancel_requested.is_set,
        )
        return store.save_image(image, params)

    return run


@app.get("/api/jobs")
def list_jobs():
    return {"jobs": jobs.list()}


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    if not jobs.cancel(job_id):
        raise HTTPException(404, "No active job with that id")
    return {"ok": True}


# -- assets ---------------------------------------------------------------------


@app.get("/api/assets")
def list_assets(kind: str | None = None, favorite: bool | None = None, limit: int = 100, offset: int = 0):
    return {"assets": store.list(kind=kind, favorite=favorite, limit=min(limit, 500), offset=offset)}


@app.get("/api/assets/{asset_id}")
def get_asset(asset_id: str):
    asset = store.get(asset_id)
    if not asset:
        raise HTTPException(404, "Asset not found")
    return asset


class AssetPatch(BaseModel):
    favorite: bool


@app.patch("/api/assets/{asset_id}")
def patch_asset(asset_id: str, patch: AssetPatch):
    asset = store.set_favorite(asset_id, patch.favorite)
    if not asset:
        raise HTTPException(404, "Asset not found")
    return asset


@app.delete("/api/assets/{asset_id}")
def delete_asset(asset_id: str):
    if not store.delete(asset_id):
        raise HTTPException(404, "Asset not found")
    return {"ok": True}


# -- realtime -------------------------------------------------------------------


@app.websocket("/api/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    jobs.clients.add(websocket)
    try:
        await websocket.send_json({"type": "jobs", "data": jobs.list()})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        jobs.clients.discard(websocket)


# -- static ---------------------------------------------------------------------

app.mount("/files", StaticFiles(directory=OUTPUTS_DIR), name="files")
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
