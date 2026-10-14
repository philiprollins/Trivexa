import os

# Long-running server alternating between large pipelines: avoid fragmentation-induced OOMs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import base64  # noqa: E402
import binascii  # noqa: E402
import io  # noqa: E402
import logging  # noqa: E402
import random
from contextlib import asynccontextmanager
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from . import models
from .config import CHECKPOINTS_DIR, FRONTEND_DIST, OUTPUTS_DIR, SETTINGS, SETTINGS_PATH, ensure_dirs
from .engines.bg_removal import BackgroundRemover
from .engines.diffusion import ARCHS, SAMPLERS, DiffusionEngine, GenParams
from .engines import texture as texturing
from .engines import texture_upstream
from .engines import triposr
from .engines.multiview import AZIMUTHS, MultiviewEngine, MVParams, adapter_available, prepare_reference
from .engines.triposr import TripoSREngine, TripoSRParams
from .jobs import Job, JobManager
from .store import Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
torch.backends.cuda.matmul.allow_tf32 = True

ensure_dirs()
store = Store()
jobs = JobManager()
diffusion = DiffusionEngine()
multiview = MultiviewEngine()
bg_remover = BackgroundRemover()
tripo = TripoSREngine()


def _texture_baker() -> tuple[str, str | None]:
    """The baker to use (settings.toml [texture] baker) and, if it had to fall back, why."""
    wanted = SETTINGS["texture"].get("baker", "upstream")
    if wanted == "upstream":
        reason = texture_upstream.unavailable_reason()
        return ("lean", f"upstream baker unavailable: {reason}") if reason else ("upstream", None)
    if wanted != "lean":
        return "lean", f"unknown baker {wanted!r} in {SETTINGS_PATH.name}"
    return "lean", None


TEXTURE_BAKER, TEXTURE_BAKER_NOTE = _texture_baker()
if TEXTURE_BAKER_NOTE:
    logging.getLogger(__name__).warning("Texture baker: using lean (%s)", TEXTURE_BAKER_NOTE)


def _exclusive(keep) -> None:
    """Only one large pipeline on the GPU at a time: unload every engine except `keep`."""
    for engine in (diffusion, multiview, tripo):
        if engine is not keep:
            engine.unload()


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
        "loaded_model": _loaded_label(),
        "pending_jobs": jobs.pending,
        "models_dir": str(CHECKPOINTS_DIR.parent),
        "outputs_dir": str(OUTPUTS_DIR),
    }


def _loaded_label() -> str | None:
    if diffusion.model_path:
        return diffusion.model_path.stem
    if multiview.model_path:
        return f"{multiview.model_path.stem} · MV-Adapter {multiview.mode}"
    return None


@app.post("/api/unload")
def unload():
    with jobs.gpu_lock:
        _exclusive(None)
    return {"ok": True}


@app.get("/api/models")
def list_models():
    return {
        "checkpoints": [c.to_dict() for c in models.list_checkpoints()],
        "archs": [{"id": a.id, "label": a.label} for a in ARCHS.values()],
        "multiview": {
            "t2mv": adapter_available("t2mv"),
            "i2mv": adapter_available("i2mv"),
            "ig2mv": adapter_available("ig2mv"),
            "tg2mv": adapter_available("tg2mv"),
            "bg_removal": bg_remover.available(),
        },
        "model3d": {"triposr": triposr.available()},
        "texture": {"baker": TEXTURE_BAKER, "note": TEXTURE_BAKER_NOTE},
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
        _exclusive(diffusion)
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


class MultiviewGenRequest(BaseModel):
    mode: Literal["t2mv", "i2mv"]
    prompt: str = Field("", max_length=10_000)
    negative: str = Field("watermark, ugly, deformed, noisy, blurry, low contrast", max_length=10_000)
    model: str
    steps: int = Field(40, ge=1, le=100)
    cfg: float = Field(7.0, ge=1.0, le=20.0)
    sampler: str = "euler"
    seed: int = Field(-1, ge=-1, le=2**32 - 1)
    count: int = Field(1, ge=1, le=8)
    # i2mv: the reference is either an existing image asset or an uploaded image (data URL).
    reference_asset: str | None = None
    reference_image: str | None = Field(None, max_length=40_000_000)
    remove_bg: bool = True
    reference_scale: float = Field(1.0, ge=0.0, le=2.0)


def _decode_data_url(data_url: str) -> Image.Image:
    try:
        _, b64 = data_url.split(",", 1)
        image = Image.open(io.BytesIO(base64.b64decode(b64)))
        image.load()
    except (ValueError, binascii.Error, UnidentifiedImageError, OSError) as e:
        raise HTTPException(400, "Could not read the uploaded image") from e
    return image


def _load_source_image(asset_id: str | None, data_url: str | None, view: int = 0) -> tuple[Image.Image, str | None]:
    """An input image from a gallery asset (an image, or one view of a multiview set) or an uploaded data URL."""
    if asset_id:
        asset = store.get(asset_id)
        if not asset or asset["kind"] not in ("image", "multiview"):
            raise HTTPException(400, "Source asset not found")
        if asset["kind"] == "multiview":
            match = next((v for v in asset["views"] if v["azimuth"] == view), None)
            if not match:
                raise HTTPException(400, f"That multiview set has no {view}° view")
            path = OUTPUTS_DIR / match["url"].removeprefix("/files/")
        else:
            path = OUTPUTS_DIR / asset["file"]
        return Image.open(path).copy(), asset["id"]
    if data_url:
        return _decode_data_url(data_url), None
    raise HTTPException(400, "An input image is required")


def _load_reference(req: MultiviewGenRequest) -> tuple[Image.Image, str | None]:
    if req.reference_asset and (a := store.get(req.reference_asset)) and a["kind"] != "image":
        raise HTTPException(400, "Reference asset not found")
    return _load_source_image(req.reference_asset, req.reference_image)


@app.post("/api/generate/multiview")
async def generate_multiview(req: MultiviewGenRequest):
    try:
        model_path, _ = models.resolve_checkpoint(req.model, {"sdxl"})
    except ValueError as e:
        raise HTTPException(400, f"{e} (MV-Adapter needs an SDXL checkpoint)") from e
    if req.sampler not in SAMPLERS:
        raise HTTPException(400, f"Unknown sampler: {req.sampler}")
    if not adapter_available(req.mode):
        raise HTTPException(400, f"MV-Adapter weights for {req.mode} are not downloaded")
    source, parent_id = None, None
    if req.mode == "i2mv":
        source, parent_id = _load_reference(req)
        if req.remove_bg and not bg_remover.available():
            raise HTTPException(400, "Background removal weights (birefnet.safetensors) not found")
    elif not req.prompt.strip():
        raise HTTPException(400, "Text-to-multiview needs a prompt")

    base_seed = random.randint(0, 2**32 - 1) if req.seed == -1 else req.seed
    created = []
    for i in range(req.count):
        params = req.model_dump(exclude={"count", "reference_image"}) | {
            "seed": (base_seed + i) % 2**32,
            "model_name": model_path.stem,
            "sampler_label": SAMPLERS[req.sampler][0],
            "arch": "sdxl",
            "azimuths": AZIMUTHS,
        }
        if req.mode == "t2mv":
            for k in ("reference_asset", "remove_bg", "reference_scale"):
                params.pop(k)
        job = Job(kind="multiview", params=params, run=_multiview_runner(model_path, params, source, parent_id))
        created.append(jobs.submit(job).public())
    return {"jobs": created}


def _multiview_runner(model_path, params: dict, source: Image.Image | None, parent_id: str | None):
    def run(job: Job) -> dict:
        _exclusive(multiview)
        reference = None
        if params["mode"] == "i2mv":
            image = bg_remover.remove(source) if params["remove_bg"] else source
            reference = prepare_reference(image)
        views = multiview.generate(
            MVParams(
                mode=params["mode"],
                model_path=model_path,
                prompt=params["prompt"] or "high quality",
                negative=params["negative"],
                steps=params["steps"],
                cfg=params["cfg"],
                sampler=params["sampler"],
                seed=params["seed"],
                reference=reference,
                reference_scale=params.get("reference_scale", 1.0),
            ),
            on_step=lambda step, total, preview: jobs.report_step(job, step, total, preview),
            is_cancelled=job.cancel_requested.is_set,
        )
        return store.save_multiview(views, AZIMUTHS, params, reference=reference, parent_id=parent_id)

    return run


class Model3DRequest(BaseModel):
    source_asset: str | None = None
    source_view: int = 0  # azimuth, when the source is a multiview set
    source_image: str | None = Field(None, max_length=40_000_000)
    remove_bg: bool = True
    foreground_ratio: float = Field(0.85, ge=0.5, le=1.0)
    mc_resolution: int = Field(256, ge=64, le=512)


@app.post("/api/generate/3d")
async def generate_3d(req: Model3DRequest):
    if not triposr.available():
        raise HTTPException(400, "TripoSR weights not found (models/triposr)")
    if req.remove_bg and not bg_remover.available():
        raise HTTPException(400, "Background removal weights (birefnet.safetensors) not found")
    source, parent_id = _load_source_image(req.source_asset, req.source_image, req.source_view)
    # "front" records which way the mesh faces; older TripoSR models (before it was set) face -Z.
    params = req.model_dump(exclude={"source_image"}) | {"engine": "triposr", "front": "+z"}
    job = Job(kind="model3d", params=params, run=_model3d_runner(params, source, parent_id))
    return {"jobs": [jobs.submit(job).public()]}


def _model3d_runner(params: dict, source: Image.Image, parent_id: str | None):
    def run(job: Job) -> dict:
        _exclusive(tripo)
        image = bg_remover.remove(source) if params["remove_bg"] else source
        mesh, processed = tripo.generate(
            TripoSRParams(image=image, foreground_ratio=params["foreground_ratio"], mc_resolution=params["mc_resolution"]),
            on_step=lambda step, total, preview: jobs.report_step(job, step, total, preview),
        )
        return store.save_model3d(mesh, processed, params, parent_id)

    return run


class TextureRequest(BaseModel):
    source_asset: str  # a 3D model asset
    model: str  # SDXL checkpoint for MV-Adapter
    mode: Literal["ig2mv", "tg2mv"] = "ig2mv"  # guided by a reference image, or by the prompt alone
    # ig2mv reference image: the model's own input image by default, or a gallery image / upload.
    reference_asset: str | None = None
    reference_image: str | None = Field(None, max_length=40_000_000)
    remove_bg: bool = True
    prompt: str = Field("high quality", max_length=10_000)
    negative: str = Field("watermark, ugly, deformed, noisy, blurry, low contrast", max_length=10_000)
    steps: int = Field(40, ge=1, le=100)
    cfg: float = Field(3.0, ge=1.0, le=20.0)
    sampler: str = "euler"
    seed: int = Field(-1, ge=-1, le=2**32 - 1)
    reference_scale: float = Field(1.0, ge=0.0, le=2.0)
    front_yaw: Literal[0, 90, 180, 270] = 0
    uv_size: Literal[1024, 2048, 4096] = 2048


@app.post("/api/generate/texture")
async def generate_texture(req: TextureRequest):
    source = store.get(req.source_asset)
    if not source or source["kind"] != "model3d":
        raise HTTPException(400, "Choose a 3D model to texture")
    try:
        model_path, _ = models.resolve_checkpoint(req.model, {"sdxl"})
    except ValueError as e:
        raise HTTPException(400, f"{e} (MV-Adapter needs an SDXL checkpoint)") from e
    if req.sampler not in SAMPLERS:
        raise HTTPException(400, f"Unknown sampler: {req.sampler}")
    if not adapter_available(req.mode):
        raise HTTPException(400, f"MV-Adapter {req.mode} weights are not downloaded")
    reference = None
    if req.mode == "tg2mv":
        if not req.prompt.strip():
            raise HTTPException(400, "Describe the texture: the prompt is all tg2mv has to go on")
    else:
        if req.remove_bg and not bg_remover.available():
            raise HTTPException(400, "Background removal weights (birefnet.safetensors) not found")
        if req.reference_asset or req.reference_image:
            reference, _ = _load_source_image(req.reference_asset, req.reference_image)
        else:
            reference = Image.open(OUTPUTS_DIR / source["input_url"].removeprefix("/files/")).copy()
    mesh_path = OUTPUTS_DIR / source["model_urls"]["glb"].removeprefix("/files/")

    image_only = {"reference_asset", "remove_bg", "reference_scale"} if req.mode == "tg2mv" else set()
    params = req.model_dump(exclude={"reference_image"} | image_only) | {
        "seed": random.randint(0, 2**32 - 1) if req.seed == -1 else req.seed,
        "model_name": model_path.stem,
        "sampler_label": SAMPLERS[req.sampler][0],
        "engine": source["params"].get("engine", "mesh") + " + mv-adapter",
        "front": "+z",
        "baker": TEXTURE_BAKER,
    }
    job = Job(kind="texture", params=params, run=_texture_runner(model_path, mesh_path, reference, params, source["id"]))
    return {"jobs": [jobs.submit(job).public()]}


def _texture_runner(model_path, mesh_path, reference_src: Image.Image | None, params: dict, parent_id: str):
    def run(job: Job) -> dict:
        import trimesh

        _exclusive(multiview)
        lean = params["baker"] == "lean"
        mesh = trimesh.load(mesh_path, force="mesh")
        # The lean baker needs our UV unwrap (~10 s); the upstream one unwraps itself.
        prepared = texturing.prepare_mesh(mesh, front_yaw_deg=params["front_yaw"], unwrap=lean)
        raster = texturing.Rasterizer(prepared)  # GL context lives on this worker thread
        try:
            pos_maps, nrm_maps = raster.render_views(768)
            reference = None
            if reference_src is not None:
                reference = prepare_reference(bg_remover.remove(reference_src) if params["remove_bg"] else reference_src)
            views = multiview.generate(
                MVParams(
                    mode=params["mode"],
                    model_path=model_path,
                    prompt=params["prompt"] or "high quality",
                    negative=params["negative"],
                    steps=params["steps"],
                    cfg=params["cfg"],
                    sampler=params["sampler"],
                    seed=params["seed"],
                    reference=reference,
                    reference_scale=params.get("reference_scale", 1.0),
                    control_images=texturing.condition_images(pos_maps, nrm_maps),
                ),
                on_step=lambda step, total, preview: jobs.report_step(job, step, total, preview),
                is_cancelled=job.cancel_requested.is_set,
            )
            if lean:
                texture = texturing.bake(prepared, raster, views, pos_maps, uv_size=params["uv_size"])
                out = texturing.textured_mesh(prepared, texture)
        finally:
            raster.release()
        if not lean:
            multiview.unload()  # the upstream baker peaks ~12 GB; it can't share the GPU with the MV pipeline
            out, texture = texture_upstream.bake(mesh_path, views, params["uv_size"], params["front_yaw"])

        grid = Image.new("RGB", (768 * 3, 768 * 2))
        for i, view in enumerate(views):
            grid.paste(view, ((i % 3) * 768, (i // 3) * 768))
        params_out = params | {"vertices": len(out.vertices), "faces": len(out.faces), "textured": True}
        # Without a reference image (tg2mv), the generated front view stands in as the model's input image.
        return store.save_model3d(
            out, reference or views[0], params_out, parent_id, images={"texture": texture, "views": grid}
        )

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


class ThumbnailUpload(BaseModel):
    image: str = Field(max_length=10_000_000)  # PNG/WebP data URL rendered by the 3D viewer


@app.post("/api/assets/{asset_id}/thumbnail")
def set_thumbnail(asset_id: str, body: ThumbnailUpload):
    asset = store.set_thumbnail(asset_id, _decode_data_url(body.image))
    if not asset:
        raise HTTPException(404, "Asset not found")
    return asset


@app.get("/api/assets/{asset_id}/obj.zip")
def download_obj_zip(asset_id: str):
    """OBJ with its .mtl and texture in one archive, so it opens textured in Blender etc."""
    import zipfile

    asset = store.get(asset_id)
    if not asset or asset["kind"] != "model3d":
        raise HTTPException(404, "3D model not found")
    folder = (OUTPUTS_DIR / asset["model_urls"]["obj"].removeprefix("/files/")).parent
    buf = io.BytesIO()
    # The OBJ, its .mtl files, and whatever textures they reference (names depend on the material).
    files = {folder / "model.obj"}
    for mtl in folder.glob("*.mtl"):
        files.add(mtl)
        for line in mtl.read_text().splitlines():
            if line.strip().startswith("map_") and (tex := folder / line.split()[-1]).is_file():
                files.add(tex)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(files):
            zf.write(f, f.name)
    name = f"trivexa_{folder.name}.zip"
    return Response(buf.getvalue(), media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{name}"'})


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
