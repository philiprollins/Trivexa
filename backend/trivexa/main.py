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
from .engines import meshflow as meshflow_engine
from .engines import skintokens
from .engines import texture_upstream
from .engines import partfield, pixal3d, triposg, triposr
from .engines.meshflow import MeshFlowEngine, MeshFlowParams
from .engines.gltf_skin import skinned_glb
from .engines.skintokens import RigParams, SkinTokensEngine
from .engines.partfield import PartFieldEngine, SegmentParams
from .engines.multiview import AZIMUTHS, MultiviewEngine, MVParams, adapter_available, prepare_reference
from .engines.pixal3d import Pixal3DEngine, Pixal3DParams
from .engines.triposg import TripoSGEngine, TripoSGParams
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
tripo_sg = TripoSGEngine()
pixal = Pixal3DEngine()
remesher = MeshFlowEngine()
rigger = SkinTokensEngine()
segmenter = PartFieldEngine()


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
    for engine in (diffusion, multiview, tripo, tripo_sg, pixal, remesher, rigger, segmenter):
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
    if tripo_sg.transformer is not None:
        return "TripoSG"
    if pixal.models is not None:
        return "Pixal3D"
    if remesher.pipe is not None:
        return "MeshFlow"
    if rigger.model is not None:
        return "SkinTokens"
    if segmenter.pvcnn is not None:
        return "PartField"
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
        "model3d": {
            "triposr": triposr.available(),
            "triposg": triposg.available(),
            "pixal3d": PIXAL3D_NOTE is None,
            "pixal3d_note": PIXAL3D_NOTE,
            "meshflow": meshflow_engine.available(),
            "meshflow_image": meshflow_engine.image_available(),
            "skintokens": skintokens.available(),
            "partfield": partfield.available(),
        },
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
    engine: Literal["triposr", "triposg", "pixal3d"] = "triposr"
    # TripoSR
    foreground_ratio: float = Field(0.85, ge=0.5, le=1.0)
    mc_resolution: int = Field(256, ge=64, le=512)
    # TripoSG and Pixal3D (steps and guidance default per engine, see _MODEL3D_DEFAULTS)
    steps: int | None = Field(None, ge=4, le=100)
    guidance: float | None = Field(None, ge=1.0, le=15.0)
    seed: int = Field(-1, ge=-1, le=2**32 - 1)
    # TripoSG: 0 keeps every face; Pixal3D: decimation target after remeshing, 0 is upstream's 1M.
    # ("faces" is the saved mesh's count.)
    face_limit: int = Field(0, ge=0, le=2_000_000)
    octree_depth: Literal[8, 9] = 9  # TripoSG
    # Pixal3D
    resolution: Literal[1024, 1536] = 1536
    fov: float = Field(0, ge=0, le=120)  # horizontal camera FOV in degrees; 0 estimates it with MoGe-2
    texture_size: Literal[1024, 2048, 4096] = 2048


_MODEL3D_FIELDS = {
    "triposr": ("foreground_ratio", "mc_resolution"),
    "triposg": ("steps", "guidance", "octree_depth", "face_limit"),
    "pixal3d": ("steps", "guidance", "face_limit", "resolution", "fov", "texture_size"),
}
_MODEL3D_DEFAULTS = {"triposg": {"steps": 50, "guidance": 7.0}, "pixal3d": {"steps": 12, "guidance": 7.5}}
PIXAL3D_NOTE = pixal3d.unavailable_reason()


@app.post("/api/generate/3d")
async def generate_3d(req: Model3DRequest):
    if req.engine == "triposr" and not triposr.available():
        raise HTTPException(400, "TripoSR weights not found (models/triposr)")
    if req.engine == "triposg" and not triposg.available():
        raise HTTPException(400, "TripoSG weights not found; run: uv run python -m trivexa.download triposg")
    if req.engine == "pixal3d" and PIXAL3D_NOTE:
        raise HTTPException(400, f"Pixal3D unavailable: {PIXAL3D_NOTE}")
    if req.remove_bg and not bg_remover.available():
        raise HTTPException(400, "Background removal weights (birefnet.safetensors) not found")
    source, parent_id = _load_source_image(req.source_asset, req.source_image, req.source_view)
    # "front" records which way the mesh faces; older TripoSR models (before it was set) face -Z.
    params = req.model_dump(include={"source_asset", "source_view", "remove_bg", *_MODEL3D_FIELDS[req.engine]})
    params |= {"engine": req.engine, "front": "+z"}
    for k, v in _MODEL3D_DEFAULTS.get(req.engine, {}).items():
        if params.get(k) is None:
            params[k] = v
    if req.engine in ("triposg", "pixal3d"):
        params["seed"] = random.randint(0, 2**32 - 1) if req.seed == -1 else req.seed
    job = Job(kind="model3d", params=params, run=_model3d_runner(params, source, parent_id))
    return {"jobs": [jobs.submit(job).public()]}


def _model3d_runner(params: dict, source: Image.Image, parent_id: str | None):
    def run(job: Job) -> dict:
        engine = {"triposg": tripo_sg, "pixal3d": pixal}.get(params["engine"], tripo)
        _exclusive(engine)
        image = bg_remover.remove(source) if params["remove_bg"] else source
        on_step = lambda step, total, preview: jobs.report_step(job, step, total, preview)  # noqa: E731
        full = None
        if engine is pixal:
            mesh, processed, info = pixal.generate(
                Pixal3DParams(
                    image=image,
                    seed=params["seed"],
                    resolution=params["resolution"],
                    steps=params["steps"],
                    guidance=params["guidance"],
                    fov=params["fov"] or None,
                    faces=params["face_limit"] or 1_000_000,
                    texture_size=params["texture_size"],
                ),
                on_step=on_step,
                is_cancelled=job.cancel_requested.is_set,
            )
            # "fov" stays the request (0 = estimated); the FOV and cascade resolution actually used are recorded.
            extra = {"textured": True, "fov_used": info["fov"], "resolution_used": info["resolution"]}
            texture = mesh.visual.material.baseColorTexture.convert("RGB")
            return store.save_model3d(mesh, processed, params | extra, parent_id, images={"texture": texture})
        if engine is tripo_sg:
            mesh, processed, full = tripo_sg.generate(
                TripoSGParams(
                    image=image,
                    steps=params["steps"],
                    guidance=params["guidance"],
                    seed=params["seed"],
                    octree_depth=params["octree_depth"],
                    faces=params["face_limit"],
                ),
                on_step=on_step,
                is_cancelled=job.cancel_requested.is_set,
            )
        else:
            mesh, processed = tripo.generate(
                TripoSRParams(image=image, foreground_ratio=params["foreground_ratio"], mc_resolution=params["mc_resolution"]),
                on_step=on_step,
            )
        extra = {"full_faces": len(full.faces)} if full is not None else {}
        return store.save_model3d(mesh, processed, params | extra, parent_id, full_mesh=full)

    return run


class RemeshRequest(BaseModel):
    source_asset: str  # a 3D model asset
    steps: int = Field(meshflow_engine.DEFAULT_STEPS, ge=4, le=100)
    seed: int = Field(-1, ge=-1, le=2**32 - 1)
    count: int = Field(1, ge=1, le=8)
    fill_holes: bool = False
    detail: Literal["low", "normal", "high", "max"] = "normal"  # latent length, see meshflow.DETAIL_LENGTHS
    # Image guidance (DINOv3): the model's own input image by default, or a gallery image / upload.
    use_image: bool = True
    reference_asset: str | None = None
    reference_image: str | None = Field(None, max_length=40_000_000)
    remove_bg: bool = True
    guidance: float = Field(meshflow_engine.DEFAULT_GUIDANCE, ge=1.0, le=10.0)


@app.post("/api/generate/remesh")
async def generate_remesh(req: RemeshRequest):
    source = store.get(req.source_asset)
    if not source or source["kind"] != "model3d":
        raise HTTPException(400, "Choose a 3D model to remesh")
    if not meshflow_engine.available():
        raise HTTPException(400, "MeshFlow weights not found (data/models/meshflow)")
    # MeshFlow samples the surface, so it gets the undecimated mesh when one was kept (TripoSG face limit).
    mesh_url = source["model_urls"].get("glb_full") or source["model_urls"]["glb"]
    mesh_path = OUTPUTS_DIR / mesh_url.removeprefix("/files/")
    input_image = Image.open(OUTPUTS_DIR / source["input_url"].removeprefix("/files/")).copy()
    reference = None
    if req.use_image:
        if not meshflow_engine.image_available():
            raise HTTPException(400, f"DINOv3 weights not found ({meshflow_engine.DINOV3_WEIGHTS.name} in data/models/meshflow)")
        if req.remove_bg and not bg_remover.available():
            raise HTTPException(400, "Background removal weights (birefnet.safetensors) not found")
        if req.reference_asset or req.reference_image:
            reference, _ = _load_source_image(req.reference_asset, req.reference_image)
        else:
            reference = input_image
    image_only = set() if req.use_image else {"reference_asset", "remove_bg", "guidance"}
    sp = source["params"]
    # TripoSR models from before "front" was recorded face -Z: turn the remesh to face +Z like everything newer.
    half_turn = sp.get("front") != "+z" and sp.get("engine") == "triposr"
    base_seed = random.randint(0, 2**32 - 1) if req.seed == -1 else req.seed
    created = []
    for i in range(req.count):
        params = req.model_dump(exclude={"count", "reference_image"} | image_only) | {
            "seed": (base_seed + i) % 2**32,
            "engine": sp.get("engine", "mesh").removesuffix(" + mv-adapter") + " + meshflow",
            "front": "+z",
            "remeshed": True,
            "source_faces": sp.get("full_faces", sp.get("faces")),  # the mesh MeshFlow read (see mesh_url)
        }
        run = _remesh_runner(mesh_path, reference or input_image, reference, half_turn, params, source["id"])
        job = Job(kind="remesh", params=params, run=run)
        created.append(jobs.submit(job).public())
    return {"jobs": created}


def _remesh_runner(
    mesh_path, shown_image: Image.Image, reference_src: Image.Image | None, half_turn: bool, params: dict, parent_id: str
):
    def run(job: Job) -> dict:
        import numpy as np
        import trimesh

        _exclusive(remesher)
        mesh = trimesh.load(mesh_path, force="mesh")
        if half_turn:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))
        reference = None
        if reference_src is not None:
            cutout = bg_remover.remove(reference_src) if params["remove_bg"] else reference_src
            # Same crop as MeshFlow's own preprocessing, which flattens onto white.
            reference = prepare_reference(cutout, 512, background=1.0)
        out = remesher.generate(
            MeshFlowParams(
                mesh=mesh,
                steps=params["steps"],
                seed=params["seed"],
                fill_holes=params["fill_holes"],
                length=meshflow_engine.DETAIL_LENGTHS[params["detail"]],
                image=reference,
                guidance=params.get("guidance", 1.0),
            ),
            on_step=lambda step, total, preview: jobs.report_step(job, step, total, preview),
            is_cancelled=job.cancel_requested.is_set,
        )
        return store.save_model3d(out, shown_image, params, parent_id)

    return run


class RigRequest(BaseModel):
    source_asset: str  # a 3D model asset
    style: Literal["auto", "humanoid"] = "auto"  # see skintokens.STYLES
    seed: int = Field(-1, ge=-1, le=2**32 - 1)
    count: int = Field(1, ge=1, le=8)
    beams: int = Field(skintokens.DEFAULT_BEAMS, ge=1, le=20)
    temperature: float = Field(1.0, ge=0.1, le=2.0)


@app.post("/api/generate/rig")
async def generate_rig(req: RigRequest):
    source = store.get(req.source_asset)
    if not source or source["kind"] != "model3d":
        raise HTTPException(400, "Choose a 3D model to rig")
    if not skintokens.available():
        raise HTTPException(400, "SkinTokens weights not found (data/models/skintokens)")
    mesh_path = OUTPUTS_DIR / source["model_urls"]["glb"].removeprefix("/files/")
    input_image = Image.open(OUTPUTS_DIR / source["input_url"].removeprefix("/files/")).copy()
    sp = source["params"]
    # Older TripoSR models face -Z: turn them to +Z like everything newer (as remeshing does).
    half_turn = sp.get("front") != "+z" and sp.get("engine") == "triposr"
    base_seed = random.randint(0, 2**32 - 1) if req.seed == -1 else req.seed
    created = []
    for i in range(req.count):
        params = req.model_dump(exclude={"count"}) | {
            "seed": (base_seed + i) % 2**32,
            "engine": sp.get("engine", "mesh") + " + skintokens",
            "front": "+z",
            "rigged": True,
            # what the rig was built on, so the panels can still describe the model
            **{k: sp[k] for k in ("textured", "remeshed") if sp.get(k)},
        }
        job = Job(kind="rig", params=params, run=_rig_runner(mesh_path, input_image, half_turn, params, source["id"]))
        created.append(jobs.submit(job).public())
    return {"jobs": created}


def _rig_runner(mesh_path, input_image: Image.Image, half_turn: bool, params: dict, parent_id: str):
    def run(job: Job) -> dict:
        import numpy as np
        import trimesh

        _exclusive(rigger)
        mesh = trimesh.load(mesh_path, force="mesh")
        if half_turn:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))
        rig = rigger.generate(
            RigParams(
                mesh=mesh,
                style=params["style"],
                seed=params["seed"],
                beams=params["beams"],
                temperature=params["temperature"],
            ),
            on_step=lambda step, total, preview: jobs.report_step(job, step, total, preview),
            is_cancelled=job.cancel_requested.is_set,
        )
        glb = skinned_glb(mesh, rig.joints, rig.parents, rig.names, rig.weights)
        return store.save_model3d(mesh, input_image, params | {"bones": len(rig.joints)}, parent_id, glb=glb)

    return run


class SegmentRequest(BaseModel):
    source_asset: str  # a 3D model asset
    parts: int = Field(partfield.DETAIL_PARTS["balanced"], ge=2, le=partfield.MAX_PARTS)
    seed: int = Field(-1, ge=-1, le=2**32 - 1)


@app.post("/api/generate/segment")
async def generate_segment(req: SegmentRequest):
    source = store.get(req.source_asset)
    if not source or source["kind"] != "model3d":
        raise HTTPException(400, "Choose a 3D model to segment")
    if not partfield.available():
        raise HTTPException(400, "PartField weights not found (data/models/partfield)")
    # The mesh the other tools see (TripoSG's decimated one, not model_full.glb): parts keep its UVs and colours.
    mesh_path = OUTPUTS_DIR / source["model_urls"]["glb"].removeprefix("/files/")
    input_image = Image.open(OUTPUTS_DIR / source["input_url"].removeprefix("/files/")).copy()
    sp = source["params"]
    # Older TripoSR models face -Z: turn them to +Z like everything newer (as remeshing does).
    half_turn = sp.get("front") != "+z" and sp.get("engine") == "triposr"
    params = req.model_dump() | {
        "seed": random.randint(0, 2**32 - 1) if req.seed == -1 else req.seed,
        "engine": sp.get("engine", "mesh") + " + partfield",
        "front": "+z",
        "segmented": True,
        # a rig does not survive (parts are plain meshes); what it was built on still describes the model
        **{k: sp[k] for k in ("textured", "remeshed") if sp.get(k)},
    }
    job = Job(kind="segment", params=params, run=_segment_runner(mesh_path, input_image, half_turn, params, source["id"]))
    return {"jobs": [jobs.submit(job).public()]}


def _segment_runner(mesh_path, input_image: Image.Image, half_turn: bool, params: dict, parent_id: str):
    def run(job: Job) -> dict:
        import numpy as np
        import trimesh

        _exclusive(segmenter)
        mesh = trimesh.load(mesh_path, force="mesh")
        if half_turn:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))
        labels = segmenter.generate(
            SegmentParams(mesh=mesh, parts=params["parts"], seed=params["seed"]),
            on_step=lambda step, total, preview: jobs.report_step(job, step, total, preview),
            is_cancelled=job.cancel_requested.is_set,
        )
        scene = partfield.split_parts(mesh, labels)
        return store.save_model3d(scene, input_image, params | {"parts": len(scene.geometry)}, parent_id)

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
        "engine": source["params"].get("engine", "mesh").removesuffix(" + mv-adapter") + " + mv-adapter",
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
