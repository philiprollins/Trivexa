"""Bake the six ig2mv views into a UV texture with MV-Adapter's own texture pipeline (vendored).

Compared with the lean baker in texture.py it upscales the views 2x with RealESRGAN, unwraps with UVAtlas
(open3d) and fills areas no camera saw by rendering extra views and inpainting them with LaMa: sharper, much
better on thin or occluded parts, and faster (~5 s), at the cost of heavier dependencies and ~12 GB VRAM peak.
"""

import importlib.util
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

from ..config import DATA_DIR, TEXTURE_WEIGHTS_DIR

log = logging.getLogger(__name__)

ESRGAN = TEXTURE_WEIGHTS_DIR / "RealESRGAN_x2plus.pth"
LAMA = TEXTURE_WEIGHTS_DIR / "big-lama.pt"
_PACKAGES = ("nvdiffrast", "open3d", "pymeshlab", "spandrel", "gltflib", "cvcuda")
# Camera ring as passed by MV-Adapter's scripts/texture_i2tex.py (front, right, back, left, top, bottom).
_AZIMUTHS = [0, 90, 180, 270, 180, 180]


def unavailable_reason() -> str | None:
    """Why this baker can't run here, or None if it can."""
    missing = [m for m in _PACKAGES if importlib.util.find_spec(m) is None]
    if missing:
        return f"missing packages ({', '.join(missing)}); run `uv sync --extra texture`"
    weights = [p.name for p in (ESRGAN, LAMA) if not p.is_file()]
    if weights:
        return f"missing weights ({', '.join(weights)}); run `uv run python -m trivexa.download esrgan lama`"
    return None


def bake(
    mesh_path: Path,
    views: list[Image.Image],
    uv_size: int,
    front_yaw_deg: float = 0.0,
    device: str = "cuda",
) -> tuple[trimesh.Trimesh, Image.Image]:
    """Return the textured mesh (UV-unwrapped copy of the input, original orientation) and its texture.

    The views must come from the ig2mv cameras with the mesh turned by front_yaw_deg about +Y; turning the
    mesh is equivalent to turning the camera ring the other way, so the yaw is folded into the azimuths.
    """
    from ..vendor.mvadapter.pipelines.pipeline_texture import ModProcessConfig, TexturePipeline

    (DATA_DIR / "tmp").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="bake_", dir=DATA_DIR / "tmp"))
    pipe = None
    try:
        local_mesh = work / "mesh.glb"  # the pipeline writes its unwrapped copy next to the input
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        manifold = _split_nonmanifold_edges(mesh)
        if manifold is mesh:
            shutil.copy(mesh_path, local_mesh)
        else:
            manifold.export(local_mesh)
        w, h = views[0].size
        row = Image.new("RGB", (w * len(views), h))
        for i, view in enumerate(views):
            row.paste(view.convert("RGB"), (i * w, 0))
        row.save(work / "views.png")

        pipe = TexturePipeline(upscaler_ckpt_path=str(ESRGAN), inpaint_ckpt_path=str(LAMA), device=device)
        out = pipe(
            mesh_path=str(local_mesh),
            save_dir=str(work),
            save_name="baked",
            uv_unwarp=True,
            preprocess_mesh=False,
            uv_size=uv_size,
            rgb_path=str(work / "views.png"),
            rgb_process_config=ModProcessConfig(view_upscale=True, inpaint_mode="view"),
            camera_azimuth_deg=[a - 90 - front_yaw_deg for a in _AZIMUTHS],
        )
        textured = trimesh.load(out.shaded_model_save_path, force="mesh", process=False)
        texture = textured.visual.material.baseColorTexture.convert("RGB")
        return textured, texture
    finally:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        shutil.rmtree(work, ignore_errors=True)


def _split_nonmanifold_edges(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Detach every face on an edge shared by more than two faces (it gets its own vertices).

    UVAtlas segfaults on non-manifold edges, which MeshFlow remeshes have; detaching keeps the surface as is.
    Returns the mesh itself when there is nothing to fix.
    """
    _, inverse, counts = np.unique(mesh.edges_sorted, axis=0, return_inverse=True, return_counts=True)
    detach = np.zeros(len(mesh.faces), dtype=bool)
    detach[mesh.edges_face[counts[inverse] > 2]] = True
    if not detach.any():
        return mesh
    log.info("Detaching %d faces on non-manifold edges before UV unwrapping", detach.sum())
    faces = mesh.faces.copy()
    faces[detach] = len(mesh.vertices) + np.arange(detach.sum() * 3).reshape(-1, 3)
    vertices = np.concatenate([mesh.vertices, mesh.vertices[mesh.faces[detach]].reshape(-1, 3)])
    out = trimesh.Trimesh(vertices, faces, process=False)
    out.remove_unreferenced_vertices()
    return out
