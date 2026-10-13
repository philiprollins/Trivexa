"""Texture an existing mesh from multi-view images (MV-Adapter ig2mv), baked into a UV texture.

Also renders the ig2mv condition maps for both bakers. The "lean" baker here is the fallback for MV-Adapter's
own texture pipeline (texture_upstream.py), which needs nvdiffrast, CV-CUDA, open3d, pymeshlab and LaMa/RealESRGAN:
rasterisation is OpenGL via moderngl on a headless EGL context, UVs come from xatlas, and texels no view can see
fall back to the mesh's own vertex colours (TripoSR provides them) or to inpainting.

Coordinate conventions follow MV-Adapter's ig2mv inference: the mesh is Y-up with its front facing +Z (glTF),
rescaled so its largest coordinate is 0.5, and rendered by six orthographic cameras
(front, right, back, left at elevation 0; top and bottom).
"""

import logging
from dataclasses import dataclass

import cv2
import moderngl
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import xatlas
from PIL import Image

from ..vendor.mvadapter.utils.mesh_utils.camera import get_orthogonal_camera

log = logging.getLogger(__name__)

VIEW_NAMES = ["front", "right", "back", "left", "top", "bottom"]
AZIMUTHS = [0, 90, 180, 270, 180, 180]
ELEVATIONS = [0, 0, 0, 0, 89.99, -89.99]
# ig2mv mesh space -> loader's "standard" space (up +y, front +x in mesh coords; see mvadapter load_mesh).
_MESH_TO_STD = np.linalg.inv(np.stack([[1, 0, 0], [0, 0, -1], [0, 1, 0]], axis=0).T)

_VERT = """
#version 330
uniform mat4 mvp;
in vec3 in_pos;
in vec3 in_nrm;
in vec3 in_col;
in vec2 in_uv;
uniform bool uv_space;
out vec3 v_pos;
out vec3 v_nrm;
out vec3 v_col;
void main() {
    v_pos = in_pos;
    v_nrm = in_nrm;
    v_col = in_col;
    gl_Position = uv_space ? vec4(in_uv * 2.0 - 1.0, 0.0, 1.0) : mvp * vec4(in_pos, 1.0);
}
"""
_FRAG = """
#version 330
in vec3 v_pos;
in vec3 v_nrm;
in vec3 v_col;
layout(location = 0) out vec4 o_pos;
layout(location = 1) out vec4 o_nrm;
layout(location = 2) out vec4 o_col;
void main() {
    o_pos = vec4(v_pos, 1.0);
    o_nrm = vec4(normalize(v_nrm), 1.0);
    o_col = vec4(v_col, 1.0);
}
"""


@dataclass
class PreparedMesh:
    """The mesh in MV-Adapter's normalised space, split along UV seams (one vertex per UV)."""

    original: trimesh.Trimesh
    vmapping: np.ndarray  # new vertex -> original vertex
    faces: np.ndarray
    uvs: np.ndarray
    pos: np.ndarray  # normalised std-space positions per new vertex
    nrm: np.ndarray
    col: np.ndarray | None  # vertex colours in [0, 1], if the mesh had them


def cameras(device: str = "cpu"):
    return get_orthogonal_camera(
        elevation_deg=ELEVATIONS,
        distance=[1.8] * 6,
        left=-0.55,
        right=0.55,
        bottom=-0.55,
        top=0.55,
        azimuth_deg=[a - 90 for a in AZIMUTHS],
        device=device,
    )


def prepare_mesh(mesh: trimesh.Trimesh, front_yaw_deg: float = 0.0, unwrap: bool = True) -> PreparedMesh:
    """Normalise like mvadapter.load_mesh(rescale=True) and, if `unwrap`, unwrap UVs with xatlas (~10 s for
    80k faces). Without unwrapping the mesh can still render the ig2mv condition maps, but not be baked.

    front_yaw_deg turns the mesh about +Y before texturing, for meshes whose front isn't +Z; only the
    cameras' view of it changes, the returned geometry keeps its original orientation.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    yaw = trimesh.transformations.rotation_matrix(np.radians(front_yaw_deg), [0, 1, 0])[:3, :3]
    scale = np.abs(vertices).max()
    std_pos = (vertices @ yaw.T / scale * 0.5) @ _MESH_TO_STD.T
    std_nrm = np.asarray(mesh.vertex_normals, dtype=np.float64) @ yaw.T @ _MESH_TO_STD.T

    colors = None
    if mesh.visual.kind == "vertex":
        colors = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float32) / 255

    if unwrap:
        vmapping, faces, uvs = xatlas.parametrize(vertices, np.asarray(mesh.faces))
    else:
        vmapping, faces = np.arange(len(vertices)), np.asarray(mesh.faces)
        uvs = np.zeros((len(vertices), 2), dtype=np.float32)
    return PreparedMesh(
        original=mesh,
        vmapping=vmapping,
        faces=faces.astype(np.int32),
        uvs=uvs.astype(np.float32),
        pos=std_pos[vmapping].astype(np.float32),
        nrm=std_nrm[vmapping].astype(np.float32),
        col=None if colors is None else colors[vmapping],
    )


class Rasterizer:
    """Headless OpenGL rasteriser. Create and use it on one thread (the GL context is bound to it)."""

    def __init__(self, pm: PreparedMesh):
        self.ctx = moderngl.create_context(standalone=True, backend="egl")
        self.prog = self.ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG)
        col = pm.col if pm.col is not None else np.full_like(pm.pos, 0.5)
        data = np.hstack([pm.pos, pm.nrm, col, pm.uvs]).astype("f4")
        self.vbo = self.ctx.buffer(data.tobytes())
        self.ibo = self.ctx.buffer(pm.faces.astype("i4").tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, "3f 3f 3f 2f", "in_pos", "in_nrm", "in_col", "in_uv")], self.ibo
        )

    def release(self) -> None:
        self.ctx.release()

    def _draw(self, size: int, mvp: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns position, normal and colour maps (H, W, 4; alpha = coverage). Row 0 is clip-space y = -1,
        the same layout nvdiffrast produces, which is what MV-Adapter's cameras assume."""
        textures = [self.ctx.texture((size, size), 4, dtype="f4") for _ in range(3)]
        depth = self.ctx.depth_renderbuffer((size, size))
        fbo = self.ctx.framebuffer(color_attachments=textures, depth_attachment=depth)
        fbo.use()
        fbo.clear(0, 0, 0, 0, depth=1.0)
        self.prog["uv_space"].value = mvp is None
        if mvp is not None:
            self.prog["mvp"].write(np.ascontiguousarray(mvp.T, dtype="f4").tobytes())
            self.ctx.enable(moderngl.DEPTH_TEST)
        else:
            self.ctx.disable(moderngl.DEPTH_TEST)
        self.vao.render(moderngl.TRIANGLES)
        out = [np.frombuffer(fbo.read(attachment=i, components=4, dtype="f4"), dtype="f4").reshape(size, size, 4) for i in range(3)]
        fbo.release()
        depth.release()
        for t in textures:
            t.release()
        return out[0], out[1], out[2]

    def render_views(self, size: int) -> tuple[np.ndarray, np.ndarray]:
        """Position and normal maps for the six ig2mv cameras: (6, H, W, 4) each."""
        mvps = cameras().mvp_mtx.numpy()
        maps = [self._draw(size, mvp) for mvp in mvps]
        return np.stack([m[0] for m in maps]), np.stack([m[1] for m in maps])

    def render_uv(self, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-texel position, normal and vertex colour in UV space."""
        return self._draw(size, None)


def condition_images(pos_maps: np.ndarray, nrm_maps: np.ndarray) -> torch.Tensor:
    """ig2mv control input: position + 0.5 and normal / 2 + 0.5, grey (0.5) where there is no surface."""
    mask = pos_maps[..., 3:] > 0
    pos = np.where(mask, pos_maps[..., :3] + 0.5, 0.5).clip(0, 1)
    nrm = np.where(mask, nrm_maps[..., :3] / 2 + 0.5, 0.5).clip(0, 1)
    return torch.from_numpy(np.concatenate([pos, nrm], axis=-1)).permute(0, 3, 1, 2).float()


def condition_preview(cond: torch.Tensor) -> Image.Image:
    """Normal maps of the six views side by side (for debugging / UI)."""
    nrm = (cond[:, 3:].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(np.concatenate(list(nrm), axis=1))


def bake(
    pm: PreparedMesh,
    raster: Rasterizer,
    views: list[Image.Image],
    view_pos_maps: np.ndarray,
    uv_size: int = 2048,
    device: str = "cuda",
) -> Image.Image:
    """Project the six generated views onto the UV texture, weighting each by how squarely it sees the surface."""
    uv_pos, uv_nrm, uv_col = raster.render_uv(uv_size)
    inside = uv_pos[..., 3] > 0
    texel_pos = torch.from_numpy(uv_pos[..., :3][inside]).to(device)
    texel_nrm = F.normalize(torch.from_numpy(uv_nrm[..., :3][inside]).to(device), dim=-1)

    cams = cameras(device)
    acc = torch.zeros(len(texel_pos), 3, device=device)
    wsum = torch.zeros(len(texel_pos), device=device)
    homo = torch.cat([texel_pos, torch.ones_like(texel_pos[:, :1])], dim=-1)
    for i, view in enumerate(views):
        img = torch.from_numpy(np.asarray(view.convert("RGB"), dtype=np.float32) / 255).to(device).permute(2, 0, 1)
        vpos = torch.from_numpy(view_pos_maps[i]).to(device).permute(2, 0, 1)  # (4, H, W)
        ndc = (homo @ cams.mvp_mtx[i].T)[:, :2]
        grid = ndc.view(1, -1, 1, 2)  # row 0 of the maps is ndc y = -1, as grid_sample expects for y = -1
        color = F.grid_sample(img[None], grid, mode="bilinear", align_corners=False)[0, :, :, 0].T
        seen = F.grid_sample(vpos[None], grid, mode="nearest", align_corners=False)[0, :, :, 0].T

        valid_px = _valid_pixels(view_pos_maps[i], cams.w2c[i].cpu().numpy())
        ok_px = F.grid_sample(torch.from_numpy(valid_px).float().to(device)[None, None], grid, mode="nearest", align_corners=False)[0, 0, :, 0] > 0.5

        visible = (seen[:, 3] > 0) & ((seen[:, :3] - texel_pos).norm(dim=-1) < 0.01) & ok_px
        to_camera = cams.c2w[i, :3, 2]  # orthographic: the camera's +z axis points from the scene to the camera
        cos = (texel_nrm * to_camera).sum(-1)
        w = torch.where(visible & (cos > 0.2), cos**3, torch.zeros_like(cos))
        acc += color * w[:, None]
        wsum += w

    covered = wsum > 0
    texels = torch.where(covered[:, None], acc / wsum.clamp_min(1e-8)[:, None], torch.zeros_like(acc))

    tex = np.zeros((uv_size, uv_size, 3), dtype=np.float32)
    tex[inside] = texels.cpu().numpy()
    hole = np.zeros((uv_size, uv_size), dtype=bool)
    hole[inside] = ~covered.cpu().numpy()
    if pm.col is not None:
        # Surfaces no camera saw (crevices, undersides) keep the mesh's own vertex colour.
        tex[hole] = uv_col[..., :3][hole]
        hole[:] = False
    log.info("Baked texture: %.1f%% of texels from views", 100 * covered.float().mean().item())

    tex8 = (tex.clip(0, 1) * 255).astype(np.uint8)
    if hole.any():
        tex8 = cv2.inpaint(tex8, hole.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    tex8 = _pad_islands(tex8, inside)
    # UV row 0 is v = 0; image row 0 is the top (v = 1 in glTF/OBJ conventions).
    return Image.fromarray(np.ascontiguousarray(tex8[::-1]))


def _valid_pixels(pos_map: np.ndarray, w2c: np.ndarray, dilation: int = 5, threshold: float = 0.1) -> np.ndarray:
    """Pixels safe to sample: on the surface and away from depth discontinuities, where generated views
    blend foreground and background."""
    mask = pos_map[..., 3] > 0
    depth = (pos_map[..., :3] @ w2c[:3, :3].T + w2c[:3, 3])[..., 2]
    depth = np.where(mask, depth, depth[mask].min() if mask.any() else 0)
    gy, gx = np.gradient(depth)
    edges = (np.hypot(gx, gy) > threshold / pos_map.shape[0] * 64) | ~mask
    edges = cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=dilation)
    return (mask & (edges == 0)).astype(np.float32)


def _pad_islands(tex: np.ndarray, inside: np.ndarray, iterations: int = 8) -> np.ndarray:
    """Bleed island colours outwards so filtering/mipmapping doesn't pull in the empty background at seams."""
    out = tex.astype(np.float32)
    filled = inside.astype(np.float32)
    kernel = np.ones((3, 3), np.float32)
    for _ in range(iterations):
        summed = cv2.filter2D(out * filled[..., None], -1, kernel, borderType=cv2.BORDER_CONSTANT)
        count = cv2.filter2D(filled, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        grow = (filled == 0) & (count > 0)
        out[grow] = summed[grow] / count[grow, None]
        filled[grow] = 1
    return out.clip(0, 255).astype(np.uint8)


def textured_mesh(pm: PreparedMesh, texture: Image.Image) -> trimesh.Trimesh:
    """The original geometry (split at UV seams) with the baked texture as a PBR base colour."""
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=texture, metallicFactor=0.0, roughnessFactor=0.9, doubleSided=False
    )
    vertices = np.asarray(pm.original.vertices)[pm.vmapping]
    visual = trimesh.visual.TextureVisuals(uv=pm.uvs, material=material)
    return trimesh.Trimesh(vertices=vertices, faces=pm.faces, visual=visual, process=False)
