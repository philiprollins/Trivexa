"""Mesh -> parts with PartField (NVIDIA, non-commercial; vendored code, local weights).

PartField predicts a feature field for a shape: a PVCNN encodes 100k surface samples into a triplane, which a
transformer refines into 448 part-feature channels. Features averaged over each face are then clustered. As upstream
does for meshes, the clustering is agglomerative (Ward) under the mesh's face adjacency, so every part is one
connected region, and the part count is simply where the merge tree is cut.

Upstream runs through Lightning and a dataset of files; this engine works on trimesh meshes directly and builds the
two networks the features need (the SDF decoder is training-only).
"""

import logging
import pickle
import types
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial import cKDTree

from . import Cancelled
from ..config import PARTFIELD_DIR

log = logging.getLogger(__name__)

CHECKPOINT = PARTFIELD_DIR / "model_objaverse.ckpt"
# configs/final/demo.yaml over upstream's defaults: the parts of the config the inference networks read.
TRIPLANE_CHANNELS_LOW = 128
TRIPLANE_CHANNELS_HIGH = 512
SDF_CHANNELS = 64  # the first 64 triplane channels feed the SDF decoder; the other 448 are the part features
PVCNN_CONFIG = {
    "point_encoder_type": "pvcnn",
    "use_point_scatter": True,
    "z_triplane_channels": 256,
    "z_triplane_resolution": 128,
    "unet_cfg": {
        "depth": 3,
        "enabled": True,
        "rolled": True,
        "use_3d_aware": True,
        "start_hidden_channels": 32,
        "use_initial_conv": False,
    },
}
SURFACE_POINTS = 100_000  # point cloud the encoder sees (upstream's pc_num_pts)
# Upstream averages 1000 samples per face over Objaverse-sized meshes; generated meshes have ~200k faces, where a few
# dozen per face give the same means. The budget caps the total, never going below MIN_FACE_SAMPLES.
FACE_SAMPLE_BUDGET = 8_000_000
MIN_FACE_SAMPLES = 8
SAMPLE_CHUNK = 500_000
MAX_PARTS = 32
# Loose pieces smaller than this are not clustered (see PartFieldEngine.generate).
MIN_PIECE_FACES = 64
MIN_PIECE_FRACTION = 0.005
# Tripo's Segment presets (Simple 3-6, Balanced 6-15, Detailed 15+ parts).
DETAIL_PARTS = {"simple": 5, "balanced": 10, "detailed": 18}


class _Cfg(dict):
    """yacs-style config: the vendored code reads it both as cfg.key and cfg["key"]."""

    def __getattr__(self, name):
        value = self[name]
        return _Cfg(value) if isinstance(value, dict) else value


@dataclass
class SegmentParams:
    mesh: trimesh.Trimesh  # any frame; normalised internally
    parts: int = DETAIL_PARTS["balanced"]
    seed: int = 0


def available() -> bool:
    return CHECKPOINT.is_file()


def _load_state_dict(path) -> dict[str, torch.Tensor]:
    """The Lightning checkpoint, unpickled without yacs (its config is pickled as a CfgNode) and without letting the
    file name any other Python object than tensors, plain containers and the CfgNode."""
    allowed = {("collections", "OrderedDict"), ("torch._utils", "_rebuild_tensor_v2"), ("__builtin__", "set")}

    class CfgNode(dict):
        def __setstate__(self, state):
            pass

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) == ("yacs.config", "CfgNode"):
                return CfgNode
            if (module, name) in allowed:
                return super().find_class(module, name)
            raise pickle.UnpicklingError(f"{module}.{name} is not allowed in the PartField checkpoint")

    pickle_module = types.SimpleNamespace(Unpickler=Unpickler, load=pickle.load, __name__="pickle")
    ckpt = torch.load(path, map_location="cpu", weights_only=False, pickle_module=pickle_module, mmap=True)
    return ckpt["state_dict"]


def _edge_adjacency(mesh: trimesh.Trimesh):
    """(F, F) faces sharing an edge. Textured meshes are split at UV seams (and some at normal creases), so this
    welds vertices by position first; welding only renumbers vertices, faces keep their order."""
    welded = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    welded.merge_vertices(merge_tex=True, merge_norm=True)
    n = len(welded.faces)
    pairs = welded.face_adjacency
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)).tocsr()
    return adj + adj.T


def _link_components(adj, centroids: np.ndarray, k: int = 10):
    """`adj` plus links between its connected components, so the merge tree spans the whole mesh.

    As upstream's `construct_face_adjacency_matrix_ccmst` with kNN: each component is represented by its face nearest
    its centroid, and those faces get a minimum spanning tree over their k nearest neighbours plus the kNN edges.
    """
    n = adj.shape[0]
    count, labels = connected_components(adj, directed=False)
    if count == 1:
        return adj
    reps, rep_centroids = [], []
    order = np.argsort(labels, kind="stable")
    bounds = np.searchsorted(labels[order], np.arange(count + 1))
    for c in range(count):
        faces = order[bounds[c] : bounds[c + 1]]
        centre = centroids[faces].mean(0)
        reps.append(faces[np.argmin(np.linalg.norm(centroids[faces] - centre, axis=1))])
        rep_centroids.append(centre)
    reps = np.array(reps)
    tree = cKDTree(np.array(rep_centroids))
    while True:  # widen the kNN until it spans every component (clustered pieces can leave it split)
        dist, idx = tree.query(rep_centroids, k=min(k, count))
        rows = np.repeat(np.arange(count), idx.shape[1])
        cols = idx.reshape(-1)
        keep = rows != cols
        rows, cols, dist = rows[keep], cols[keep], dist.reshape(-1)[keep]
        knn = coo_matrix((dist + 1e-9, (rows, cols)), shape=(count, count))
        if k >= count or connected_components(knn, directed=False)[0] == 1:
            break
        k *= 4
    mst = minimum_spanning_tree(knn).tocoo()
    extra_r = np.concatenate([reps[mst.row], reps[rows]])
    extra_c = np.concatenate([reps[mst.col], reps[cols]])
    extra = coo_matrix((np.ones(len(extra_r)), (extra_r, extra_c)), shape=(n, n)).tocsr()
    return adj + extra + extra.T


class PartFieldEngine:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.pvcnn = None
        self.transformer = None

    def load(self) -> None:
        if self.pvcnn is not None:
            return
        from ..vendor.partfield.model.PVCNN.encoder_pc import TriPlanePC2Encoder
        from ..vendor.partfield.model.triplane import TriplaneTransformer

        log.info("Loading PartField from %s", CHECKPOINT)
        state = _load_state_dict(CHECKPOINT)
        pvcnn = TriPlanePC2Encoder(_Cfg(PVCNN_CONFIG), device="cpu", shape_min=-1, shape_length=2)
        transformer = TriplaneTransformer(
            input_dim=TRIPLANE_CHANNELS_LOW * 2,
            transformer_dim=1024,
            transformer_layers=6,
            transformer_heads=8,
            triplane_low_res=32,
            triplane_high_res=128,
            triplane_dim=TRIPLANE_CHANNELS_HIGH,
        )
        for module, prefix in ((pvcnn, "pvcnn."), (transformer, "triplane_transformer.")):
            module.load_state_dict({k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)})
        del state
        self.pvcnn = pvcnn.to(self.device).eval().requires_grad_(False)
        self.transformer = transformer.to(self.device).eval().requires_grad_(False)

    def unload(self) -> None:
        if self.pvcnn is None:
            return
        self.pvcnn = self.transformer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def face_features(self, mesh: trimesh.Trimesh, seed: int, is_cancelled: Callable[[], bool]) -> np.ndarray:
        """(F, 448) PartField features averaged over each face, L2-normalised (as upstream clusters them)."""
        from ..vendor.partfield.model.PVCNN.encoder_pc import sample_triplane_feat

        self.load()
        lo, hi = mesh.bounds
        vertices = (mesh.vertices - (lo + hi) / 2) * (2 * 0.9 / (hi - lo).max())  # upstream's normalisation
        normalised = trimesh.Trimesh(vertices, mesh.faces, process=False)
        rng = np.random.default_rng(seed)
        pc, _ = trimesh.sample.sample_surface(normalised, SURFACE_POINTS, seed=rng)
        pc = torch.as_tensor(pc, dtype=torch.float32, device=self.device)[None]
        with torch.autocast("cuda", dtype=torch.float16):  # upstream predicts with 16-mixed precision
            planes = self.transformer(self.pvcnn(pc, pc))
        part_planes = planes[:, :, SDF_CHANNELS:]

        faces = len(mesh.faces)
        per_face = max(MIN_FACE_SAMPLES, min(1000, FACE_SAMPLE_BUDGET // faces))
        tri = torch.as_tensor(np.array(normalised.triangles), dtype=torch.float32, device=self.device)  # (F, 3, 3)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        faces_per_chunk = max(1, SAMPLE_CHUNK // per_face)
        out = torch.empty(faces, part_planes.shape[2], device=self.device)
        for start in range(0, faces, faces_per_chunk):
            if is_cancelled():
                raise Cancelled()
            t = tri[start : start + faces_per_chunk]
            # Uniform barycentric samples, as upstream's sample_points (borrowed from Kaolin).
            u = torch.rand(len(t), per_face, 1, device=self.device, generator=gen).sqrt()
            v = torch.rand(len(t), per_face, 1, device=self.device, generator=gen)
            points = (1 - u) * t[:, None, 0] + u * (1 - v) * t[:, None, 1] + u * v * t[:, None, 2]
            with torch.autocast("cuda", dtype=torch.float16):
                feat = sample_triplane_feat(part_planes, points.reshape(1, -1, 3))
            out[start : start + len(t)] = feat.reshape(len(t), per_face, -1).float().mean(1)
        return torch.nn.functional.normalize(out, dim=-1).cpu().numpy()

    def generate(
        self,
        p: SegmentParams,
        on_step: Callable[[int, int, str | None], None],
        is_cancelled: Callable[[], bool],
    ) -> np.ndarray:
        """Per-face part labels, 0 for the largest part (by area) and so on."""
        from sklearn.cluster import AgglomerativeClustering

        on_step(5, 100, None)
        features = self.face_features(p.mesh, p.seed, is_cancelled)
        on_step(40, 100, None)
        if is_cancelled():
            raise Cancelled()
        # Pixal3D and MeshFlow meshes carry hundreds of small loose pieces (floaters, shells under the surface).
        # Linked into the merge tree they gather into one scattered part, so only substantial pieces are clustered
        # and each small one takes the label of the nearest clustered face.
        adjacency = _edge_adjacency(p.mesh)
        _, piece = connected_components(adjacency, directed=False)
        faces = len(p.mesh.faces)
        main = np.bincount(piece)[piece] >= max(MIN_PIECE_FACES, MIN_PIECE_FRACTION * faces)
        if main.sum() < p.parts:
            main[:] = True
        idx = np.flatnonzero(main)
        centroids = p.mesh.triangles_center
        connectivity = _link_components(adjacency[idx][:, idx], centroids[idx])
        parts = min(p.parts, len(idx))
        ward = AgglomerativeClustering(n_clusters=parts, connectivity=connectivity, linkage="ward")
        labels = np.empty(faces, dtype=np.int64)
        labels[idx] = ward.fit_predict(features[idx])
        if len(idx) < faces:
            rest = np.flatnonzero(~main)
            labels[rest] = labels[idx[cKDTree(centroids[idx]).query(centroids[rest])[1]]]
        area = np.bincount(labels, weights=p.mesh.area_faces, minlength=parts)
        rank = np.empty(parts, dtype=np.int64)
        rank[np.argsort(-area)] = np.arange(parts)
        on_step(100, 100, None)
        return rank[labels]


def split_parts(mesh: trimesh.Trimesh, labels: np.ndarray) -> trimesh.Scene:
    """One named node per part (part_01 = the largest), each keeping the source's UVs/material or vertex colours."""
    scene = trimesh.Scene()
    for i in range(labels.max() + 1):
        part = mesh.submesh([np.flatnonzero(labels == i)], append=True)
        name = f"part_{i + 1:02d}"
        scene.add_geometry(part, node_name=name, geom_name=name)
    return scene
