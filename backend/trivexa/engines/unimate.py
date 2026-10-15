"""Text -> motion for a rigged model with UniMate (Mou et al., SIGGRAPH Asia 2026; MIT code, CC BY-NC 4.0 weights).

UniMate is a flow-matching transformer that animates skeletons of any topology. It is conditioned on the skeleton
itself: its rest pose in a canonical frame (facing +Z, centred, scaled to a leaf-to-leaf diameter of 2, grounded), the
tree's graph distances, edge types, depths and Laplacian eigenvectors, and a flan-t5-base embedding of each joint's
*label* in a shared anatomical vocabulary ("Left Thigh", "Tail"). The caption reaches it through adaLN. It writes 60
frames at 30 fps: per joint a root-relative position, a rotation relative to the rest pose and a velocity; the root
slot holds the facing and the trajectory.

Upstream prepares a rig with Blender (`data_process/rig_preprocess`): export, label the joints, pick a "facing pair",
canonicalise, bake a canonical GLB. Trivexa's rigs (SkinTokens) already are skeletons with identity rest rotations, so
this engine does the stages that matter itself: the rule labeller (vendored as is; offline, with geometric labels for
SkinTokens' unnamed `bone_<i>` joints), stage 4's T-pose canonicalisation and topology conditioning, and, after
sampling, the inverse of the canonical transform so the motion drives the original GLB's joints directly.
"""

import json
import logging
import tempfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from . import Cancelled
from ..config import UNIMATE_DIR

log = logging.getLogger(__name__)

RUN_DIR = UNIMATE_DIR / "unimate_uniml3d_f60_v3"
CHECKPOINT = RUN_DIR / "checkpoints" / "checkpoint_step_150000.pt"
T5_DIR = UNIMATE_DIR / "flan-t5-base"
FPS = 30
FRAMES = 60  # the model's fixed window (2 s)
DEFAULT_CFG = 3.0  # upstream's sampling.cfg_scale
# Upstream rig_preprocess: rigs are labelled like the 'general' profile, whose stats fall back to objaverse's
# (STATS_FALLBACK); the v3 run normalises every dataset with one pool anyway.
PROFILE = "general"
STATS_KEY = "objaverse"
FACING_EPS = 1e-6  # data_process/utils/skeleton.py


def available() -> bool:
    return CHECKPOINT.is_file() and (RUN_DIR / "dataset_stats.npy").is_file() and (T5_DIR / "model.safetensors").is_file()


@dataclass
class Skeleton:
    joints: np.ndarray  # (J, 3) rest positions (Y-up), rest rotations identity
    parents: list[int]
    names: list[str]


@dataclass
class Prepared:
    """A skeleton labelled and canonicalised as UniMate's stage 4 would, in BFS order."""

    order: list[int]  # BFS position -> index in the input skeleton
    parents: list[int]
    labels: list[str]  # clean joint labels (BFS order)
    face: dict  # the facing pair entry (raw / clean names), empty when none was found
    facing: np.ndarray  # (4,) w, x, y, z: the root rotation that turns the rest pose to face +Z
    scale: float
    center_xz: np.ndarray  # (3,) subtracted after the facing turn, before scaling
    ground: float  # subtracted from Y after scaling
    tpos: np.ndarray  # (J, 3) canonical rest pose
    cond: dict  # topology fields of the cond entry
    notes: list[str] = field(default_factory=list)


@dataclass
class AnimateParams:
    prompt: str
    seed: int = 0
    cfg_scale: float = DEFAULT_CFG


@dataclass
class Motion:
    rotations: np.ndarray  # (F, J, 4) x, y, z, w local rotations of the input skeleton's joints (glTF order)
    root_translation: np.ndarray  # (F, 3) the root joint's local translation
    fps: int
    labels: list[str]  # clean label per input joint
    face: dict
    consistency: float  # mean distance between the model's own joint positions and the rotations' FK, per diameter


class UniMateEngine:
    def __init__(self):
        self.model = None
        self.t5 = None
        self.tokenizer = None
        self.stats = None
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- loading ------------------------------------------------------------------------------------------------

    def load(self) -> None:
        if self.model is not None:
            return
        from transformers import T5EncoderModel, T5Tokenizer

        from ..vendor.unimate.models.denoiser import UniMateGraphAdaLN

        self.config = json.loads((RUN_DIR / "config.json").read_text())
        ds, mc = self.config["dataset"], self.config["model"]
        if (mc["attention"], mc["text_cond"]) != ("graph", "adaln") or mc["text_encoder_version"] != "google/flan-t5-base":
            raise RuntimeError(f"unsupported UniMate run in {RUN_DIR}")
        stats = np.load(RUN_DIR / "dataset_stats.npy", allow_pickle=True).item()[STATS_KEY]
        self.stats = {k: np.asarray(stats[k], dtype=np.float64) for k in ("mean_root", "std_root", "mean_local", "std_local")}

        # upstream's create_model, for this run's axes
        model = UniMateGraphAdaLN(
            feature_len=ds["feature_len"],
            max_motion_length=ds["max_motion_length"],
            max_joints=ds["max_joints"],
            max_depth=ds["max_depth"],
            latent_dim=mc["latent_dim"],
            ff_size=mc["ff_size"],
            num_layers=mc["num_layers"],
            num_heads=mc["num_heads"],
            dropout=mc["dropout"],
            use_spectral_rope=mc["use_spectral_rope"],
            max_freqs=mc["max_freqs"],
            use_signnet=mc["use_signnet"],
            cond_mode=mc["cond_mode"],
            cond_mask_prob=mc["cond_mask_prob"],
            text_dim=768,  # flan-t5-base
            use_joint_name_emb=mc["use_joint_name_emb"],
            use_graph_emb=mc["use_graph_emb"],
            use_depth_emb=mc["use_depth_emb"],
            concat_parent_features=mc["concat_parent_features"],
            num_tpos_queries=mc["num_tpos_queries"],
            inject_tpos_to_adaln=mc["inject_tpos_to_adaln"],
            use_graph_attn_bias=mc["use_graph_attn_bias"],
            share_graph_attn_bias=mc["share_graph_attn_bias"],
        )
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        # Inference uses the EMA weights (upstream's EMAModel.copy_to: shadow params in parameter order).
        trainable = [p for p in model.parameters() if p.requires_grad]
        shadow = ckpt["ema_state_dict"]["shadow_params"]
        if len(shadow) != len(trainable) or any(s.shape != p.shape for s, p in zip(shadow, trainable)):
            raise RuntimeError("UniMate EMA weights do not match the model")
        with torch.no_grad():
            for s, p in zip(shadow, trainable):
                p.copy_(s)
        del ckpt
        self.model = model.to(self.device).eval()
        self.tokenizer = T5Tokenizer.from_pretrained(T5_DIR)
        self.t5 = T5EncoderModel.from_pretrained(T5_DIR, torch_dtype=torch.float32).to(self.device).eval()
        log.info("UniMate loaded (%.1fM parameters)", sum(p.numel() for p in model.parameters()) / 1e6)

    def unload(self) -> None:
        if self.model is None:
            return
        self.model = self.t5 = self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- skeleton preparation -----------------------------------------------------------------------------------

    def prepare(self, skeleton: Skeleton) -> Prepared:
        return prepare(skeleton, self.config["dataset"]["max_joints"] if self.config else 71)

    # -- sampling -----------------------------------------------------------------------------------------------

    @torch.no_grad()
    def _encode(self, texts: list[str]) -> list[np.ndarray]:
        """T5 hidden states mean-pooled over each text's tokens; the empty string pools to zero (upstream's rule)."""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        mask = inputs["attention_mask"].clone()
        for i, text in enumerate(texts):
            if text == "":
                mask[i] = 0
        hidden = self.t5(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
        out = []
        for i in range(len(texts)):
            n = int(mask[i].sum())
            out.append(hidden[i, :n].float().mean(0).cpu().numpy() if n else np.zeros(hidden.shape[-1], np.float32))
        return out

    @torch.no_grad()
    def animate(
        self,
        skeleton: Skeleton,
        params: AnimateParams,
        on_step: Callable[[int, int, object], None] = lambda *a: None,
        is_cancelled: Callable[[], bool] = lambda: False,
    ) -> Motion:
        from torchdiffeq import odeint

        self.load()
        prep = self.prepare(skeleton)
        for note in prep.notes:
            log.info("UniMate: %s", note)
        cond, mean, std = self._cond(prep, params.prompt)
        J = len(prep.parents)
        ds = self.config["dataset"]
        shape = (1, ds["max_joints"], ds["feature_len"], ds["max_motion_length"])
        generator = torch.Generator(device=self.device).manual_seed(params.seed)
        noise = torch.randn(shape, generator=generator, device=self.device)

        total = 100

        def drift(t, x):  # upstream: velocity ODE, classifier-free guidance around one conditional/unconditional pair
            if is_cancelled():
                raise Cancelled()
            on_step(min(int(float(t) * total), total - 1), total, None)
            ts = torch.full((x.shape[0],), float(t), device=x.device)
            out = self.model(x, ts, cond)
            if params.cfg_scale == 1.0:
                return out
            uncond = self.model(x, ts, cond, force_mask=True)
            return uncond + params.cfg_scale * (out - uncond)

        # upstream's Sampler.sample_ode defaults: dopri5, atol 1e-6, rtol 1e-3, 50 saved points on [0, 1]
        t = torch.linspace(0, 1, 50, device=self.device)
        sample = odeint(drift, noise, t, method="dopri5", atol=[1e-6], rtol=[1e-3])[-1]
        on_step(total, total, None)

        feats = sample[0, :J].permute(2, 0, 1).double().cpu().numpy() * std[None] + mean[None]  # (F, J, 12)
        return self._decode(prep, skeleton, feats)

    def _cond(self, prep: Prepared, prompt: str):
        """The batch `create_sample_condition` + `mixture_batch_collate` build for one rest-pose reference clip."""
        from ..vendor.unimate.quaternions import Quaternions

        ds = self.config["dataset"]
        J, max_joints, D = len(prep.parents), ds["max_joints"], ds["feature_len"]
        mean = np.zeros((J, D))
        std = np.zeros((J, D))
        mean[0], std[0] = self.stats["mean_root"], self.stats["std_root"]
        mean[1:], std[1:] = self.stats["mean_local"], self.stats["std_local"]

        tpos = prep.tpos.copy()
        tpos[..., 1] -= tpos[..., 1].min()  # ground_motion_height
        pad = np.zeros((J, D - 3))
        pad[:, :6] = Quaternions.id(1).rotation_matrix(cont6d=True)[0]  # identity 6D rotation, zero velocity
        tpos = (np.concatenate([tpos, pad], axis=-1) - mean) / std
        tpos_parents = tpos.copy()
        for j, p in enumerate(prep.parents):
            if p != -1:
                tpos_parents[j] = tpos[p]

        names = list(dict.fromkeys(prep.labels))
        name_emb = dict(zip(names, self._encode(names)))
        caption = self._encode([prompt])[0]

        def pad_joints(a, tail):
            out = torch.zeros((1, max_joints, *tail))
            out[0, : len(a)] = torch.as_tensor(np.asarray(a), dtype=torch.float32)
            return out

        def pad_square(a):
            out = torch.zeros((1, max_joints, max_joints), dtype=torch.long)
            out[0, :J, :J] = torch.as_tensor(np.asarray(a), dtype=torch.long)
            return out

        c = prep.cond
        n_joints = torch.tensor([J])
        dev = self.device
        cond = {
            "motion_length": torch.tensor([ds["max_motion_length"]], device=dev),
            "n_joints": n_joints.to(dev),
            "joint_mask": (torch.arange(max_joints)[None] < n_joints[:, None])[:, None, None].to(dev),
            "tpos_first_frame": pad_joints(tpos, (D,)).to(dev),
            "tpos_first_frame_parents": pad_joints(tpos_parents, (D,)).to(dev),
            "parents": [np.asarray(prep.parents)],
            "edge_indexs": [torch.as_tensor(c["edge_indexs"], dtype=torch.long)],
            "joint_relations": pad_square(c["joint_relations"]).to(dev),
            "graph_dist": pad_square(c["joint_graph_dists"]).to(dev),
            "joint_depths": pad_joints(c["joint_depths"], ()).long().to(dev),
            "spectral_feats": pad_joints(c["spectral_feats"], (c["spectral_feats"].shape[1],)).to(dev),
            "joint_names_emb": pad_joints(np.stack([name_emb[n] for n in prep.labels]), (768,)).to(dev),
            "offsets": pad_joints(c["offsets"], (3,)).to(dev),
            "caption_emb": torch.as_tensor(caption, dtype=torch.float32)[None].to(dev),
        }
        return cond, mean, std

    def _decode(self, prep: Prepared, skeleton: Skeleton, feats: np.ndarray) -> Motion:
        """Rotations and root path of the input skeleton from the denormalised features (F, J, 12, BFS order)."""
        from ..vendor.unimate.quaternions import Quaternions
        from ..vendor.unimate.utils.motion_utils import (
            hml_rotations_to_bvh_quaternions,
            recover_unimate_joint_pos_from_ric,
            recover_unimate_root_quat_and_pos,
        )

        _, root_pos = recover_unimate_root_quat_and_pos(feats[:, 0])
        rel = hml_rotations_to_bvh_quaternions(feats[:, :, 3:9], prep.parents)  # T-pose-relative local rotations
        q = Quaternions(prep.facing)
        # The rest pose's local rotations are the facing turn at the root and identity elsewhere, so a T-pose-relative
        # rotation r_j (upstream's compute_rots_from_tpos) and the input skeleton's local rotation b_j are related by
        # b_j = q^-1 r_j q at every joint, root included; positions map back through the canonical transform.
        local = (-q)[None, None] * rel * q[None, None]
        root = (-q)[None] * ((root_pos + [0.0, prep.ground, 0.0]) / prep.scale + prep.center_xz)

        F, J = local.shape
        rotations = np.zeros((F, J, 4))
        rotations[:, prep.order] = local.qs[..., [1, 2, 3, 0]]  # glTF quaternions are x, y, z, w

        # How far the rotations' FK lands from the joint positions the model also predicts (a diagnostic, as a share
        # of the skeleton's diameter): upstream drives meshes with the rotations, the positions only cross-check them.
        g = np.asarray(skeleton.joints, dtype=np.float64)[prep.order]
        fk = _fk(local.qs, root, g - g[[max(p, 0) for p in prep.parents]], prep.parents)
        fk_canonical = prep.scale * (q[None, None] * fk - prep.center_xz) - [0.0, prep.ground, 0.0]
        ric = recover_unimate_joint_pos_from_ric(feats)
        consistency = float(np.linalg.norm(fk_canonical - ric, axis=-1).mean() / 2.0)

        labels = [""] * J
        for pos, idx in enumerate(prep.order):
            labels[idx] = prep.labels[pos]
        # The root joint hangs off an identity "Armature" node, so its local translation is its position.
        return Motion(rotations, root, FPS, labels, prep.face, consistency)


def _fk(local_wxyz: np.ndarray, root: np.ndarray, offsets: np.ndarray, parents: list[int]) -> np.ndarray:
    """Global positions (F, J, 3) from local rotations, the root path and rest offsets (parent-relative)."""
    from ..vendor.unimate.quaternions import Quaternions

    F, J = local_wxyz.shape[:2]
    pos = np.zeros((F, J, 3))
    rot = np.zeros((F, J, 4))
    for j, p in enumerate(parents):
        if p < 0:
            pos[:, j], rot[:, j] = root, local_wxyz[:, j]
        else:
            pos[:, j] = pos[:, p] + Quaternions(rot[:, p]) * np.broadcast_to(offsets[j], (F, 3))
            rot[:, j] = (Quaternions(rot[:, p]) * Quaternions(local_wxyz[:, j])).qs
    return pos


# -- stage 3 + 4 for one skeleton ---------------------------------------------------------------------------------


def prepare(skeleton: Skeleton, max_joints: int = 71) -> Prepared:
    from ..vendor.unimate.data_process.rig_preprocess.annotate import annotate
    from ..vendor.unimate.data_process.utils import topology
    from ..vendor.unimate.quaternions import Quaternions

    joints = np.asarray(skeleton.joints, dtype=np.float64)
    parents, names = list(skeleton.parents), list(skeleton.names)
    J = len(parents)
    if parents.count(-1) != 1:
        raise ValueError(f"the skeleton has {parents.count(-1)} roots; UniMate needs one")
    if J > max_joints:
        raise ValueError(f"the skeleton has {J} joints; UniMate takes at most {max_joints}")
    if J < 2:
        raise ValueError("the skeleton needs at least two joints")
    notes = []

    # Stage 3: rule labels and the facing pair, from an export NPZ like upstream's (rest rotations are identity).
    local_pos = joints - np.where(np.asarray(parents)[:, None] >= 0, joints[[max(p, 0) for p in parents]], 0.0)
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "motions").mkdir()
        np.savez(
            Path(tmp) / "motions" / "asset-rest_pose.npz",
            names=np.array(names),
            parents=np.array(parents),
            rest_local_pos=local_pos,
            rest_local_rot=np.tile([1.0, 0.0, 0.0, 0.0], (J, 1)),
            fps=np.array(FPS),
        )
        labels, face, ann_notes = annotate("rule", PROFILE, "asset", names, tmp, rest_pos=joints)
    notes += ann_notes

    # Stage 4: BFS order (children by subtree size, then bone length), facing, centre XZ, scale, ground.
    order, bfs_parents = _bfs_order(parents, np.linalg.norm(local_pos, axis=-1))
    g = joints[order]
    bfs_names = [names[i] for i in order]
    bfs_labels = [labels[i] for i in order]
    r, l = face.get("r_hip", {}).get("raw"), face.get("l_hip", {}).get("raw")
    facing = Quaternions.id(1)[0]
    if r and l and r in bfs_names and l in bfs_names:
        facing = _facing_quat(g, bfs_names.index(r), bfs_names.index(l), face.get("body_axis", False))
    else:
        notes.append("no facing pair: the rest pose is taken to face +Z")
    turned = facing[None] * g
    center = turned[0] * np.array([1.0, 0.0, 1.0])
    diameter = _diameter(bfs_parents, np.linalg.norm(local_pos[order], axis=-1))
    if not np.isfinite(diameter) or diameter <= 1e-8:
        raise ValueError("degenerate skeleton (zero-length bones)")
    scale = 2.0 / diameter
    scaled = (turned - center) * scale
    ground = float(scaled[:, 1].min())
    tpos = scaled - [0.0, ground, 0.0]

    relations, graph_dists = topology.compute_edge_relations_and_distances(np.asarray(bfs_parents), max_path_len=5)
    spectral, _ = topology.compute_laplacian_eigenvectors(np.asarray(bfs_parents), max_freqs=8)
    cond = {
        "joint_relations": relations,
        "joint_graph_dists": graph_dists,
        "joint_depths": topology.compute_joint_depths(np.asarray(bfs_parents)),
        "edge_indexs": topology.compute_edge_indexs(np.asarray(bfs_parents)),
        "spectral_feats": spectral,
        # upstream's offsets_from_positions on the grounded T-pose: root position, then parent-relative offsets
        "offsets": tpos - np.where(np.asarray(bfs_parents)[:, None] >= 0, tpos[[max(p, 0) for p in bfs_parents]], 0.0),
    }
    return Prepared(order, bfs_parents, bfs_labels, face, facing.qs, scale, center, ground, tpos, cond, notes)


def _bfs_order(parents: list[int], bone_lengths: np.ndarray) -> tuple[list[int], list[int]]:
    """Upstream's bfs_reorder_joints: breadth first from the root, children by subtree size (largest first), then
    bone length (shortest first)."""
    n = len(parents)
    children = [[] for _ in range(n)]
    for j, p in enumerate(parents):
        if p >= 0:
            children[p].append(j)
    root = parents.index(-1)
    size = [1] * n
    topo, q = [], deque([root])
    while q:
        node = q.popleft()
        topo.append(node)
        q.extend(children[node])
    for node in reversed(topo):
        for c in children[node]:
            size[node] += size[c]
    for node in range(n):
        children[node].sort(key=lambda c: (-size[c], bone_lengths[c]))
    order, q = [], deque([root])
    while q:
        node = q.popleft()
        order.append(node)
        q.extend(children[node])
    new = {old: i for i, old in enumerate(order)}
    return order, [-1 if parents[old] == -1 else new[parents[old]] for old in order]


def _diameter(parents: list[int], bone_lengths: np.ndarray) -> float:
    """Leaf-to-leaf geodesic diameter of the bone-length-weighted tree (upstream's get_skeleton_diameter)."""
    adj = {i: [] for i in range(len(parents))}
    for i, p in enumerate(parents):
        if p != -1:
            adj[p].append((i, bone_lengths[i]))
            adj[i].append((p, bone_lengths[i]))

    def farthest(start):
        dist, queue, best = {start: 0.0}, [start], (start, 0.0)
        for u in queue:
            for v, w in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + w
                    queue.append(v)
                    if dist[v] > best[1]:
                        best = (v, dist[v])
        return best

    return farthest(farthest(0)[0])[1]


def _facing_quat(g: np.ndarray, r: int, l: int, body_axis: bool):
    """Upstream's get_root_facing_quat for one frame: the Y turn taking forward = Y x (right - left) onto +Z."""
    from ..vendor.unimate.quaternions import Quaternions

    across = g[r] - g[l]
    across = across / max(np.linalg.norm(across), 1e-8)
    forward = np.cross([0.0, 1.0, 0.0], across)
    if np.linalg.norm(forward) <= FACING_EPS:
        raise ValueError("the facing pair has no horizontal separation")
    forward = forward / max(np.linalg.norm(forward), 1e-8)
    if body_axis:
        forward = Quaternions.from_angle_axis(np.array(-np.pi / 2), np.array([0.0, 1.0, 0.0])) * forward
    target = np.array([0.0, 0.0, 1.0])
    if forward @ target < -1.0 + 1e-9:
        return Quaternions(np.array([0.0, 0.0, 1.0, 0.0]))
    return Quaternions.between(forward[None], target[None])[0]
