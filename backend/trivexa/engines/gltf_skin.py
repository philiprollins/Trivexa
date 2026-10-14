"""Write a skinned GLB: trimesh's export of the mesh (materials, textures, vertex colours) plus a glTF skin.

trimesh has no notion of skins, so the joints, inverse bind matrices and JOINTS_0/WEIGHTS_0 attributes are appended to
its GLB. Joints are nodes with translations only (rest pose = identity rotations); the root joint is added to the scene
next to the mesh, as Blender and three.js expect.
"""

import json
import struct

import numpy as np
import trimesh

_GLB_MAGIC, _JSON, _BIN = 0x46546C67, 0x4E4F534A, 0x004E4942
_FLOAT, _UBYTE, _USHORT = 5126, 5121, 5123


def skinned_glb(mesh: trimesh.Trimesh, joints: np.ndarray, parents: list[int], names: list[str], weights: np.ndarray) -> bytes:
    """`weights` is (vertices, joints) with at most four non-zero entries per row, each row summing to 1."""
    gltf, binary = _split(mesh.export(file_type="glb"))
    mesh_nodes = [i for i, n in enumerate(gltf["nodes"]) if "mesh" in n]
    if len(mesh_nodes) != 1 or len(gltf["meshes"][gltf["nodes"][mesh_nodes[0]]["mesh"]]["primitives"]) != 1:
        raise ValueError("expected a single mesh with one primitive")
    primitive = gltf["meshes"][gltf["nodes"][mesh_nodes[0]]["mesh"]]["primitives"][0]
    if gltf["accessors"][primitive["attributes"]["POSITION"]]["count"] != len(mesh.vertices):
        raise ValueError("the exported mesh's vertices do not match the rig's")

    # Up to four influences per vertex, strongest first.
    order = np.argsort(-weights, axis=1)[:, :4]
    vertex_weights = np.take_along_axis(weights, order, axis=1).astype(np.float32)
    order[vertex_weights == 0] = 0  # unused slots point at joint 0 with weight 0
    vertex_joints = order.astype(np.uint8 if len(joints) < 256 else np.uint16)

    def add(data: np.ndarray, component: int, kind: str, target: int | None = None, minmax: bool = False) -> int:
        nonlocal binary
        binary += b"\0" * (-len(binary) % 4)
        view = {"buffer": 0, "byteOffset": len(binary), "byteLength": data.nbytes}
        if target:
            view["target"] = target
        binary += data.tobytes()
        gltf["bufferViews"].append(view)
        accessor = {"bufferView": len(gltf["bufferViews"]) - 1, "componentType": component, "count": len(data), "type": kind}
        if minmax:
            accessor["min"], accessor["max"] = data.min(0).tolist(), data.max(0).tolist()
        gltf["accessors"].append(accessor)
        return len(gltf["accessors"]) - 1

    primitive["attributes"]["JOINTS_0"] = add(
        vertex_joints, _UBYTE if vertex_joints.dtype == np.uint8 else _USHORT, "VEC4", target=34962
    )
    primitive["attributes"]["WEIGHTS_0"] = add(vertex_weights, _FLOAT, "VEC4", target=34962)

    # Bind pose: joint j sits at joints[j] with no rotation, so its inverse bind matrix is a translation by -joints[j].
    joints = np.asarray(joints, dtype=np.float64)
    inverse_bind = np.tile(np.eye(4, dtype=np.float32), (len(joints), 1, 1))
    inverse_bind[:, 3, :3] = -joints  # column-major: the translation is the fourth column
    ibm = add(inverse_bind.reshape(len(joints), 16), _FLOAT, "MAT4")

    first = len(gltf["nodes"])
    children: dict[int, list[int]] = {}
    for j, parent in enumerate(parents):
        children.setdefault(parent, []).append(first + j)
    for j, parent in enumerate(parents):
        local = joints[j] - (joints[parent] if parent >= 0 else 0)
        node = {"name": names[j], "translation": local.tolist()}
        if j in children:
            node["children"] = children[j]
        gltf["nodes"].append(node)
    roots = children.get(-1, [])
    gltf["skins"] = [{"inverseBindMatrices": ibm, "joints": list(range(first, first + len(joints))), "skeleton": roots[0]}]
    gltf["nodes"][mesh_nodes[0]]["skin"] = 0
    gltf["nodes"].append({"name": "Armature", "children": roots})
    gltf["scenes"][gltf.get("scene", 0)]["nodes"].append(len(gltf["nodes"]) - 1)
    gltf["buffers"][0]["byteLength"] = len(binary)
    return _join(gltf, binary)


def _split(glb: bytes) -> tuple[dict, bytes]:
    magic, _, _ = struct.unpack_from("<III", glb, 0)
    if magic != _GLB_MAGIC:
        raise ValueError("not a GLB")
    offset, gltf, binary = 12, None, b""
    while offset < len(glb):
        length, kind = struct.unpack_from("<II", glb, offset)
        chunk = glb[offset + 8 : offset + 8 + length]
        if kind == _JSON:
            gltf = json.loads(chunk)
        elif kind == _BIN:
            binary = bytes(chunk)
        offset += 8 + length
    return gltf, binary


def _join(gltf: dict, binary: bytes) -> bytes:
    text = json.dumps(gltf, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    binary += b"\0" * (-len(binary) % 4)
    total = 12 + 8 + len(text) + 8 + len(binary)
    return (
        struct.pack("<III", _GLB_MAGIC, 2, total)
        + struct.pack("<II", len(text), _JSON) + text
        + struct.pack("<II", len(binary), _BIN) + binary
    )
