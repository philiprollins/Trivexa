# Trivexa: written for Trivexa, not upstream code. UniMate builds on the Motion library's `Quaternions` class (no
# license), so this is a from-scratch implementation of the part of its interface that `utils/motion_utils.py` and the
# UniMate engine use, with the same conventions: (w, x, y, z) arrays, `a * b` the Hamilton product, `q * v` rotating
# 3-vectors, `-q` the inverse of a unit quaternion, and indexing / `repeat` acting on the leading (quaternion) axes.
import numpy as np


class Quaternions:
    def __init__(self, qs):
        if isinstance(qs, Quaternions):
            qs = qs.qs
        self.qs = np.asarray(qs, dtype=np.float64)

    @property
    def shape(self):
        return self.qs.shape[:-1]

    def __len__(self):
        return len(self.qs)

    def __getitem__(self, k):
        return Quaternions(self.qs[k])

    def __setitem__(self, k, v):
        self.qs[k] = v.qs

    def copy(self):
        return Quaternions(self.qs.copy())

    def repeat(self, n, **kwargs):
        return Quaternions(self.qs.repeat(n, **kwargs))

    def __neg__(self):
        return Quaternions(self.qs * np.array([1.0, -1.0, -1.0, -1.0]))

    def __mul__(self, other):
        if isinstance(other, Quaternions):
            return Quaternions(_qmul(self.qs, other.qs))
        other = np.asarray(other, dtype=np.float64)
        if other.shape[-1] == 3:
            # q v q^-1 on (0, v)
            vq = np.concatenate([np.zeros(other.shape[:-1] + (1,)), other], axis=-1)
            return _qmul(_qmul(self.qs, vq), (-self).qs)[..., 1:]
        raise TypeError(f"cannot multiply Quaternions by an array of shape {other.shape}")

    @property
    def lengths(self):
        return np.sqrt((self.qs**2).sum(axis=-1))

    def normalized(self):
        lengths = self.lengths[..., None]
        lengths[lengths == 0] = 1
        return Quaternions(self.qs / lengths)

    def transforms(self):
        """Rotation matrices (..., 3, 3), acting on column vectors."""
        w, x, y, z = (self.qs[..., i] for i in range(4))
        s = 2.0 / (self.qs**2).sum(-1)
        m = np.stack(
            [
                1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w),
                s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w),
                s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y),
            ],
            -1,
        )
        return m.reshape(self.qs.shape[:-1] + (3, 3))

    def rotation_matrix(self, cont6d=False):
        """The rotation matrices, or with `cont6d` their first two columns concatenated (the 6D representation)."""
        m = self.transforms()
        return np.concatenate([m[..., 0], m[..., 1]], axis=-1) if cont6d else m

    @classmethod
    def id(cls, n):
        shape = (n,) if isinstance(n, int) else tuple(n)
        qs = np.zeros(shape + (4,))
        qs[..., 0] = 1.0
        return cls(qs)

    @classmethod
    def between(cls, v0s, v1s):
        """The shortest rotations taking the vectors `v0s` onto `v1s`."""
        a = np.cross(v0s, v1s)
        w = np.sqrt((v0s**2).sum(axis=-1) * (v1s**2).sum(axis=-1)) + (v0s * v1s).sum(axis=-1)
        return cls(np.concatenate([w[..., None], a], axis=-1)).normalized()

    @classmethod
    def from_angle_axis(cls, angles, axis):
        axis = np.asarray(axis, dtype=np.float64)
        axis = axis / np.sqrt((axis**2).sum(axis=-1))[..., None]
        half = np.asarray(angles, dtype=np.float64)[..., None] / 2.0
        return cls(np.concatenate([np.cos(half), np.sin(half) * axis], axis=-1))


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.broadcast_arrays(a, b)
    w1, x1, y1, z1 = (a[..., i] for i in range(4))
    w2, x2, y2, z2 = (b[..., i] for i in range(4))
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        -1,
    )
