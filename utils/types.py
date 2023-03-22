from typing import Tuple

import jax
from flax.struct import dataclass


AABB = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
DensityAndRGB = Tuple[jax.Array, jax.Array]


@dataclass
class PinholeCamera:
    # resolutions
    W: int
    H: int

    # clipping plane distance, must be positive
    near: float
    far: float

    # focal length
    focal: float


@dataclass
class RayMarchingOptions:
    steps: int


@dataclass
class RenderingOptions:
    ray_chunk_size: int


@dataclass
class RigidTransformation:
    # [3, 3] rotatio matrix
    rotation: jax.Array
    # [3] translation vector
    translation: jax.Array
