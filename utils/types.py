from typing import Literal, Tuple

import jax
from flax.struct import dataclass


AABB = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
DensityAndRGB = Tuple[jax.Array, jax.Array]
LogLevel = Literal["DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"]


@dataclass
class PinholeCamera:
    # resolutions
    W: int
    H: int

    # focal length
    focal: float


@dataclass
class RayMarchingOptions:
    steps: int


@dataclass
class RenderingOptions:
    ray_chunk_size: int
    use_white_bg: bool


@dataclass
class RigidTransformation:
    # [3, 3] rotatio matrix
    rotation: jax.Array
    # [3] translation vector
    translation: jax.Array
