from typing import Literal, Tuple

import jax
from flax.struct import dataclass


AABB = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
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
    stratified: bool
    n_importance: int


@dataclass
class RenderingOptions:
    ray_chunk_size: int
    use_white_bg: bool


@dataclass
class RigidTransformation:
    # [3, 3] rotation matrix
    rotation: jax.Array
    # [3] translation vector
    translation: jax.Array
