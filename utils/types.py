from typing import Literal, Tuple, Union

import jax
from flax.struct import dataclass


AABB = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
DensityAndRGB = Tuple[jax.Array, jax.Array]
LogLevel = Literal["DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"]
RGBColor = Tuple[float, float, float]


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
    # background color for transparent parts of the image, has no effect if `random_bg` is True
    bg: RGBColor
    # ignore `bg` specification and use random color for transparent parts of the image
    random_bg: bool


@dataclass
class RigidTransformation:
    # [3, 3] rotation matrix
    rotation: jax.Array
    # [3] translation vector
    translation: jax.Array
