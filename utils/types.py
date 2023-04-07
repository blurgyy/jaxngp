from typing import Literal, Tuple

from flax.struct import dataclass
import jax


PositionalEncodingType = Literal["identity", "frequency", "hashgrid"]
DirectionalEncodingType = Literal["identity", "sh", "shcuda"]
ActivationType = Literal[
    "exponential",
    "relu",
    "sigmoid",
    "truncated_exponential",
    "thresholded_exponential",
    "truncated_thresholded_exponential",
]

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
