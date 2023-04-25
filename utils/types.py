import logging
from pathlib import Path
from typing import Literal, Tuple

import chex
from flax.metrics import tensorboard
from flax.struct import dataclass
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp


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

DensityAndRGB = Tuple[jax.Array, jax.Array]
LogLevel = Literal["DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"]
RGBColor = Tuple[float, float, float]


class Logger(logging.Logger):
    tb: tensorboard.SummaryWriter


@dataclass
class OccupancyDensityGrid:
    # float32, full-precision density values
    density: jax.Array
    # bool, a non-compact representation of the occupancy bitfield
    occ_mask: jax.Array
    # uint8, each bit is an occupancy value of a grid cell
    occupancy: jax.Array

    @classmethod
    def create(cls, cascades: int, grid_resolution: int=128):
        """
        Inputs:
            cascades: number of cascades, paper: 𝐾 = 1 for all synthetic NeRF scenes (single grid)
                      and 𝐾 ∈ [1, 5] for larger real-world scenes (up to 5 grids, depending on scene
                      size)
            grid_resolution: resolution of the occupancy grid, the NGP paper uses 128.

        Example usage:
            ogrid = OccupancyDensityGrid.create(cascades=5, grid_resolution=128)
        """
        occupancy = 255 * jnp.ones(
            shape=(cascades*grid_resolution**3 // 8,),  # each bit is an occupancy value
            dtype=jnp.uint8,
        )
        density = jnp.zeros(
            shape=(cascades*grid_resolution**3,),
            dtype=jnp.float32,
        )
        occ_mask = jnp.zeros(
            shape=(cascades*grid_resolution**3,),
            dtype=jnp.bool_,
        )
        return cls(density=density, occ_mask=occ_mask, occupancy=occupancy)


@dataclass
class NeRFBatchConfig:
    mean_effective_samples_per_ray: int
    mean_samples_per_ray: int
    n_rays: int

    @property
    def estimated_batch_size(self):
        return self.n_rays * self.mean_samples_per_ray


class NeRFTrainState(TrainState):
    ogrid: OccupancyDensityGrid
    batch_config: NeRFBatchConfig

    @property
    def update_ogrid_interval(self):
        return min(2 ** (int(self.step) // 2048 + 4), 512)

    @property
    def should_call_update_ogrid(self):
        return (
            int(self.step) < 256
            or (
                int(self.step) > 0
                and int(self.step) % self.update_ogrid_interval == 0
            )
        )

    @property
    def should_update_all_ogrid_cells(self):
        return int(self.step) < 256

    @property
    def should_update_batch_config(self):
        return (
            int(self.step) > 0
            and int(self.step) % 16 == 0
        )

    @property
    def should_write_batch_metrics(self):
        return self.step % 16 == 0


@dataclass
class PinholeCamera:
    # resolutions
    W: int
    H: int

    # focal length
    focal: float

    @property
    def n_pixels(self) -> int:
        return self.H * self.W


@dataclass
class RayMarchingOptions:
    # for calculating the length of a minimal ray marching step, the NGP paper uses 1024 (appendix
    # E.1)
    diagonal_n_steps: int

    # whether to fluctuate the first sample along the ray with a tiny perturbation
    perturb: bool

    # this is the same thing as `dt_gamma` in ashawkey/torch-ngp
    stepsize_portion: float

    # resolution for the auxiliary density/occupancy grid, the NGP paper uses 128 (appendix E.2)
    density_grid_res: int


@dataclass
class RenderingOptions:
    # background color for transparent parts of the image, has no effect if `random_bg` is True
    bg: RGBColor

    # ignore `bg` specification and use random color for transparent parts of the image
    random_bg: bool


@dataclass
class SceneOptions:
    # half width of axis-aligned bounding-box, i.e. aabb's width is `bound*2`
    bound: float

    # scale camera positions with this scalar
    scale: float


@dataclass
class RigidTransformation:
    # [3, 3] rotation matrix
    rotation: jax.Array

    # [3] translation vector
    translation: jax.Array

    def __post_init__(self):
        chex.assert_shape([self.rotation, self.translation], [(3, 3), (3,)])


@dataclass
class ImageMetadata:
    H: int
    W: int
    xys: jax.Array  # int,[H*W, 2]: original integer coordinates in range [0, W] for x and [0, H] for y
    uvs: jax.Array  # float,[H*W, 2]: normalized coordinates in range [0, 1]
    rgbs: jax.Array  # float,[H*W, 3]: normalized rgb values in range [0, 1]


@dataclass
class ViewMetadata:
    H: int
    W: int
    xys: jax.Array  # int,[H*W, 2]: original integer coordinates in range [0, W] for x and [0, H] for y
    rgbas: jax.Array  # float,[H*W, 4]: normalized rgb values in range [0, 1]
    transform: RigidTransformation
    file: Path

    @property
    def image_rgba(self) -> jax.Array:
        return self.rgbas.reshape(self.H, self.W, 4)


# TODO:
#   Make this `camera`'s H, W configurable and resize loaded images accordingly (specified H,W
#   must have same aspect ratio as the loaded images).
#   For now it's just read from the dataset.
@dataclass
class SceneMetadata:
    camera: PinholeCamera  # the camera model used to render this scene
    all_xys: jax.Array  # int,[n_pixels, 2], flattened xy coordinates from loaded images
    all_rgbas: jax.Array  # float,[n_pixels, 4], flattened rgb values from loaded images
    all_transforms: jax.Array  # float,[n_views, 9+3] each row comprises of R(flattened,9), T(3), from loaded images


@dataclass
class RenderedImage:
    bg: jax.Array
    rgb: jax.Array
    depth: jax.Array
