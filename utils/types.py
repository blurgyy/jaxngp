from typing import Literal, Tuple

import chex
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


@dataclass
class PinholeCamera:
    # resolutions
    W: int
    H: int

    # focal length
    focal: float


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
class RigidTransformation:
    # [3, 3] rotation matrix
    rotation: jax.Array

    # [3] translation vector
    translation: jax.Array

    def __post_init__(self):
        chex.assert_shape([self.rotation, self.translation], [(3, 3), (3,)])

