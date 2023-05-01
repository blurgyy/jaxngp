from pathlib import Path
from typing import Callable, Literal, Tuple

import chex
from flax import struct
from flax.struct import dataclass
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp


PositionalEncodingType = Literal["identity", "frequency", "hashgrid"]
DirectionalEncodingType = Literal["identity", "sh", "shcuda"]
EncodingType = Literal[PositionalEncodingType, DirectionalEncodingType]
ActivationType = Literal[
    "exponential",
    "relu",
    "sigmoid",
    "thresholded_exponential",
    "truncated_exponential",
    "truncated_thresholded_exponential",
]

DensityAndRGB = Tuple[jax.Array, jax.Array]
LogLevel = Literal["DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"]
RGBColor = Tuple[float, float, float]


def empty_impl(clz):
    if "__dataclass_fields__" not in clz.__dict__:
        raise TypeError("class `{}` is not a dataclass".format(clz.__name__))

    fields = clz.__dict__["__dataclass_fields__"]

    def empty_fn(cls, /, **kwargs):
        """
        Create an empty instance of the given class, with untransformed fields set to given values.
        """
        for field_name, annotation in fields.items():
            if field_name not in kwargs:
                kwargs[field_name] = getattr(annotation.type, "empty", lambda: None)()
        return cls(**kwargs)

    setattr(clz, "empty", classmethod(empty_fn))
    return clz


@empty_impl
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


@empty_impl
@dataclass
class NeRFBatchConfig:
    mean_effective_samples_per_ray: int
    mean_samples_per_ray: int
    n_rays: int

    @property
    def estimated_batch_size(self):
        return self.n_rays * self.mean_samples_per_ray


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

    def scaled(self, factor: float) -> "PinholeCamera":
        "same focal length, different resolution"
        return self\
            .replace(H=self.H // factor)\
            .replace(W=self.W // factor)

    def zoomed(self, factor: float) -> "PinholeCamera":
        "same resolution, different focal length"
        return self.replace(focal=self.focal * factor)


@empty_impl
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


@empty_impl
@dataclass
class RenderingOptions:
    # background color for transparent parts of the image, has no effect if `random_bg` is True
    bg: RGBColor

    # ignore `bg` specification and use random color for transparent parts of the image
    random_bg: bool


@empty_impl
@dataclass
class SceneOptions:
    # half width of axis-aligned bounding-box, i.e. aabb's width is `bound*2`
    bound: float

    # scale camera positions with this scalar
    scale: float

    # whether the scene has a background
    with_bg: bool


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


@empty_impl
class NeRFState(TrainState):
    # WARN:
    #   do not annotate fields with jax.Array as members with flax.truct.field(pytree_node=False),
    #   otherwise wierd issues happen, e.g. jax tracer leak, array-to-boolean conversion exception
    #   while calling a jitted function with no helpful traceback.
    ogrid: OccupancyDensityGrid

    # WARN:
    #   annotating batch_config with flax.struct.field(pytree_node=False) halves GPU utilization by
    #   2x, consequently halving training speed by 2x as well.
    #   ... why?
    batch_config: NeRFBatchConfig

    raymarch: RayMarchingOptions=struct.field(pytree_node=False)
    render: RenderingOptions=struct.field(pytree_node=False)
    scene: SceneOptions=struct.field(pytree_node=False)

    nerf_fn: Callable=struct.field(pytree_node=False)
    bg_fn: Callable=struct.field(pytree_node=False)

    @classmethod
    def create(cls, *args, **kwargs):
        return super().create(apply_fn=None, *args, **kwargs)

    def __post_init__(self):
        assert self.apply_fn is None

    @property
    def use_background_model(self) -> bool:
        return self.scene.with_bg and self.params.get("bg") is not None

    @property
    def locked_params(self):
        return jax.lax.stop_gradient(self.params)

    @property
    def update_ogrid_interval(self) -> int:
        return min(2 ** (int(self.step) // 2048 + 4), 512)

    @property
    def should_call_update_ogrid(self) -> bool:
        return (
            self.step < 256
            or (
                self.step > 0
                and self.step % self.update_ogrid_interval == 0
            )
        )

    @property
    def should_update_all_ogrid_cells(self) -> bool:
        return self.step < 256

    @property
    def should_update_batch_config(self) -> bool:
        return (
            self.step > 0
            and self.step % 16 == 0
        )

    @property
    def should_write_batch_metrics(self) -> bool:
        return self.step % 16 == 0
