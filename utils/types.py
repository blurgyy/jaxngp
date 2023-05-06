import dataclasses
import functools
import json
import math
from pathlib import Path
from typing import Callable, Literal, Tuple

from PIL import Image
import chex
from flax import struct
from flax.struct import dataclass
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import jax.random as jran
import numpy as np
import pydantic
from volrendjax import morton3d_invert, packbits


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

ColmapMatcherType = Literal["Exhaustive", "Sequential"]
LogLevel = Literal["DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"]

DensityAndRGB = Tuple[jax.Array, jax.Array]
RGBColor = Tuple[float, float, float]
RGBColorU8 = Tuple[int, int, int]
FourFloats = Tuple[float, float, float, float]
Matrix4x4 = Tuple[FourFloats, FourFloats, FourFloats, FourFloats]


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
    fx: float
    fy: float

    # principal point
    cx: float
    cy: float

    @property
    def n_pixels(self) -> int:
        return self.H * self.W

    @classmethod
    def from_colmap_txt(cls, txt_path: str | Path) -> "PinholeCamera":
        """
        Example usage:
            cam = PinholeCamera.from_colmap_txt("path/to/txt")
        """
        with open(txt_path, "r") as f:
            lines = f.readlines()
        # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        _, camera_model, width, height, fx, fy, cx, cy = lines[-1].strip().split()
        assert camera_model == "PINHOLE", "invalid camera model, expected PINHOLE, got {}".format(camera_model)
        return cls(
            W=int(width),
            H=int(height),
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
        )


@empty_impl
@dataclass
class RayMarchingOptions:
    # for calculating the length of a minimal ray marching step, the NGP paper uses 1024 (appendix
    # E.1)
    diagonal_n_steps: int

    # whether to fluctuate the first sample along the ray with a tiny perturbation
    perturb: bool

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
@pydantic.dataclasses.dataclass(frozen=True)
class SceneOptions:
    # scale both the scene's camera positions and bounding box with this factor
    world_scale: float

    # scale input images in case they are too large
    image_scale: float

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


@pydantic.dataclasses.dataclass(frozen=True)
class TransformJsonFrame:
    file_path: str
    transform_matrix: Matrix4x4

    # unused, kept for compatibility with the original nerf_synthetic dataset
    rotation: float=0.0

    # unused, kept for compatibility with instant-ngp
    sharpness: float=1e5

    @property
    def transform_matrix_numpy(self) -> np.ndarray:
        return np.asarray(self.transform_matrix)

    @property
    def transform_matrix_jax_array(self) -> jax.Array:
        return jnp.asarray(self.transform_matrix)

    def scale_camera_positions(self, scale: float) -> "TransformJsonFrame":
        new_transform_matrix = self.transform_matrix_numpy
        new_transform_matrix[:3, 3] *= scale
        return dataclasses.replace(
            self,
            transform_matrix=new_transform_matrix.tolist(),
        )


@pydantic.dataclasses.dataclass(frozen=True)
class TransformJsonBase:
    frames: Tuple[TransformJsonFrame, ...]

    # scene's bound, the name `aabb_scale` is for compatibility with instant-ngp (note that
    # instant-ngp requires this value to be a power of 2, other than that a value that can work with
    # instant-ngp will work with this code base as well).
    aabb_scale: float=dataclasses.field(default_factory=lambda: 1., kw_only=True)

    # camera's translation vectors should be scaled with this factor while loading (default value
    # taken from NVLabs/instant-ngp/include/neural-graphics-primitives/nerf_loader.h)
    # NOTE: this value does not affect scene's bounding box
    scale: float=dataclasses.field(default_factory=lambda: 1/3, kw_only=True)

    def scale_camera_positions(self) -> "TransformJsonBase":
        return dataclasses.replace(
            self,
            frames=tuple(map(lambda f: f.scale_camera_positions(self.scale), self.frames)),
        )

    def merge(self, rhs: "TransformJsonBase") -> "TransformJsonBase":
        if rhs is None:
            return self
        # sanity checks, transforms to be merged should have same camera intrinsics
        assert isinstance(rhs, type(self))
        assert all(
            getattr(rhs, attr_name) == getattr(self, attr_name)
            for attr_name in (
                field
                for field in self.__dataclass_fields__
                if field != "frames"
            )
        )
        return dataclasses.replace(
            self,
            frames=self.frames + rhs.frames,
        )

    def make_absolute(self, parent_dir: Path | str) -> "TransformJsonBase":
        parent_dir = Path(parent_dir)
        return dataclasses.replace(
            self,
            frames=tuple(map(
                lambda f: dataclasses.replace(
                    f,
                    file_path=(
                        f.file_path
                        if Path(f.file_path).is_absolute()
                        else parent_dir.joinpath(f.file_path).absolute().as_posix()
                    ),
                ),
                self.frames,
            )),
        )

    def as_json(self, /, indent: int=2) -> str:
        return json.dumps(dataclasses.asdict(self), indent=indent)

    @classmethod
    def from_json(cls, jsonstr: str) -> "TransformJsonBase":
        return cls(**json.loads(jsonstr))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(self.as_json())

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)
        return cls.from_json(path.read_text())


@pydantic.dataclasses.dataclass(frozen=True)
class TransformJsonNeRFSynthetic(TransformJsonBase):
    camera_angle_x: float


@pydantic.dataclasses.dataclass(frozen=True)
class TransformJsonNGP(TransformJsonBase):
    fl_x: float
    fl_y: float
    cx: float
    cy: float

    w: int
    h: int


@dataclass
class ViewMetadata:
    scale: float
    transform: RigidTransformation
    file: Path

    def __post_init__(self):
        assert 0 <= self.scale <= 1, "scale must be in range [0, 1], got {}".format(self.scale)

    @property
    def image_pil(self) -> Image.Image:
        image = Image.open(self.file)
        image = image.resize((int(image.width * self.scale), int(image.height * self.scale)), resample=Image.LANCZOS)
        return image

    @property
    def image_rgba_u8(self) -> jax.Array:
        image = jnp.asarray(self.image_pil)
        if image.shape[-1] == 1:
            image = jnp.concatenate([image] * 3 + [255 * jnp.ones_like(image[..., :1])], axis=-1)
        elif image.shape[-1] == 3:
            image = jnp.concatenate([image, 255 * jnp.ones_like(image[..., :1])], axis=-1)
        chex.assert_axis_dimension(image, -1, 4)
        return image

    @property
    def H(self) -> int:
        return self.image_pil.height

    @property
    def W(self) -> int:
        return self.image_pil.width

    # int, [H*W, 2]: original integer coordinates in range [0, W] for x and [0, H] for y
    @property
    def xys(self) -> jax.Array:
        x, y = jnp.meshgrid(jnp.arange(self.W), jnp.arange(self.H))
        x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        return jnp.concatenate([x, y], axis=-1)

    # float, [H*W, 4]: normalized rgba values in range [0, 1]
    @property
    def rgba_u8(self) -> jax.Array:
        flattened = self.image_rgba_u8.reshape(self.H * self.W, -1)
        chex.assert_axis_dimension(flattened, -1, 4)
        return flattened


@dataclass
class SceneMeta:
    # half width of axis-aligned bounding-box, i.e. aabb's width is `bound*2`
    bound: float

    # the camera model used to render this scene
    camera: PinholeCamera

    frames: Tuple[TransformJsonFrame, ...]=struct.field(pytree_node=False)

    @property
    def cascades(self) -> int:
        return max(1, int(1 + math.ceil(math.log2(self.bound))))

    @property
    def n_pixels(self) -> float:
        return self.camera.n_pixels * len(self.frames)

    # this is the same thing as `dt_gamma` in ashawkey/torch-ngp
    @property
    def stepsize_portion(self) -> float:
        if self.bound > 64:
            return 1e-2
        elif self.bound > 16:
            return 1/128
        elif self.bound > 4:
            return 5e-3
        elif self.bound > 1:
            return 1/256
        else:
            return 0


@dataclass
class SceneData:
    meta: SceneMeta

    # float,[n_pixels, 4], flattened rgb values from loaded images
    all_rgbas_u8: jax.Array

    # float,[n_views, 9+3] each row comprises of R(flattened,9), T(3), from loaded images
    all_transforms: jax.Array


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
    scene_options: SceneOptions=struct.field(pytree_node=False)
    scene_meta: SceneMeta=struct.field(pytree_node=False)

    nerf_fn: Callable=struct.field(pytree_node=False)
    bg_fn: Callable=struct.field(pytree_node=False)

    @classmethod
    def create(cls, *args, **kwargs):
        return super().create(apply_fn=None, *args, **kwargs)

    def __post_init__(self):
        assert self.apply_fn is None

    @functools.partial(jax.jit, static_argnames=["cas", "update_all"])
    def update_ogrid_density(self, KEY: jran.KeyArray, cas: int, update_all: bool) -> "NeRFState":
        G3 = self.raymarch.density_grid_res ** 3
        cas_slice = slice(cas * G3, (cas + 1) * G3)

        decay = .95
        cas_density_grid = self.ogrid.density[cas_slice] * decay

        if update_all:
            M = G3
            indices = jnp.arange(M, dtype=jnp.uint32)
        else:
            M = G3 // 2
            # The first 𝑀/2 cells are sampled uniformly among all cells.
            KEY, key_firsthalf, key_secondhalf = jran.split(KEY, 3)
            indices_firsthalf = jran.choice(key=key_firsthalf, a=jnp.arange(G3, dtype=jnp.uint32), shape=(M//2,), replace=True)  # allow duplicated choices
            # Rejection sampling is used for the remaining samples to restrict selection to cells
            # that are currently occupied.
            # NOTE: Below is just uniformly sampling the occupied cells, not rejection sampling.
            cas_occ_mask = self.ogrid.occ_mask[cas_slice]
            p = cas_occ_mask.astype(jnp.float32)
            indices_secondhalf = jran.choice(key=key_secondhalf, a=jnp.arange(G3, dtype=jnp.uint32), shape=(M//2,), replace=True, p=p)
            indices = jnp.concatenate([indices_firsthalf, indices_secondhalf])

        coordinates = morton3d_invert(indices).astype(jnp.float32)
        coordinates = coordinates / (self.raymarch.density_grid_res - 1) * 2 - 1  # in [-1, 1]
        mip_bound = min(self.scene_meta.bound, 2**cas)
        half_cell_width = mip_bound / self.raymarch.density_grid_res
        coordinates *= mip_bound - half_cell_width  # in [-mip_bound+half_cell_width, mip_bound-half_cell_width]
        # random point inside grid cells
        KEY, key = jran.split(KEY, 2)
        coordinates += jran.uniform(
            key,
            coordinates.shape,
            coordinates.dtype,
            minval=-half_cell_width,
            maxval=half_cell_width,
        )
        new_densities = self.nerf_fn(
            {"params": self.locked_params["nerf"]},
            jax.lax.stop_gradient(coordinates),
            None,
        )
        cas_density_grid = cas_density_grid.at[indices].set(
            jnp.maximum(cas_density_grid[indices], new_densities.ravel())
        )
        new_ogrid = self.ogrid.replace(
            density=self.ogrid.density.at[cas_slice].set(cas_density_grid),
        )
        return self.replace(ogrid=new_ogrid)

    @jax.jit
    def threshold_ogrid(self) -> "NeRFState":
        density_threshold = .01 * self.raymarch.diagonal_n_steps / (2 * min(self.scene_meta.bound, 1) * 3**.5)
        mean_density = jnp.sum(jnp.where(self.ogrid.density > 0, self.ogrid.density, 0)) / jnp.sum(jnp.where(self.ogrid.density > 0, 1, 0))
        density_threshold = jnp.minimum(density_threshold, mean_density)
        occupied_mask, occupancy_bitfield = packbits(
            density_threshold=density_threshold,
            density_grid=self.ogrid.density,
        )
        new_ogrid = self.ogrid.replace(
            occ_mask=occupied_mask,
            occupancy=occupancy_bitfield,
        )
        return self.replace(ogrid=new_ogrid)

    @property
    def use_background_model(self) -> bool:
        return self.scene_options.with_bg and self.params.get("bg") is not None

    @property
    def locked_params(self):
        return jax.lax.stop_gradient(self.params)

    @property
    def update_ogrid_interval(self) -> int:
        return min(2 ** (int(self.step) // 2048 + 4), 128)

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
