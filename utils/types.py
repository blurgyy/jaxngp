import dataclasses
import functools
import json
import math
from pathlib import Path
from typing import Callable, List, Literal, Tuple, Type

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
from tqdm import tqdm
from volrendjax import morton3d_invert, packbits

from ._constants import tqdm_format


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
TransformsProvider = Literal["loaded", "orbit"]

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


def replace_impl(clz):
    if "__dataclass_fields__" not in clz.__dict__:
        raise TypeError("class `{}` is not a dataclass".format(clz.__name__))

    fields = clz.__dict__["__dataclass_fields__"]

    def replace_fn(self, /, **kwargs) -> Type[clz]:
        for k in kwargs.keys():
            if k not in fields:
                raise RuntimeError("class `{}` does not have a field with name '{}'".format(clz.__name__, k))
        ret = dataclasses.replace(self, **kwargs)
        return ret

    setattr(clz, "replace", replace_fn)
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

    # uint32, indices of the grids that are alive (trainable)
    alive_indices: jax.Array

    # list of `int`s, upper bound of each cascade
    alive_indices_offset: List[int]=struct.field(pytree_node=False)

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
        G3 = grid_resolution**3
        n_grids = cascades * G3
        occupancy = 255 * jnp.ones(
            shape=(n_grids // 8,),  # each bit is an occupancy value
            dtype=jnp.uint8,
        )
        density = jnp.zeros(
            shape=(n_grids,),
            dtype=jnp.float32,
        )
        occ_mask = jnp.zeros(
            shape=(n_grids,),
            dtype=jnp.bool_,
        )
        return cls(
            density=density,
            occ_mask=occ_mask,
            occupancy=occupancy,
            alive_indices=jnp.arange(n_grids, dtype=jnp.uint32),
            alive_indices_offset=np.cumsum([0] + [G3] * cascades).tolist(),
        )


@empty_impl
@dataclass
class NeRFBatchConfig:
    mean_samples_per_ray: int

    running_mean_effective_samples_per_ray: float
    running_mean_samples_per_ray: float

    n_rays: int

    @property
    def mean_effective_samples_per_ray(self) -> int:
        return int(self.running_mean_effective_samples_per_ray + 0.5)

    @classmethod
    def create(cls, /, mean_effective_samples_per_ray: int, mean_samples_per_ray: int, n_rays: int) -> "NeRFBatchConfig":
        return cls(
            mean_samples_per_ray=mean_samples_per_ray,
            running_mean_effective_samples_per_ray=mean_effective_samples_per_ray,
            running_mean_samples_per_ray=mean_samples_per_ray,
            n_rays=n_rays,
        )

    def update(self, /, new_measured_batch_size: int, new_measured_batch_size_before_compaction: int) -> "NeRFBatchConfig":
        decay = .95
        return self.replace(
            running_mean_effective_samples_per_ray=self.running_mean_effective_samples_per_ray * decay + (1 - decay) * new_measured_batch_size / self.n_rays,
            running_mean_samples_per_ray=self.running_mean_samples_per_ray * decay + (1 - decay) * new_measured_batch_size_before_compaction / self.n_rays,
        )

    def commit(self, total_samples: int) -> "NeRFBatchConfig":
        new_mean_samples_per_ray=int(self.running_mean_samples_per_ray + 1.5)
        new_n_rays=int(total_samples // new_mean_samples_per_ray)
        return self.replace(
            mean_samples_per_ray=new_mean_samples_per_ray,
            n_rays=new_n_rays,
        )


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

    near: float=0.1

    @property
    def n_pixels(self) -> int:
        return self.H * self.W

    @property
    def K_numpy(self) -> np.ndarray:
        return np.asarray([
            [self.fx,      0., self.cx],
            [     0., self.fy, self.cy],
            [     0.,      0.,      1.]
        ])

    @property
    def K(self) -> jax.Array:
        return jnp.asarray(self.K_numpy)

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

    def scale_resolution(self, scale: int | float) -> "PinholeCamera":
        return self.replace(
            W=int(self.W * scale),
            H=int(self.H * scale),
            fx=self.fx * scale,
            fy=self.fy * scale,
            cx=self.cx * scale,
            cy=self.cy * scale,
        )

    def scale_world(self, scale: int | float) -> "PinholeCamera":
        return self.replace(near=self.near * scale)


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
@replace_impl
@pydantic.dataclasses.dataclass(frozen=True)
class CameraOverrideOptions:
    width: int | None=None
    height: int | None=None
    focal: float | None=None
    near: float=0.1

    def __post_init__(self):
        if self.width is None and self.height is None:
            return
        if int(self.width is not None) + int(self.height is not None) == 1:
            side = self.width if self.width is not None else self.height
            self.__init__(
                width=side,
                height=side,
                focal=self.focal,
                near=self.near,
            )
        assert self.width > 0 and self.height > 0
        if self.enabled and self.focal is None:
            self.__init__(
                width=self.width,
                height=self.height,
                focal=min(self.width, self.height),
                near=self.near,
            )

    @property
    def enabled(self) -> bool:
        return (
            self.width is not None
            and self.height is not None
        )

    @property
    def camera(self) -> PinholeCamera:
        assert self.enabled
        return PinholeCamera(
            W=self.width,
            H=self.height,
            fx=self.focal,
            fy=self.focal,
            cx=self.width / 2,
            cy=self.height / 2,
            near=self.near,
        )


@empty_impl
@replace_impl
@pydantic.dataclasses.dataclass(frozen=True)
class SceneOptions:
    # images with sharpness lower than this value will be discarded
    sharpness_threshold: float

    # scale both the scene's camera positions and bounding box with this factor
    world_scale: float

    # scale input images in case they are too large, camera intrinsics are also scaled to match the
    # updated image resolution.
    resolution_scale: float


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


@replace_impl
@pydantic.dataclasses.dataclass(frozen=True)
class TransformJsonFrame:
    file_path: Path | None
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
        return self.replace(
            transform_matrix=new_transform_matrix.tolist(),
        )


@replace_impl
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
    scale: float=dataclasses.field(default_factory=lambda: 2/3, kw_only=True)

    bg: bool=dataclasses.field(default_factory=lambda: False, kw_only=True)

    def scale_camera_positions(self) -> "TransformJsonBase":
        return self.replace(
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
        return self.replace(frames=self.frames + rhs.frames)

    def make_absolute(self, parent_dir: Path | str) -> "TransformJsonBase":
        parent_dir = Path(parent_dir)
        return self.replace(
            frames=tuple(map(
                lambda f: f.replace(
                    file_path=(
                        f.file_path
                        if f.file_path.is_absolute()
                        else parent_dir.joinpath(f.file_path).absolute().as_posix()
                    ),
                ),
                self.frames,
            )),
        )

    def as_json(self, /, indent: int=2) -> str:
        d = dataclasses.asdict(self)
        d = jax.tree_util.tree_map(lambda x: x.as_posix() if isinstance(x, Path) else x, d)
        return json.dumps(d, indent=indent)

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


@replace_impl
@pydantic.dataclasses.dataclass(frozen=True)
class TransformJsonNeRFSynthetic(TransformJsonBase):
    camera_angle_x: float


@replace_impl
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
class OrbitTrajectoryOptions:
    # cameras' distance to the orbiting axis
    radius: float=1

    # lowest height of generated trajectory
    low: float=0.0

    # highest height of generated trajectory
    high: float=0.8

    # how many frames should be rendered per orbit
    n_frames_per_orbit: int=144

    n_orbit: int=2

    # all orbiting cameras will look at this point
    centroid: Tuple[float, float, float]=(0., 0., 0.1)

    @property
    def n_frames(self) -> int:
        return self.n_frames_per_orbit * self.n_orbit


# scene's metadata (computed from SceneOptions and TransformJson)
@dataclass
class SceneMeta:
    # half width of axis-aligned bounding-box, i.e. aabb's width is `bound*2`
    bound: float

    # whether the scene should be modeled with a background that is not part of the scene geometry
    bg: bool

    # the camera model used to render this scene
    camera: PinholeCamera

    frames: Tuple[TransformJsonFrame, ...]=struct.field(pytree_node=False)

    @property
    def cascades(self) -> int:
        return max(1, int(1 + math.ceil(math.log2(self.bound))))

    @property
    def n_pixels(self) -> float:
        return self.camera.n_pixels * len(self.frames)

    @property
    def sharpness_range(self) -> float:
        return functools.reduce(
            lambda prev, frame: (min(prev[0], frame.sharpness), max(prev[1], frame.sharpness)),
            self.frames,
            (1e9, -1e9),
        )

    # this is the same thing as `dt_gamma` in ashawkey/torch-ngp
    @property
    def stepsize_portion(self) -> float:
        if self.bound >= 64:
            return 1/128
        elif self.bound >= 16:
            return 6e-3
        elif self.bound >= 4:
            return 5e-3
        elif self.bound > 1:
            return 1/256
        else:
            return 0

    def make_frames_with_orbiting_trajectory(
        self,
        opts: OrbitTrajectoryOptions,
    ) -> "SceneMeta":
        assert isinstance(opts, OrbitTrajectoryOptions)

        thetas = np.linspace(0, opts.n_orbit * 2 * np.pi, opts.n_frames + 1)[:-1]
        xs = np.asarray(tuple(map(np.cos, thetas))) * opts.radius
        ys = np.asarray(tuple(map(np.sin, thetas))) * opts.radius
        elevation_range = opts.high - opts.low
        mid_elevation = opts.low + .5 * elevation_range
        zs = mid_elevation + .5 * elevation_range * np.sin(np.linspace(0, 2 * np.pi, opts.n_frames + 1)[:-1])
        xyzs = np.stack([xs, ys, zs]).T

        view_dirs = (jnp.asarray(opts.centroid) - xyzs) / np.linalg.norm(xyzs, axis=-1, keepdims=True)
        right_dirs = np.stack([-np.sin(thetas), np.cos(thetas), np.zeros_like(thetas)]).T
        up_dirs = -np.cross(view_dirs, right_dirs)
        up_dirs = up_dirs / np.linalg.norm(up_dirs, axis=-1, keepdims=True)

        rot_cws = np.concatenate([right_dirs[..., None], up_dirs[..., None], -view_dirs[..., None]], axis=-1)

        frames = tuple(map(
            lambda rot_cw, t_cw: TransformJsonFrame(
                file_path=None,
                transform_matrix=np.concatenate(
                    [np.concatenate([rot_cw, t_cw.reshape(3, 1)], axis=-1), np.ones((1, 4))],
                    axis=0,
                ).tolist(),
            ),
            rot_cws,
            xyzs,
        ))

        return self.replace(frames=frames)


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
        G3 = self.raymarch.density_grid_res**3
        cas_slice = slice(cas * G3, (cas + 1) * G3)
        cas_alive_indices = self.ogrid.alive_indices[self.ogrid.alive_indices_offset[cas]:self.ogrid.alive_indices_offset[cas+1]]
        aligned_indices = cas_alive_indices % G3  # values are in range [0, G3)
        n_grids = aligned_indices.shape[0]

        decay = .95
        cas_occ_mask = self.ogrid.occ_mask[cas_slice]
        cas_density_grid = self.ogrid.density[cas_slice].at[aligned_indices].set(self.ogrid.density[cas_slice][aligned_indices] * decay)

        if update_all:
            # During the first 256 training steps, we sample 𝑀 = 𝐾 · 128^{3} cells uniformly without
            # repetition.
            cas_updated_indices = aligned_indices
        else:
            M = max(1, n_grids // 2)
            # The first 𝑀/2 cells are sampled uniformly among all cells.
            KEY, key_firsthalf, key_secondhalf = jran.split(KEY, 3)
            indices_firsthalf = jran.choice(
                key=key_firsthalf,
                a=aligned_indices,
                shape=(max(1, M//2),),
                replace=True,  # allow duplicated choices
            )
            # Rejection sampling is used for the remaining samples to restrict selection to cells
            # that are currently occupied.
            # NOTE: Below is just uniformly sampling the occupied cells, not rejection sampling.
            cas_alive_occ_mask = cas_occ_mask[aligned_indices]
            indices_secondhalf = jran.choice(
                key=key_secondhalf,
                a=aligned_indices,
                shape=(max(1, M//2),),
                replace=True,  # allow duplicated choices
                p=cas_alive_occ_mask.astype(jnp.float32),  # only care about occupied grids
            )
            cas_updated_indices = jnp.concatenate([indices_firsthalf, indices_secondhalf])

        coordinates = morton3d_invert(cas_updated_indices).astype(jnp.float32)
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
        cas_density_grid = cas_density_grid.at[cas_updated_indices].set(
            jnp.maximum(cas_density_grid[cas_updated_indices], new_densities.ravel())
        )
        new_ogrid = self.ogrid.replace(
            density=self.ogrid.density.at[cas_slice].set(cas_density_grid),
        )
        return self.replace(ogrid=new_ogrid)

    @jax.jit
    def threshold_ogrid(self) -> "NeRFState":
        density_threshold = .01 * self.raymarch.diagonal_n_steps / (2 * min(self.scene_meta.bound, 1) * 3**.5)
        mean_density = self.ogrid.density[self.ogrid.alive_indices[:self.ogrid.alive_indices_offset[1]]].mean()
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

    def mark_untrained_density_grid(self) -> "NeRFState":
        G = self.raymarch.density_grid_res
        G3 = G*G*G
        n_grids = self.scene_meta.cascades * G3
        all_indices = jnp.arange(n_grids, dtype=jnp.uint32)
        level, pos_idcs = all_indices // G3, all_indices % G3
        mip_bound = jnp.minimum(2 ** level, self.scene_meta.bound).astype(jnp.float32)
        cell_width = 2 * mip_bound / G
        xyzs = morton3d_invert(pos_idcs).astype(jnp.float32)  # [G3, 3]
        xyzs /= G  # in range [0, 1)
        xyzs -= 0.5  # in range [-0.5, 0.5)
        xyzs *= 2 * mip_bound[:, None]  # in range [-mip_bound, mip_bound)
        vertex_offsets = cell_width[:, None, None] * jnp.asarray([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ], dtype=jnp.float32)
        grid_vertices = xyzs[:, None, :] + vertex_offsets
        alive_marker = jnp.zeros(n_grids, dtype=jnp.bool_)

        @jax.jit
        def mark_untrained_density_grid_single_frame(
            alive_marker: jax.Array,
            transform_cw: jax.Array,
        ):
            rot_cw, t_cw = transform_cw[:3, :3], transform_cw[:3, 3]
            # p_world, p_cam, T: [3, 1]
            # rot_cw: [3, 3]
            # p_world = rot_cw @ p_cam + t_cw
            #   => p_cam = rot_cw.T @ (p_world - t_cw)
            p_aligned = grid_vertices - t_cw
            p_cam = (p_aligned[..., None, :] * rot_cw.T).sum(-1)

            # camera looks along the -z axis
            in_front_of_camera = p_cam[..., -1] < -self.scene_meta.camera.near - 1e-4

            uvz = (p_cam[..., None, :] * self.scene_meta.camera.K).sum(-1)
            uvz /= uvz[..., -1:]
            uv = uvz[..., :2] / jnp.asarray([self.scene_meta.camera.W, self.scene_meta.camera.H], dtype=jnp.float32)
            within_frame_range = (uv >= 0.) & (uv < 1.)
            within_frame_range = (
                within_frame_range  # shape is [n_grids, 8, 2]
                    .all(axis=-1)  # u and v must both be within frame
            )
            visible_by_camera = (in_front_of_camera & within_frame_range).any(axis=-1)  # grid should be trained if any of its 8 vertices is visible
            return alive_marker | visible_by_camera

        # cam_t = np.asarray(list(map(
        #     lambda frame: frame.transform_matrix_numpy[:3, 3],
        #     self.scene_meta.frames,
        # )))
        # np.savetxt("cams.xyz", cam_t)

        for frame in (pbar := tqdm(self.scene_meta.frames, desc="marking trainable grids".format(n_grids), bar_format=tqdm_format)):
            alive_marker = mark_untrained_density_grid_single_frame(alive_marker, frame.transform_matrix_jax_array)
            n_alive_grids = alive_marker.sum()
            ratio_trainable = n_alive_grids / n_grids
            pbar.set_description_str("marked {}/{} ({:.2f}%) grids as trainable".format(n_alive_grids, n_grids, ratio_trainable * 100))

        marked_density = jnp.where(alive_marker, 1e-15, -1.)
        marked_occ_mask, marked_occupancy = packbits(
            density_threshold=-.5,
            density_grid=marked_density
        )

        # rgb = jnp.stack([~marked_occ_mask, jnp.zeros_like(marked_occ_mask, dtype=jnp.float32), marked_occ_mask]).T
        # xyzrgb = np.asarray(jnp.concatenate([xyzs, rgb], axis=-1))
        # np.savetxt("blue_for_trainable.txt", xyzrgb)
        # np.savetxt("trainable.txt", xyzrgb[np.where(marked_occ_mask)])
        # np.savetxt("untrainable.txt", xyzrgb[np.where(~marked_occ_mask)])

        return self.replace(
            ogrid=self.ogrid.replace(
                density=marked_density,
                occ_mask=marked_occ_mask,
                occupancy=marked_occupancy,
                alive_indices=all_indices[alive_marker],
                alive_indices_offset=np.cumsum([0] + list(map(
                    lambda cas_alive_marker: int(cas_alive_marker.sum()),
                    jnp.split(alive_marker, self.scene_meta.cascades),
                ))).tolist(),
            ),
        )

    @property
    def use_background_model(self) -> bool:
        return self.scene_meta.bg and self.params.get("bg") is not None

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
    def should_commit_batch_config(self) -> bool:
        return (
            self.step > 0
            and self.step % 16 == 0
        )

    def update_batch_config(self, /, new_measured_batch_size: int, new_measured_batch_size_before_compaction: int) -> "NeRFState":
        return self.replace(
            batch_config=self.batch_config.update(
                new_measured_batch_size=new_measured_batch_size,
                new_measured_batch_size_before_compaction=new_measured_batch_size_before_compaction,
            )
        )

    @property
    def should_write_batch_metrics(self) -> bool:
        return self.step % 16 == 0
