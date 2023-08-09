from concurrent.futures import ThreadPoolExecutor
import dataclasses
import functools
import json
import math
from pathlib import Path
from typing import Callable, List, Literal, Tuple, Type
from typing_extensions import assert_never

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


CameraModelType = Literal[
    "SIMPLE_PINHOLE",
    "PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "OPENCV",
    "OPENCV_FISHEYE",
]
PositionalEncodingType = Literal["identity", "frequency", "hashgrid", "tcnn-hashgrid"]
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

    def mean_density_up_to_cascade(self, cas: int) -> float | jax.Array:
        return self.density[self.alive_indices[:self.alive_indices_offset[cas]]].mean()


@dataclass
class Camera:
    # resolutions
    width: int=struct.field(pytree_node=False)
    height: int=struct.field(pytree_node=False)

    # focal length
    fx: float=struct.field(pytree_node=False)
    fy: float=struct.field(pytree_node=False)

    # principal point
    cx: float=struct.field(pytree_node=False)
    cy: float=struct.field(pytree_node=False)

    near: float=struct.field(pytree_node=False)

    # distortion parameters
    k1: float=struct.field(default=0.0, pytree_node=False)
    k2: float=struct.field(default=0.0, pytree_node=False)
    k3: float=struct.field(default=0.0, pytree_node=False)
    k4: float=struct.field(default=0.0, pytree_node=False)
    p1: float=struct.field(default=0.0, pytree_node=False)
    p2: float=struct.field(default=0.0, pytree_node=False)

    model: CameraModelType=struct.field(default="OPENCV", pytree_node=False)

    @property
    def has_distortion(self) -> bool:
        return (
            True
            or self.k1 != 0.0
            or self.k2 != 0.0
            or self.k3 != 0.0
            or self.k4 != 0.0
            or self.p1 != 0.0
            or self.p2 != 0.0
        )

    @property
    def _type(self) -> Literal["PERSPECTIVE", "FISHEYE"]:
        if "fisheye" in self.model.lower():
            return "FISHEYE"
        else:
            return "PERSPECTIVE"

    @property
    def n_pixels(self) -> int:
        return self.height * self.width

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
    def from_colmap_txt(cls, txt_path: str | Path) -> "Camera":
        """Initialize a camera from a colmap's TXT format model.
        References of camera parameters from colmap (search for `InitializeParamsInfo`):
            <https://github.com/colmap/colmap/blob/fac2fa6217a1f5498830769d64861b54c67009dc/src/colmap/base/camera_models.h#L778>

        Example usage:
            cam = PinholeCamera.from_colmap_txt("path/to/txt")
        """
        with open(txt_path, "r") as f:
            lines = f.readlines()
        cam_line = lines[-1].strip()
        cam_desc = cam_line.split()
        _front = 0
        def next_descs(cnt: int) -> Tuple[str, ...]:
            nonlocal _front
            rear = _front + cnt
            ret = cam_desc[_front:rear]
            _front = rear
            return ret[0] if cnt == 1 else ret
        assert int(next_descs(1)) == 1, "creating scenes with multiple cameras is not supported"
        camera_model: CameraModelType = next_descs(1)
        width, height = next_descs(2)
        k1, k2, k3, k4, p1, p2 = [0.] * 6
        if camera_model == "SIMPLE_PINHOLE":
            f, cx, cy = next_descs(3)
            fx = fy = f
        elif camera_model == "SIMPLE_RADIAL":
            f, cx, cy, k1 = next_descs(4)
            fx = fy = f
        elif camera_model == "RADIAL":
            f, cx, cy, k1, k2 = next_descs(5)
            fx = fy = f
        else:
            fx, fy, cx, cy = next_descs(4)
            if camera_model == "PINHOLE":
                pass
            elif camera_model == "OPENCV":
                k1, k2, p1, p2 = next_descs(4)
            elif camera_model == "OPENCV_FISHEYE":
                k1, k2, k3, k4 = next_descs(4)
            else:
                assert_never(camera_model)
        return cls(
            width=int(width),
            height=int(height),
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            near=0.,
            k1=float(k1),
            k2=float(k2),
            k3=float(k3),
            k4=float(k4),
            p1=float(p1),
            p2=float(p2),
        )

    def scale_resolution(self, scale: int | float) -> "Camera":
        return self.replace(
            width=int(self.width * scale),
            height=int(self.height * scale),
            fx=self.fx * scale,
            fy=self.fy * scale,
            cx=self.cx * scale,
            cy=self.cy * scale,
        )

    def distort(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Computes distorted coords.
        REF:
            * <https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html>
            * <https://en.wikipedia.org/wiki/Distortion_%28optics%29>
            * <https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html>

        Inputs:
            x, y `float`: normalized undistorted coordinates

        Returns:
            x, y `float`: distorted coordinates
        """
        k1, k2, k3, k4, = self.k1, self.k2, self.k3, self.k4
        xx, yy = jnp.square(x), jnp.square(y)
        rr = xx + yy
        if self._type == "FISHEYE":
            r = jnp.sqrt(rr)
            theta = jnp.arctan(r)
            thth = theta * theta
            thetad = theta * (1. + thth * (k1 + thth * (k2 + thth * (k3 + thth * k4))))
            dist = thetad / r - 1.
            dx, dy = (
                jnp.where(r < 1e-15, 0., x * dist),
                jnp.where(r < 1e-15, 0., y * dist),
            )
        else:
            radial = rr * (k1 + rr * (k2 + rr * (k3 + rr * k4)))

            # radial distort
            dx, dy = x * radial, y * radial

            p1, p2 = self.p1, self.p2

            # tangential distort
            xy = x * y
            dx += 2 * p1 * xy + p2 * (rr + 2 * xx)
            dx += 2 * p2 * xy + p1 * (rr + 2 * xx)

        return x + dx, y + dy

    def undistort(self, x: jax.Array, y: jax.Array, eps: float=1e-3, max_iterations: int=10) -> jax.Array:
        """Computes undistorted coords.
        REF:
            * <https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/camera_utils.py#L477-L509>
            * <https://github.com/nerfstudio-project/nerfstudio/blob/004d8ca9d24b294b1877d4d5599879c4ce812bc7/nerfstudio/cameras/camera_utils.py#L411-L448>

        Inputs:
            x, y `float`: normalized (x_normalized = (x + cx) / fx) distorted coordinates
            eps `float`: epsilon for the convergence
            max_iterations `int`: maximum number of iterations to perform

        Returns:
            x, y `float`: undistorted coordinates
        """
        # the original distorted coordinates
        xd, yd = x.copy(), y.copy()

        @jax.jit
        def compute_residual_and_jacobian(x, y) -> Tuple[Tuple[jax.Array, jax.Array], Tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
            """Auxiliary function of radial_and_tangential_undistort() that computes residuals and
            jacobians.
            REF:
                * <https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/camera_utils.py#L427-L474>
                * <https://github.com/nerfstudio-project/nerfstudio/blob/004d8ca9d24b294b1877d4d5599879c4ce812bc7/nerfstudio/cameras/camera_utils.py#L345-L407>

            Inputs:
                x: The updated x coordinates.
                y: The updated y coordinates.

            Returns:
                The residuals (fx, fy) and jacobians (fx_x, fx_y, fy_x, fy_y).
            """
            k1, k2, k3, k4 = self.k1, self.k2, self.k3, self.k4
            p1, p2 = self.p1, self.p2
            # let r(x, y) = x^2 + y^2;
            #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3 +
            #                   k4 * r(x, y)^4;
            r = x * x + y * y
            d = 1.0 + r * (k1 + r * (k2 + r * (k3 + r * k4)))

            # The perfect projection is:
            # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
            # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
            #
            # Let's define
            #
            # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
            # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
            #
            # We are looking for a solution that satisfies
            # fx(x, y) = fy(x, y) = 0;
            fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
            fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

            # Compute derivative of d over [x, y]
            d_r = k1 + r * (2.0 * k2 + r * (3.0 * k3 + r * 4.0 * k4))
            d_x = 2.0 * x * d_r
            d_y = 2.0 * y * d_r

            # Compute derivative of fx over x and y.
            fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
            fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

            # Compute derivative of fy over x and y.
            fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
            fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

            return (fx, fy), (fx_x, fx_y, fy_x, fy_y)

        for _ in range(max_iterations):
            (fx, fy), (fx_x, fx_y, fy_x, fy_y) = compute_residual_and_jacobian(x, y)
            denominator = fy_x * fx_y - fx_x * fy_y
            x_numerator = fx * fy_y - fy * fx_y
            y_numerator = fy * fx_x - fx * fy_x
            step_x = jnp.where(jnp.abs(denominator) > eps, x_numerator / denominator, jnp.zeros_like(denominator))
            step_y = jnp.where(jnp.abs(denominator) > eps, y_numerator / denominator, jnp.zeros_like(denominator))

            x = x + step_x
            y = y + step_y

        return x, y

    def make_ray_directions_from_pixel_coordinates(
        self,
        x: jax.Array,
        y: jax.Array,
        use_pixel_center: bool,
    ) -> jax.Array:
        """Given distorted unnormalized pixel coordinates, generate a ray direction for each of them

        Inputs:
            x, y `uint32` `[N]`: unnormalized pixel coordinates

        Returns:
            dirs `float` `[N, 3]`: directions that have taken distortion into account
        """
        chex.assert_type([x, y], jnp.uint32)
        chex.assert_rank([x, y], 1)
        chex.assert_equal_shape([x, y])
        pixel_offset = 0.5 if use_pixel_center else 0.0
        x, y, z = (  # in CV coordinates, axes are flipped later
            ((x + pixel_offset) - self.cx) / self.fx,
            ((y + pixel_offset) - self.cy) / self.fy,
            jnp.ones_like(x),
        )
        if self.has_distortion:
            x, y = self.undistort(x, y)
        if self._type == "FISHEYE":
            theta = jnp.sqrt(jnp.square(x) + jnp.square(y))
            theta = jnp.clip(theta, 0., jnp.pi)
            co, si = jnp.cos(theta), jnp.sin(theta)
            x, y, z = (
                x * si / theta,
                y * si / theta,
                z * co,
            )
        dirs = jnp.stack([x, y, z], axis=-1)

        # flip axis from CV coordinates to CG coordinates
        dirs = dirs @ jnp.diag(jnp.asarray([1., -1., -1.]))

        return dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)


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
    near: float | None=None
    k1: float | None=None
    k2: float | None=None
    k3: float | None=None
    k4: float | None=None
    p1: float | None=None
    p2: float | None=None
    model: CameraModelType | None=None
    distortion: bool=True

    @property
    def fx(self) -> float:
        return self.focal
    @property
    def fy(self) -> float:
        return self.focal
    @property
    def cx(self) -> float:
        return self.width / 2.
    @property
    def cy(self) -> float:
        return self.height / 2.

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
                k1=self.k1,
                k2=self.k2,
                k3=self.k3,
                k4=self.k4,
                p1=self.p1,
                p2=self.p2,
                model=self.model,
                distortion=self.distortion,
            )
        assert self.width > 0 and self.height > 0
        if self.focal is None:
            self.__init__(
                width=self.width,
                height=self.height,
                focal=min(self.width, self.height),
                near=self.near,
                k1=self.k1,
                k2=self.k2,
                k3=self.k3,
                k4=self.k4,
                p1=self.p1,
                p2=self.p2,
                model=self.model,
                distortion=self.distortion,
            )

    def update_camera(self, camera: Camera | None=None) -> Camera:
        def try_override(name: str) -> int | float | CameraModelType:
            return getattr(self, name) or getattr(camera, name)
        def try_override_if_has_distortion_else_zero(name: str) -> float:
            return try_override(name) if self.distortion else 0.
        width, height, fx, fy, cx, cy, near, model = map(
            try_override,
            ["width", "height", "fx", "fy", "cx", "cy", "near", "model"],
        )
        k1, k2, k3, k4, p1, p2 = map(
            try_override_if_has_distortion_else_zero,
            ["k1", "k2", "k3", "k4", "p1", "p2"],
        )
        return camera.replace(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            near=near,
            k1=k1,
            k2=k2,
            k3=k3,
            k4=k4,
            p1=p1,
            p2=p2,
            model=model,
        )

    @property
    def enabled(self) -> bool:
        return any(map(
            lambda name: getattr(self, name) is not None,
            ["width", "height", "focal", "near", "model"]
            + ["k1", "k2", "k3", "k4", "p1", "p2"],
        ))


@empty_impl
@replace_impl
@pydantic.dataclasses.dataclass(frozen=True)
class SceneOptions:
    # images with sharpness lower than this value will be discarded
    sharpness_threshold: float

    # scale input images in case they are too large, camera intrinsics are also scaled to match the
    # updated image resolution.
    resolution_scale: float

    camera_near: float

    # maximum GPU memory to consume in MB, the pixels are reloaded into GPU memory before each epoch
    # if the scene has more than this number of pixels, otherwise all pixels are loaded once
    max_mem_mbytes: int

    # overrides `aabb_scale` in transforms.json
    bound: float | None=None

    # overrides `up` in transforms.json
    up: Tuple[float, float, float] | None=None

    def __post_init__(self):
        assert 0 <= self.resolution_scale <= 1, (
            "resolution_scale must be in range [0, 1], got {}".format(self.resolution_scale)
        )

    @property
    def up_unitvec(self) -> Tuple[float, float, float] | None:
        if self.up is None:
            return None
        up = np.asarray(self.up)
        up = up / np.linalg.norm(up)
        return tuple(up.tolist())


@dataclass
class RigidTransformation:
    # [3, 3] rotation matrix
    rotation: jax.Array

    # [3] translation vector
    translation: jax.Array

    def __post_init__(self):
        chex.assert_shape([self.rotation, self.translation], [(3, 3), (3,)])


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

    def rotate_world_up(self, up: Tuple[float, float, float] | np.ndarray) -> "TransformJsonFrame":
        def rotmat(a, b):
            "copied from NVLabs/instant-ngp/scripts/colmap2nerf.py"
            a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
            v = np.cross(a, b)
            c = np.dot(a, b)
            # handle exception for the opposite direction input
            if c < -1 + 1e-10:
                return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))
        up = np.asarray(up)
        R = rotmat(up, [0, 0, 1])
        R = np.pad(R, [0, 1])
        R[-1, -1] = 1
        new_transform_matrix = R @ self.transform_matrix_numpy
        return self.replace(
            transform_matrix=new_transform_matrix.tolist(),
        )

    def scale_camera_positions(self, scale: float) -> "TransformJsonFrame":
        new_transform_matrix = self.transform_matrix_numpy
        new_transform_matrix[:3, 3] *= scale * 2
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
    aabb_scale: float=dataclasses.field(default=1., kw_only=True)

    # scale camera's translation vectors by this factor while loading (default value taken from
    # NVLabs/instant-ngp/include/neural-graphics-primitives/nerf_loader.h), since current
    # implementation (this codebase) represents the scene inside a 2^3 cube centered at origin, to
    # achieve the same scene scale as that of NVLabs/instant-ngp while using the same
    # transform*.json files, the camera translation vectors will be scaled by 2 time this value.
    # I.e. if the transform*.json specifies `"scale": 0.3`, loaded cameras' translation vectors will
    # be scaled by `0.6`.  See `utils.types.TransformJsonFrame.scale_camera_positions` for details.
    # NOTE: this value does not affect scene's bounding box
    scale: float=dataclasses.field(default=1/3, kw_only=True)

    bg: bool=dataclasses.field(default=False, kw_only=True)

    up: Tuple[float, float, float]=dataclasses.field(default=(0, 0, 1), kw_only=True)

    n_extra_learnable_dims: int=dataclasses.field(default=0, kw_only=True)

    def rotate_world_up(self) -> "TransformJsonBase":
        return self.replace(
            frames=tuple(map(lambda f: f.rotate_world_up(self.up), self.frames)),
            up=(0., 0., 1.),  # so that this operation is idempotent
        )

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

    k1: float=0.
    k2: float=0.
    k3: float=0.
    k4: float=0.
    p1: float=0.
    p2: float=0.

    camera_model: CameraModelType="OPENCV"


@replace_impl
@pydantic.dataclasses.dataclass(frozen=True)
class SceneCreationOptions:
    # given that the cameras' average distance to the origin is (4.0 * `camera_scale`), what would
    # the scene's bound be?
    bound: float

    # `Sequntial` for continuous frames, `Exhaustive` for all possible pairs
    matcher: ColmapMatcherType

    camera_model: CameraModelType="OPENCV"

    # upon loading the created scene during training/inference, scale the camera positions with this
    # factor
    camera_scale: float=dataclasses.field(default=1/3)

    # whether to enable background model
    bg: bool=dataclasses.field(default=False)

    # dimension of NeRF-W-style per-image appearance embeddings, set to 0 to disable
    n_extra_learnable_dims: int=dataclasses.field(default=16)

    # whether undistort the images and write them to disk, so that the camera in transforms.json is
    # a pinhole camera
    undistort: bool=dataclasses.field(default=False)


@dataclass
class ImageMetadata:
    H: int
    W: int
    xys: jax.Array  # int,[H*W, 2]: original integer coordinates in range [0, W] for x and [0, H] for y
    uvs: jax.Array  # float,[H*W, 2]: normalized coordinates in range [0, 1]
    rgbs: jax.Array  # float,[H*W, 3]: normalized rgb values in range [0, 1]


@dataclass
class OrbitTrajectoryOptions:
    # cameras' distance to the orbiting axis
    radius: float=1.8

    # lowest height of generated trajectory
    low: float=0.0

    # highest height of generated trajectory
    high: float=1.3

    # how many frames should be rendered per orbit
    n_frames_per_orbit: int=144

    n_orbit: int=2

    # all orbiting cameras will look at this point
    centroid: Tuple[float, float, float]=(0., 0., 0.2)

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
    camera: Camera

    n_extra_learnable_dims: int

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
        xs = np.cos(thetas) * opts.radius
        ys = np.sin(thetas) * opts.radius
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
    @dataclass
    class ViewData:
        width: int
        height: int
        transform: RigidTransformation
        file: Path=struct.field(pytree_node=False)

        @property
        def image_pil(self) -> Image.Image:
            image = Image.open(self.file)
            image = image.resize((self.width, self.height), resample=Image.LANCZOS)
            return image

        @property
        def image_rgba_u8(self) -> jax.Array:
            image = np.asarray(self.image_pil)
            if image.shape[-1] == 1:
                image = np.concatenate([image] * 3 + [255 * np.ones_like(image[..., :1])], axis=-1)
            elif image.shape[-1] == 3:
                image = np.concatenate([image, 255 * np.ones_like(image[..., :1])], axis=-1)
            chex.assert_axis_dimension(image, -1, 4)
            return image

        # float, [H*W, 4]: rgba values of type uint8
        @property
        def rgba_u8(self) -> jax.Array:
            return self.image_rgba_u8.reshape(-1, 4)

    meta: SceneMeta

    # maximum GPU memory to consume in MB, the pixels are reloaded into GPU memory before each epoch
    # if the scene has more than this number of pixels, otherwise all pixels are loaded once
    max_mem_mbytes: int

    # uint32, [n_pixels]
    _view_indices: jax.Array | None=None

    # uint32, [n_pixels]
    _pixel_indices: jax.Array | None=None

    # uint8, [n_pixels, 4]
    rgbas_u8: jax.Array | None=None

    def _free(self):
        backend = jax.lib.xla_bridge.get_backend()
        if self._view_indices is not None: backend.buffer_from_pyval(self._view_indices).delete()
        if self._pixel_indices is not None: backend.buffer_from_pyval(self._pixel_indices).delete()
        if self.rgbas_u8 is not None: backend.buffer_from_pyval(self.rgbas_u8).delete()
        backend.buffer_from_pyval(self.transforms).delete()
        jax.lib.xla_bridge.get_backend().defragment()

    @functools.cached_property
    def all_views(self) -> Tuple[ViewData, ...]:
        return tuple(map(
            lambda frame: self.ViewData(
                width=self.meta.camera.width,
                height=self.meta.camera.height,
                transform=RigidTransformation(
                    rotation=frame.transform_matrix_numpy[:3, :3],
                    translation=frame.transform_matrix_numpy[:3, 3],
                ),
                file=frame.file_path,
            ),
            self.meta.frames,
        ))

    # float, [n_views, 9+3]
    @functools.cached_property
    def transforms(self) -> jax.Array:
        return jnp.stack(list(ThreadPoolExecutor().map(
            lambda view: jnp.concatenate([
                view.transform.rotation.ravel(),
                view.transform.translation,
            ]),
            self.all_views,
        )))

    def _should_load_all_pixels(self, /, max_mem_mbytes: int) -> bool:
        n_bytes = max_mem_mbytes * 1024 * 1024
        required_total_mem_bytes = 4 * self.meta.n_pixels
        return n_bytes >= required_total_mem_bytes

    @property
    def load_all_pixels(self) -> bool:
        return self._should_load_all_pixels(max_mem_mbytes=self.max_mem_mbytes)

    @property
    def n_views(self) -> int:
        return len(self.all_views)

    def _calculate_num_pixels(self, /, max_mem_mbytes: int) -> int:
        if self._should_load_all_pixels(max_mem_mbytes=max_mem_mbytes):
            return self.meta.n_pixels
        else:
            # Dividing by 3 because also need to keep track of view indices and pixel indices, each
            # consuming the same amount of memory as rgba_u8.
            n_bytes = max_mem_mbytes * 1024 * 1024
            return int(n_bytes * .33 / 4)

    @property
    def n_pixels(self) -> int:
        return self._calculate_num_pixels(max_mem_mbytes=self.max_mem_mbytes)

    def get_view_indices(self, perm: jax.Array) -> jax.Array:
        if self._view_indices is not None:
            return self._view_indices[perm]
        else:
            return jnp.floor_divide(perm, self.meta.camera.n_pixels)

    def get_pixel_indices(self, perm: jax.Array) -> jax.Array:
        if self._pixel_indices is not None:
            return self._pixel_indices[perm]
        else:
            return jnp.mod(perm, self.meta.camera.n_pixels)

    def resample_pixels(self, /, KEY: jran.KeyArray, new_max_mem_mbytes: int | None=None) -> "SceneData":
        load_all_pixels = self._should_load_all_pixels(new_max_mem_mbytes)
        if (
            (self.load_all_pixels and self.rgbas_u8 is not None)
            and (new_max_mem_mbytes is None or load_all_pixels)
        ):
            return dataclasses.replace(self, max_mem_mbytes=new_max_mem_mbytes)
        if new_max_mem_mbytes is None:
            new_max_mem_mbytes = self.max_mem_mbytes

        # free up GPU memory
        self._free()
        n_pixels = self._calculate_num_pixels(max_mem_mbytes=new_max_mem_mbytes)
        make_progress_bar = functools.partial(
            tqdm,
            total=self.n_views,
            desc="| {} {} ({:.2f}% of {}) training pixels".format(
                "loading"
                if self._view_indices is None
                else "resampling",
                n_pixels,
                n_pixels / self.meta.n_pixels * 100,
                self.meta.n_pixels,
            ),
            bar_format=tqdm_format,
        )

        if load_all_pixels:
            rgbas_u8 = jnp.concatenate(list(make_progress_bar(ThreadPoolExecutor().map(
                lambda view: view.rgba_u8,
                self.all_views,
            ))))
            return dataclasses.replace(
                self,
                max_mem_mbytes=new_max_mem_mbytes,
                _view_indices=None,
                _pixel_indices=None,
                rgbas_u8=rgbas_u8,
            )
        else:
            KEY, key_n, key_view_perm, key_pixel_idcs = jran.split(KEY, 4)
            ns = jran.uniform(key_n, shape=(self.n_views - 1,), minval=7, maxval=13)
            ns = ns / ns.sum()
            ns = (ns * n_pixels / self.n_views * (self.n_views - 1)).astype(jnp.uint32)
            ns = jnp.concatenate([ns, (n_pixels - ns.sum()) * jnp.ones_like(ns[:1])])
            assert ns.sum() == n_pixels
            sections, ns = jnp.cumsum(ns), ns.tolist()

            pixel_idcs = jran.choice(
                key=key_pixel_idcs,
                a=self.meta.camera.n_pixels,
                shape=(n_pixels,),
                replace=True,
            )
            pixel_idcs_per_view = jnp.split(pixel_idcs, sections)

            view_perm = jran.permutation(key=key_view_perm, x=self.n_views)
            view_idcs = jnp.concatenate(list(ThreadPoolExecutor().map(
                lambda vi, pidcs: vi * jnp.ones(pidcs.shape[0], dtype=jnp.uint32),
                view_perm,
                pixel_idcs_per_view,
            )))

            rgbas_u8 = jnp.concatenate(list(make_progress_bar(ThreadPoolExecutor().map(
                lambda vi, pidcs: self.all_views[vi].rgba_u8[pidcs],
                view_perm,
                pixel_idcs_per_view,
            ))))

            return dataclasses.replace(
                self,
                max_mem_mbytes=new_max_mem_mbytes,
                _view_indices=view_idcs,
                _pixel_indices=pixel_idcs,
                rgbas_u8=rgbas_u8,
            )


@dataclass
class RenderedImage:
    bg: jax.Array
    rgb: jax.Array
    depth: jax.Array


@empty_impl
class NeRFState(TrainState):
    # WARN:
    #   do not annotate fields with jax.Array as members with flax.truct.field(pytree_node=False),
    #   otherwise weird issues happen, e.g. jax tracer leak, array-to-boolean conversion exception
    #   while calling a jitted function with no helpful traceback.
    ogrid: OccupancyDensityGrid

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

    def update_ogrid_density(
        self,
        KEY: jran.KeyArray,
        cas: int,
        update_all: bool,
        max_inference: int,
    ) -> "NeRFState":
        G3 = self.raymarch.density_grid_res**3
        cas_slice = slice(cas * G3, (cas + 1) * G3)
        cas_alive_indices = self.ogrid.alive_indices[self.ogrid.alive_indices_offset[cas]:self.ogrid.alive_indices_offset[cas+1]]
        aligned_indices = cas_alive_indices % G3  # values are in range [0, G3)
        n_grids = aligned_indices.shape[0]

        decay = .95
        cas_occ_mask = self.ogrid.occ_mask[cas_slice]
        cas_density_grid = self.ogrid.density[cas_slice].at[aligned_indices].set(self.ogrid.density[cas_slice][aligned_indices] * decay)

        if update_all:
            # During the first 256 training steps, we sample M = K * 128^{3} cells uniformly without
            # repetition.
            cas_updated_indices = aligned_indices
        else:
            M = max(1, n_grids // 2)
            # The first M/2 cells are sampled uniformly among all cells.
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

        new_densities = map(
            lambda coords_part: jax.jit(self.nerf_fn)(
                {"params": self.locked_params["nerf"]},
                coords_part,
                None,
                None,
            )[0].ravel(),
            jnp.array_split(jax.lax.stop_gradient(coordinates), max(1, n_grids // (max_inference))),
        )
        new_densities = jnp.concatenate(list(new_densities))

        cas_density_grid = cas_density_grid.at[cas_updated_indices].set(
            jnp.maximum(cas_density_grid[cas_updated_indices], new_densities)
        )
        new_ogrid = self.ogrid.replace(
            density=self.ogrid.density.at[cas_slice].set(cas_density_grid),
        )
        return self.replace(ogrid=new_ogrid)

    @jax.jit
    def threshold_ogrid(self) -> "NeRFState":
        mean_density = self.ogrid.mean_density_up_to_cascade(1)
        density_threshold = jnp.minimum(self.density_threshold_from_min_step_size, mean_density)
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
        grid_xyzs = morton3d_invert(pos_idcs).astype(jnp.float32)  # [G3, 3]
        grid_xyzs /= G  # in range [0, 1)
        grid_xyzs -= 0.5  # in range [-0.5, 0.5)
        grid_xyzs *= 2 * mip_bound[:, None]  # in range [-mip_bound, mip_bound)
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
        all_grid_vertices = grid_xyzs[:, None, :] + vertex_offsets

        @jax.jit
        def mark_untrained_density_grid_single_frame(
            alive_marker: jax.Array,
            transform_cw: jax.Array,
            grid_vertices: jax.Array,
        ):
            rot_cw, t_cw = transform_cw[:3, :3], transform_cw[:3, 3]
            # p_world, p_cam, T: [3, 1]
            # rot_cw: [3, 3]
            # p_world = rot_cw @ p_cam + t_cw
            p_aligned = grid_vertices - t_cw
            p_cam = (p_aligned[..., None, :] * rot_cw.T).sum(-1)

            # camera looks along the -z axis
            in_front_of_camera = p_cam[..., -1] < 0

            u, v = jnp.split(p_cam[..., :2] / (-p_cam[..., -1:]), [1], axis=-1)

            if self.scene_meta.camera.has_distortion:
                # distort
                u, v = self.scene_meta.camera.distort(u, v)

                # Pixel coordinates outside the image plane may produce the same `u, v` as those inside
                # the image plane, check if the produced `u, v` match the ray we started with.
                # REF: <https://github.com/NVlabs/instant-ngp/blob/99aed93bbe8c8e074a90ec6c56c616e4fe217a42/src/testbed_nerf.cu#L481-L483>
                re_u, re_v = self.scene_meta.camera.undistort(u, v)
                redir = jnp.concatenate([re_u, re_v, -jnp.ones_like(re_u)], axis=-1)
                redir = redir / jnp.linalg.norm(redir, axis=-1, keepdims=True)
                ogdir = p_cam / jnp.linalg.norm(p_cam, axis=-1, keepdims=True)
                same_ray = (redir * ogdir).sum(axis=-1) > 1. - 1e-3
            else:
                same_ray = True

            uv = jnp.concatenate([
                u * self.scene_meta.camera.fx + self.scene_meta.camera.cx,
                v * self.scene_meta.camera.fy + self.scene_meta.camera.cy,
            ], axis=-1)
            uv = uv / jnp.asarray([self.scene_meta.camera.width, self.scene_meta.camera.height], dtype=jnp.float32)

            within_frame_range = (uv >= 0.) & (uv < 1.)
            within_frame_range = (
                within_frame_range  # shape is [n_grids, 8, 2]
                    .all(axis=-1)  # u and v must both be within frame
            )

            visible_by_camera = (in_front_of_camera & within_frame_range & same_ray).any(axis=-1)  # grid should be trained if any of its 8 vertices is visible

            return alive_marker | visible_by_camera

        # cam_t = np.asarray(list(map(
        #     lambda frame: frame.transform_matrix_numpy[:3, 3],
        #     self.scene_meta.frames,
        # )))
        # np.savetxt("cams.xyz", cam_t)

        alive_marker = jnp.zeros(n_grids, dtype=jnp.bool_)
        for frame in (pbar := tqdm(self.scene_meta.frames, desc="| marking trainable grids".format(n_grids), bar_format=tqdm_format)):
            new_alive_marker_parts = map(
                lambda alive_marker_part, grid_vertices_part: mark_untrained_density_grid_single_frame(
                    alive_marker=alive_marker_part,
                    transform_cw=frame.transform_matrix_jax_array,
                    grid_vertices=grid_vertices_part,
                ),
                jnp.array_split(alive_marker, self.scene_meta.cascades),  # alive_marker_part
                jnp.array_split(all_grid_vertices, self.scene_meta.cascades),  # grid_vertices_part
            )
            alive_marker = jnp.concatenate(list(new_alive_marker_parts), axis=0)
            n_alive_grids = alive_marker.sum()
            ratio_trainable = n_alive_grids / n_grids
            pbar.set_description_str("| marked {}/{} ({:.2f}%) grids as trainable".format(n_alive_grids, n_grids, ratio_trainable * 100))
            if n_alive_grids == n_grids:
                pbar.close()
                break

        marked_density = jnp.where(alive_marker, self.ogrid.density, -1.)
        marked_occ_mask, marked_occupancy = packbits(
            density_threshold=min(self.density_threshold_from_min_step_size, self.ogrid.mean_density_up_to_cascade(1)) if self.step > 0 else -.5,
            density_grid=marked_density
        )

        # rgb = jnp.stack([~marked_occ_mask, jnp.zeros_like(marked_occ_mask, dtype=jnp.float32), marked_occ_mask]).T
        # xyzrgb = np.asarray(jnp.concatenate([grid_xyzs, rgb], axis=-1))
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

    def epoch(self, iters: int) -> int:
        return self.step // iters

    @property
    def density_threshold_from_min_step_size(self) -> float:
        return .01 * self.raymarch.diagonal_n_steps / (2 * min(self.scene_meta.bound, 1) * 3**.5)

    @property
    def use_background_model(self) -> bool:
        return self.scene_meta.bg and self.params.get("bg") is not None

    @property
    def locked_params(self):
        return jax.lax.stop_gradient(self.params)

    @property
    def update_ogrid_interval(self) -> int:
        return min(16, self.step // 16 + 1)

    @property
    def should_call_update_ogrid(self) -> bool:
        return (
            self.step > 0
            and self.step % self.update_ogrid_interval == 0
        )

    @property
    def should_update_all_ogrid_cells(self) -> bool:
        return self.step < 256

    @property
    def should_write_batch_metrics(self) -> bool:
        return self.step % 16 == 0
