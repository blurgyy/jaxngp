from typing import Callable, Tuple

import chex
from flax.core.scope import FrozenVariableDict
import flax.linen as nn
import jax
import jax.numpy as jnp
from tqdm import tqdm

from utils.common import tqdm_format, vmap_jaxfn_with
from utils.data import set_pixels


@jax.jit
def integrate_rays(densities: jax.Array, rgbs: jax.Array) -> jax.Array:
    """
    Inputs:
        densities [steps, 1]: density values along a ray, from ray origin to 
        rgbs [steps, 3]: rgb values along a ray

    Returns:
        rgb [3]: integrated ray colors according to input densities and rgbs.
    """
    # [N, 4]
    # drgb = jax.lax.fori_loop(
    #     0,
    #     densities.shape[0],
    #     lambda i, prev_drgb: jnp.concatenate([densities[i], rgbs[i]], axis=-1) if densities[i] >= prev_drgb[0] else prev_drgb,
    #     jnp.zeros(4)
    # )
    # [1]
    pos = jnp.argmax(densities, axis=0)
    # [3]
    return (densities[pos] * rgbs[pos]).squeeze()


@dataclass
class Camera:
    ...


@dataclass
class PinholeCamera(Camera):
    # resolutions
    W: int
    H: int

    # clipping plane distance, must be positive
    near: float
    far: float

    # focal length
    focal: float


@jax.jit
@vmap_jaxfn_with(in_axes=(0, 0, None))
def make_ndc_rays(
        o_world: jax.Array,
        d_world: jax.Array,
        camera: Camera,
    ) -> Tuple[jax.Array, jax.Array]:
    # shift ray origins to near plane
    t = -(camera.near + o_world[2]) / d_world[2]
    o_world = o_world + t * d_world

    ox = -camera.focal * o_world[0] / (camera.W / 2 * o_world[2])
    oy = -camera.focal * o_world[1] / (camera.H / 2 * o_world[2])
    oz = 1 + 2 * near / o_world[2]
    o_ndc = jnp.asarray([ox, oy, oz])

    dx = -(camera.focal * (d_world[0] / d_world[2] - o_world[0] / o_world[2])) / (camera.W / 2)
    dy = -(camera.focal * (d_world[1] / d_world[2] - o_world[1] / o_world[2])) / (camera.H / 2)
    dz = -2 * camera.near / o_world[2]
    d_ndc = jnp.asarray([dx, dy, dz])

    return o_ndc, d_ndc


class RayMarcher(nn.Module):
    camera: Camera
    steps: int
    apply_fn: Callable[[jax.Array, jax.Array], Tuple[jax.Array, jax.Array]]

    @nn.jit
    @vmap_jaxfn_with(in_axes=(None, 0, 0, None))  # first `None` is for `self`
    def __call__(
            self,
            o_world: jax.Array,
            d_world: jax.Array,
            param_dict: FrozenVariableDict,
        ) -> jax.Array:
        """
        Inputs:
            o_world [3]: ray origins, in world space
            d_world [3]: ray directions (unit vectors), in world space
            param_dict: NeRF model params

        Returns:
            sampled_points [steps, 3]: `step`
        """
        chex.assert_shape([o_world, d_world], [[..., 3], [..., 3]])

        # # NGP paper:
        # #   In synthetic NeRF scenes, which we bound to the unit cube [0, 1]^3, we use a fixed ray
        # #   marching step size equal to Δ𝑡 := √3/1024; √3 represents the diagonal of the unit cube.
        # # NeRF paper:
        # #   For experiments with synthetic images, we scale the scene so that it lies within a cube
        # #   of side length 2 centered at the origin, and only query the representation within this
        # #   bounding volume.
        # # NeRF synthetic data has bounds of [-1, 1]^3, thus the use of 2√3 as Δ𝑡 here.
        # # [steps]
        # delta_t = (2 * jnp.sqrt(3)) * (jnp.arange(steps) + 1) / steps
        # # [1, steps, 1]
        # delta_t = delta_t.reshape(1, steps, 1)
        # # [..., steps, 3]: `steps` sampled points along each input ray
        # sampled_points = o_world[..., None, :] + delta_t * d_world[..., None, :]

        # linearly sample `steps` points from near to far bounds
        # [steps]
        delta_t = (jnp.arange(self.steps) + 1) / self.steps
        # [steps]
        delta_t = self.camera.near + (self.camera.far - self.camera.near) * delta_t
        # [steps, 1]
        delta_t = delta_t.reshape(self.steps, 1)
        # [steps, 3]
        ray_pts = o_world[None, :] + delta_t * d_world[None, :]

        density, rgb = self.apply_fn(
            param_dict,
            ray_pts,
            jnp.broadcast_to(d_world, ray_pts.shape),
        )

        return integrate_rays(density, rgb)


class VolumeRenderer(nn.Module):
    camera: Camera
    raymarcher: RayMarcher
    ray_chunk_size: int

    @nn.jit
    def __call__(self, R_cw: jax.Array, T_cw: jax.Array, param_dict: FrozenVariableDict) -> jax.Array:
        """
        Inputs:
            R_cw [3, 3]: camera-to-world rotation matrix
            T_cw [3]: camera-to-world translation vector

        Returns:
            image_array [H, W, 3]: rendered rgb image, with colors in range [0, 1]
        """
        chex.assert_shape([R_cw, T_cw], [(3, 3), (3,)])
        image_array = jnp.empty((self.camera.H, self.camera.W, 3), dtype=jnp.uint8)
        n_pixels = self.camera.H * self.camera.W

        # generate camera-space rays
        # [H*W, 1]
        d_cam_idcs = jnp.arange(n_pixels).reshape(-1, 1)
        # [H*W, 1]
        d_cam_xs = jnp.mod(d_cam_idcs, self.camera.W)
        d_cam_xs = ((d_cam_xs + 0.5) - self.camera.W/2)
        # [H*W, 1]
        d_cam_ys = jnp.floor_divide(d_cam_idcs, self.camera.W)
        d_cam_ys = -((d_cam_ys + 0.5) - self.camera.H/2)  # NOTE: y axis indexes from bottom to top, so negate it
        # [H*W, 1]
        d_cam_zs = -self.camera.focal * jnp.ones_like(d_cam_idcs)
        # [H*W, 3]
        d_cam = jnp.concatenate([d_cam_xs, d_cam_ys, d_cam_zs], axis=-1)
        d_cam = d_cam / (jnp.linalg.norm(d_cam, axis=-1, keepdims=True) + 1e-15)

        # [H*W, 3]
        o_world = jnp.broadcast_to(T_cw, d_cam.shape)
        # [H*W, 3]
        d_world = d_cam @ R_cw.T
        d_world = d_world / jnp.linalg.norm(d_world, axis=-1, keepdims=True)

        if self.ray_chunk_size > n_pixels:
            n_chunks = 1
        else:
            n_chunks = n_pixels // self.ray_chunk_size
        indices = jnp.arange(self.camera.H * self.camera.W)
        _xs = jnp.mod(indices, self.camera.W)
        _ys = jnp.floor_divide(indices, self.camera.H)
        xys = jnp.concatenate([_xs.reshape(-1, 1), _ys.reshape(-1, 1)], axis=-1)
        indices = jnp.array_split(indices, n_chunks)
        for idcs in tqdm(indices, desc="rendering {}x{} image".format(self.camera.W, self.camera.H), bar_format=tqdm_format):
            rgbs = self.raymarcher(o_world[idcs], d_world[idcs], param_dict)
            image_array = set_pixels(image_array, xys, idcs, rgbs)

        return image_array


if __name__ == "__main__":
    import numpy as np
    import json
    from models.nerfs import make_test_cube
    from PIL import Image

    with open("./data/nerf/nerf_synthetic/lego/transforms_train.json", "r") as f:
        d = json.load(f)

    W, H = 1024, 1024
    focal = .5 * 800 / jnp.tan(d["camera_angle_x"] / 2)
    near = 2
    far = 6

    # K, key, keyy = jran.split(jran.PRNGKey(0xabcdef), 3)

    # o = jran.normal(key, (8, 3)) + 0.5
    # d = 99 * jran.normal(keyy, (8, 3))
    # d /= jnp.linalg.norm(d, axis=-1, keepdims=True)
    camera = PinholeCamera(W=W, H=H, near=near, far=far*2, focal=focal)
    # ndc_o, ndc_d = make_ndc_rays(o, d, camera)
    rm = RayMarcher(
        camera=camera,
        steps=1024,
        apply_fn=make_test_cube(width=1).apply,
    )
    vr = VolumeRenderer(
        camera=camera,
        raymarcher=rm,
        ray_chunk_size=2**20,
    )

    # R_cw = jnp.eye(3)
    # T_cw = jnp.asarray([0, 0, 5])
    # img = vr(R_cw, T_cw, {})
    # Image.fromarray(np.asarray(img)).save("tmp.jpg")
    # exit()

    # pts = rm(o, d)
    # x, y, z = jran.uniform(key, (3,), minval=0, maxval=jnp.pi * 2)
    # cosx, sinx = jnp.cos(x), jnp.sin(x)
    # cosy, siny = jnp.cos(y), jnp.sin(y)
    # cosz, sinz = jnp.cos(z), jnp.sin(z)
    # Rx = jnp.asarray([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
    # Ry = jnp.asarray([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
    # Rz = jnp.asarray([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])
    # R_cw = Rx @ Ry @ Rz
    # R_cw = R_cw / jnp.linalg.norm(R_cw, axis=0, keepdims=True)
    # T_cw = jnp.asarray([1, 0.5, -1])
    for i, frame in enumerate(d["frames"]):
        transformation = jnp.asarray(frame["transform_matrix"])
        R_cw = transformation[:3, :3]
        T_cw = transformation[:3, 3]
        # print(R_cw, R_cw.T)
        print(jnp.linalg.det(R_cw))
        img = vr(R_cw, T_cw, {})
        print(img.max(), img.min())
        print(img.shape)
        img = Image.fromarray(np.asarray(img))
        img.save("rendered-{:03d}.png".format(i))
    # np.savetxt("pts.xyz", np.asarray(pts.reshape(-1, 3)))
