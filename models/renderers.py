from typing import Callable, Tuple

import chex
from flax.core.scope import FrozenVariableDict
import jax
import jax.numpy as jnp
from tqdm import tqdm

from utils.common import jit_jaxfn_with, tqdm_format, vmap_jaxfn_with
from utils.data import set_pixels
from utils.types import (
    AABB,
    DensityAndRGB,
    PinholeCamera,
    RayMarchingOptions,
    RenderingOptions,
    RigidTransformation,
)


def integrate_ray(delta_ts: jax.Array, densities: jax.Array, rgbs: jax.Array) -> jax.Array:
    """
    Inputs:
        delta_ts [steps]: delta_ts[i] is the distance between the i-th sample and the (i-1)th sample,
                          with delta_ts[0] being the distance between the first sample and ray
                          origin
        densities [steps, 1]: density values along a ray
        rgbs [steps, 3]: rgb values along a ray

    Returns:
        rgb [3]: integrated ray colors according to input densities and rgbs.
    """
    # NOTE:
    #   jax.lax.fori_loop is slower than vectorized operations (below is vectorized version)

    # [steps, 1]
    alphas = 1 - jnp.exp(-densities * delta_ts[:, None])
    # [steps, 1]
    weights = alphas * jnp.cumprod(1 - alphas + 1e-15, axis=0)
    # [3]
    final_rgb = jnp.sum(weights * rgbs, axis=0)
    return final_rgb


def make_near_far_from_aabb(
        aabb: AABB,  # [3, 2]
        o: jax.Array,  # [3]
        d: jax.Array,  # [3]
    ):
    # find a smallest non-negative `t` for each ray, such that o+td is inside the given aabb
    tx0, tx1 = (
        (aabb[0][0] - o[0]) / (d[0] + 1e-15),
        (aabb[0][1] - o[0]) / (d[0] + 1e-15),
    )
    ty0, ty1 = (
        (aabb[1][0] - o[1]) / (d[1] + 1e-15),
        (aabb[1][1] - o[1]) / (d[1] + 1e-15),
    )
    tz0, tz1 = (
        (aabb[2][0] - o[2]) / (d[2] + 1e-15),
        (aabb[2][1] - o[2]) / (d[2] + 1e-15),
    )
    tx_start, tx_end = jnp.minimum(tx0, tx1), jnp.maximum(tx0, tx1)
    ty_start, ty_end = jnp.minimum(ty0, ty1), jnp.maximum(ty0, ty1)
    tz_start, tz_end = jnp.minimum(tz0, tz1), jnp.maximum(tz0, tz1)

    t_start = jnp.maximum(jnp.maximum(tx_start, ty_start), tz_start)
    t_end = jnp.minimum(jnp.minimum(tx_end, ty_end), tz_end)

    # t_start should be larger than zero
    t_start = jnp.clip(t_start, 0, None)

    # t_end should be larger than t_start for the ray to intersect with the aabb
    t_end = jnp.clip(t_end, t_start + 1e-5, None)

    return t_start, t_end


@jit_jaxfn_with(static_argnames=["options", "nerf_fn"])
@vmap_jaxfn_with(in_axes=(0, 0, None, None, None, None))
def march_rays(
        o_world: jax.Array,
        d_world: jax.Array,
        aabb: AABB,
        options: RayMarchingOptions,
        param_dict: FrozenVariableDict,
        nerf_fn: Callable[[FrozenVariableDict, jax.Array, jax.Array], DensityAndRGB],
    ) -> jax.Array:
    """
    Given a pack of rays, render the colors along them.

    Inputs:
        o_world [3]: ray origins, in world space
        d_world [3]: ray directions (unit vectors), in world space
        aabb [3, 2]: scene bounds on each of x, y, z axes
        options: see :class:`RayMarchingOptions`
        param_dict: :class:`NeRF` model params
        nerf_fn: function that takes the param_dict, xyz, and viewing directions as inputs, and
                 outputs densities and rgbs.

    Returns:
        rgb [3]: rendered colors
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

    # skip the empty space between camera and scene bbox
    t_start, t_end = make_near_far_from_aabb(
        aabb=aabb,
        o=o_world,
        d=d_world
    )

    # linearly sample `steps` points inside clipped bbox
    # [steps]
    z_vals = t_start + (t_end - t_start) * jnp.linspace(0, 1, options.steps)
    # [steps, 3]
    ray_pts = o_world[None, :] + z_vals[:, None] * d_world[None, :]

    density, rgb = nerf_fn(
        param_dict,
        ray_pts,
        jnp.broadcast_to(d_world, ray_pts.shape),
    )

    # [steps]
    delta_ts = jnp.ones(options.steps) * (t_end - t_start) / options.steps
    # first sample's distance from camera is t_start
    delta_ts = delta_ts.at[0].set(t_start)
    # we want to stop ray marching at last sample
    delta_ts = delta_ts.at[-1].set(1e10)

    return integrate_ray(delta_ts, density, rgb)


@jit_jaxfn_with(static_argnames=["camera"])
def make_rays_worldspace(
        camera: PinholeCamera,
        transform_cw: RigidTransformation,
    ):
    """
    Generate world-space rays for each pixel in the given camera's projection plane.

    Inputs:
        camera: camera model in-use
        transform_cw[rotation, translation]: camera to world transformation
            rotation [3, 3]: rotation matrix
            translation [3]: translation vector

    Returns:
        o_world [H*W, 3]: ray origins, in world-space
        d_world [H*W, 3]: ray directions, in world-space
    """
    H, W = camera.H, camera.W
    n_pixels = H * W
    # [H*W, 1]
    d_cam_idcs = jnp.arange(n_pixels).reshape(-1, 1)
    # [H*W, 1]
    d_cam_xs = jnp.mod(d_cam_idcs, camera.W)
    d_cam_xs = ((d_cam_xs + 0.5) - camera.W/2)
    # [H*W, 1]
    d_cam_ys = jnp.floor_divide(d_cam_idcs, camera.W)
    d_cam_ys = -((d_cam_ys + 0.5) - camera.H/2)  # NOTE: y axis indexes from bottom to top, so negate it
    # [H*W, 1]
    d_cam_zs = -camera.focal * jnp.ones_like(d_cam_idcs)
    # [H*W, 3]
    d_cam = jnp.concatenate([d_cam_xs, d_cam_ys, d_cam_zs], axis=-1)
    d_cam = d_cam / (jnp.linalg.norm(d_cam, axis=-1, keepdims=True) + 1e-15)

    # [H*W, 3]
    o_world = jnp.broadcast_to(transform_cw.translation, d_cam.shape)
    # [H*W, 3]
    d_world = d_cam @ transform_cw.rotation.T

    return o_world, d_world


@jit_jaxfn_with(static_argnames=["H", "W", "chunk_size"])
def get_indices_chunks(
        H: int,
        W: int,
        chunk_size: int,
    ):
    n_pixels = H * W
    n_chunks = (n_pixels + chunk_size - 1) // chunk_size
    indices = jnp.arange(H * W)
    _xs = jnp.mod(indices, W)
    _ys = jnp.floor_divide(indices, H)
    xys = jnp.concatenate([_xs.reshape(-1, 1), _ys.reshape(-1, 1)], axis=-1)

    return xys, jnp.array_split(indices, n_chunks)


# WARN:
#   Enabling jit makes the rendering loop seems much faster but hangs for several seconds before
#   returning, sometimes even OOMs during this hang.  Guessing it's because jitted function doesn't
#   block operations even for the values that needs to be blocked and updated (the `image_array` in
#   this function).
#
# @jit_jaxfn_with(static_argnames=["camera", "options", "raymarch_options", "nerf_fn"])
def render_image(
        aabb: AABB,
        camera: PinholeCamera,
        transform_cw: RigidTransformation,
        options: RenderingOptions,
        raymarch_options: RayMarchingOptions,
        param_dict: FrozenVariableDict,
        nerf_fn: Callable[[FrozenVariableDict, jax.Array, jax.Array], DensityAndRGB],
    ) -> jax.Array:
    """
    Given a rigid transformation and a camera model, render the image as seen by the camera.

    Inputs:
        camera: camera model in-use
        transform_cw[rotation, translation]: camera to world transformation
            rotation [3, 3]: rotation matrix
            translation [3]: translation vector
        options: see :class:`RenderingOptions`
        raymarch_options: passed to :func:`march_rays()`, see :class:`RayMarchingOptions`
        param_dict: :class:`NeRF` model params
        nerf_fn: function that takes the param_dict, xyz, and viewing directions as inputs, and
                 outputs densities and rgbs.

    Returns:
        image_array [H, W, 3]: rendered rgb image, with colors in range [0, 1]
    """
    chex.assert_shape([transform_cw.rotation, transform_cw.translation], [(3, 3), (3,)])

    o_world, d_world = make_rays_worldspace(camera=camera, transform_cw=transform_cw)

    xys, indices = get_indices_chunks(camera.H, camera.W, options.ray_chunk_size)

    image_array = jnp.empty((camera.H, camera.W, 3), dtype=jnp.uint8)
    for idcs in tqdm(indices, desc="rendering {}x{} image".format(camera.W, camera.H), bar_format=tqdm_format):
        # rgbs = raymarcher(o_world[idcs], d_world[idcs], param_dict)
        rgbs = march_rays(
            o_world[idcs],
            d_world[idcs],
            aabb,
            raymarch_options,
            param_dict,
            nerf_fn,
        )
        image_array = set_pixels(image_array, xys, idcs, rgbs)

    # print("returning from fn ... ", end="")
    return image_array


@jax.jit
@vmap_jaxfn_with(in_axes=(0, 0, None))
def make_ndc_rays(
        o_world: jax.Array,
        d_world: jax.Array,
        camera: PinholeCamera,
    ) -> Tuple[jax.Array, jax.Array]:
    raise NotImplementedError("Not using make_ndc_rays now")
    # TODO:
    #   add eps for every denominator in this function to avoid division by zero

    # shift ray origins to near plane
    t = -(camera.near + o_world[2]) / d_world[2]
    o_world = o_world + t * d_world

    ox = -camera.focal * o_world[0] / (camera.W / 2 * o_world[2])
    oy = -camera.focal * o_world[1] / (camera.H / 2 * o_world[2])
    oz = 1 + 2 * camera.near / o_world[2]
    o_ndc = jnp.asarray([ox, oy, oz])

    dx = -(camera.focal * (d_world[0] / d_world[2] - o_world[0] / o_world[2])) / (camera.W / 2)
    dy = -(camera.focal * (d_world[1] / d_world[2] - o_world[1] / o_world[2])) / (camera.H / 2)
    dz = -2 * camera.near / o_world[2]
    d_ndc = jnp.asarray([dx, dy, dz])

    return o_ndc, d_ndc


def main():
    import numpy as np
    import json
    from models.nerfs import make_test_cube
    from PIL import Image

    with open("./data/nerf/nerf_synthetic/lego/transforms_train.json", "r") as f:
        d = json.load(f)

    W, H = 1024, 1024
    focal = .5 * 800 / np.tan(d["camera_angle_x"] / 2)

    # K, key, keyy = jran.split(jran.PRNGKey(0xabcdef), 3)

    # o = jran.normal(key, (8, 3)) + 0.5
    # d = 99 * jran.normal(keyy, (8, 3))
    # d /= jnp.linalg.norm(d, axis=-1, keepdims=True)
    camera = PinholeCamera(W=W, H=H, focal=focal)
    # ndc_o, ndc_d = make_ndc_rays(o, d, camera)
    # WARN:
    #   since the `test_cube`'s color is varying through space, rendering with too few steps causes
    #   under-sampling, which results in darker appearance in rendered image.
    raymarch_options = RayMarchingOptions(steps=2**10)
    bound = 1.5
    aabb = [[-bound, bound]] * 3
    render_options = RenderingOptions(
        ray_chunk_size=2**13,
    )
    nerf_fn = make_test_cube(
        width=1,
        aabb=aabb,
        density=32,
    ).apply

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
        img = render_image(
            aabb=aabb,
            camera=camera,
            transform_cw=RigidTransformation(rotation=R_cw, translation=T_cw),
            options=render_options,
            raymarch_options=raymarch_options,
            param_dict={},
            nerf_fn=nerf_fn,
        )
        # img = vr(R_cw, T_cw, {})
        # img = render_fn(R_cw, T_cw, {}, camera, 2**19, rm)
        print("got img!")
        print(img.max(), img.min())
        print(img.shape)
        img = Image.fromarray(np.asarray(img))
        img.save("rendered-{:03d}.png".format(i))
    # np.savetxt("pts.xyz", np.asarray(pts.reshape(-1, 3)))


if __name__ == "__main__":
    main()
