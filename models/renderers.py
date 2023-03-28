from typing import Callable, Tuple

import chex
from flax.core.scope import FrozenVariableDict
import jax
import jax.numpy as jnp
import jax.random as jran
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


def integrate_ray(
        z_vals: jax.Array,
        densities: jax.Array,
        rgbs: jax.Array,
        use_white_bg: bool,
    ):
    """
    Inputs:
        z_vals [steps]: z_vals[i] is the distance of the i-th sample from the camera
        densities [steps, 1]: density values along a ray
        rgbs [steps, 3]: rgb values along a ray

    Returns:
        rgb [3]: integrated ray colors according to input densities and rgbs.
    """
    # [steps-1]
    delta_ts = z_vals[1:] - z_vals[:-1]
    # [steps]
    # set infinite delta_t for last sample since we want to stop ray marching at the last sample
    delta_ts = jnp.concatenate([delta_ts, 1e10 * jnp.ones_like(delta_ts[:1])])

    # NOTE:
    #   jax.lax.fori_loop is slower than vectorized operations (below is vectorized version)

    # [steps]
    alphas = 1 - jnp.exp(-densities.squeeze() * delta_ts)

    # Compute accumulated transmittance up to this sample.  Prepending a value of `1.0` to reflect
    # that the transmittance at the first sample is 100%.
    #
    # In the original NeRF code the author used `tf.math.cumprod(exclusive=True)` to prepend the 1.0
    # and cut the final sample, but jax.numpy does not have an `exclusive` flag for its `cumprod`
    # function, so we manually do it like below.
    # [steps]
    acc_transmittance = jnp.concatenate(
        [
            jnp.ones_like(alphas[:1]),
            jnp.cumprod(1 - alphas[:-1] + 1e-15),  # exclude the final sample which we set its density to inf earlier
        ],
    )

    # weights, reflects the probability of the ray not being absorbed up to this sample.
    # [steps]
    weights = alphas * acc_transmittance

    if rgbs is None:
        return weights
    else:
        # [3]
        final_rgb = jnp.sum(weights[:, None] * rgbs, axis=0)
        # [1]
        depth = jnp.sum(weights * z_vals, axis=-1)

        final_rgb += use_white_bg * (1 - jnp.sum(weights, axis=0))

        return weights, final_rgb, depth


def make_near_far_from_aabb(
        aabb: AABB,  # [3, 2]
        o: jax.Array,  # [3]
        d: jax.Array,  # [3]
    ):
    "Finds a smallest non-negative `t` for each ray, such that o+td is inside the given aabb."

    # make sure d is normalized
    d /= jnp.linalg.norm(d, axis=-1, keepdims=True)

    # avoid d[j] being zero
    eps = 1e-15
    d = jnp.where(
        jnp.signbit(d),  # True for negatives, False for non-negatives
        jnp.clip(d, None, -eps * jnp.ones_like(d)),  # if negative, upper-bound is -eps
        jnp.clip(d, eps * jnp.ones_like(d)),  # if non-negative, lower-bound is eps
    )

    tx0, tx1 = (
        (aabb[0][0] - o[0]) / d[0],
        (aabb[0][1] - o[0]) / d[0],
    )
    ty0, ty1 = (
        (aabb[1][0] - o[1]) / d[1],
        (aabb[1][1] - o[1]) / d[1],
    )
    tz0, tz1 = (
        (aabb[2][0] - o[2]) / d[2],
        (aabb[2][1] - o[2]) / d[2],
    )
    tx_start, tx_end = jnp.minimum(tx0, tx1), jnp.maximum(tx0, tx1)
    ty_start, ty_end = jnp.minimum(ty0, ty1), jnp.maximum(ty0, ty1)
    tz_start, tz_end = jnp.minimum(tz0, tz1), jnp.maximum(tz0, tz1)

    t_start = jnp.maximum(
        jnp.maximum(jnp.maximum(tx_start, ty_start), tz_start),  # last axis that gose inside the bbox
        0,  # t_start should be larger than zero
    )
    t_end = jnp.maximum(
        jnp.minimum(jnp.minimum(tx_end, ty_end), tz_end),  # first axis that goes out of the bbox
        t_start + 1e-5,  # t_end should be larger than t_start for the ray to intersect with the aabb
    )

    return t_start, t_end


def sample_pdf(
        K: jran.KeyArray,
        bins: jax.Array,
        weights: jax.Array,
        n_importance: int,
    ):
    """
    Importance sampling according to given weights.  Adapted from <bmild/nerf>

    Inputs:
        bins [steps-1]
        weights [steps-2]
        n_importance int

    Returns:
        samples [n_importance]
    """
    weights += 1e-5
    # [steps-2]
    pdf = weights / jnp.sum(weights)
    # [steps-1]
    cdf = jnp.concatenate([jnp.zeros_like(pdf[:1]), jnp.cumsum(pdf)])

    # sample
    K, key = jran.split(K, 2)
    # [n_importance]
    u = jran.uniform(key, (n_importance,), dtype=bins.dtype, minval=0, maxval=1)
    # [n_importance], 0 <= inds < steps-1
    inds = jnp.searchsorted(cdf, u, side="right", method="sort")  # "sort" method is more performant on accelerators
    # [n_importance], 0 <= inds_below < steps-1
    inds_below = jnp.maximum(0, inds - 1)
    # [n_importance], 0 < inds_above < steps-1
    inds_above = jnp.minimum(cdf.shape[-1] - 1, inds)

    # [n_importance]
    interval_lengths = cdf[inds_above] - cdf[inds_below]
    interval_lengths = jnp.where(
        jnp.signbit(interval_lengths - 1e-5),  # True if interval_lengths < 1e-5
        jnp.ones_like(interval_lengths),
        interval_lengths,
    )

    # t is again normalized into range [0, 1]
    t = (u - cdf[inds_below]) / interval_lengths

    new_z_vals = bins[inds_below] + t * (bins[inds_above] - bins[inds_below])

    return new_z_vals


@jit_jaxfn_with(static_argnames=["options", "nerf_fn"])
@vmap_jaxfn_with(in_axes=(None, 0, 0, None, None, None, None, None))
def march_rays(
        K: jran.KeyArray,
        o_world: jax.Array,
        d_world: jax.Array,
        aabb: AABB,
        use_white_bg: bool,
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

    # make sure d_world is normalized
    d_world /= jnp.linalg.norm(d_world, axis=-1, keepdims=True)
    # skip the empty space between camera and scene bbox
    t_start, t_end = make_near_far_from_aabb(
        aabb=aabb,
        o=o_world,
        d=d_world
    )

    # linearly sample `steps` points inside clipped bbox
    # [steps]
    z_vals = t_start + (t_end - t_start) * jnp.linspace(0, 1, options.steps)

    if options.stratified:
        # [steps-1]
        mids = .5 * (z_vals[1:] + z_vals[:-1])
        # [steps]
        upper = jnp.concatenate([mids, z_vals[-1:]], axis=-1)
        # [steps]
        lower = jnp.concatenate([z_vals[:1], mids], axis=-1)
        K, key = jran.split(K, 2)
        # [steps]
        t_rand = jran.uniform(key, z_vals.shape, dtype=z_vals.dtype, minval=0, maxval=1)
        # [steps]
        z_vals = lower + (upper - lower) * t_rand

    # [steps, 3]
    ray_pts = o_world[None, :] + z_vals[:, None] * d_world[None, :]

    density, rgb = nerf_fn(
        param_dict,
        ray_pts,
        jnp.broadcast_to(d_world, ray_pts.shape),
    )

    if options.n_importance > 0:
        weights = integrate_ray(z_vals, density, None, use_white_bg)

        z_mids = .5 * (z_vals[1:] + z_vals[:-1])
        K, key = jran.split(K, 2)
        z_samples = sample_pdf(key, z_mids, weights[1:-1], options.n_importance)
        z_samples = jax.lax.stop_gradient(z_samples)

        # update z_vals
        z_vals = jnp.sort(jnp.concatenate([z_vals, z_samples]))

        # update ray_pts
        # [steps, 3]
        ray_pts = o_world[None, :] + z_vals[:, None] * d_world[None, :]

        density, rgb = nerf_fn(
            param_dict,
            ray_pts,
            jnp.broadcast_to(d_world, ray_pts.shape),
        )

    _, rgbs, depths = integrate_ray(z_vals, density, rgb, use_white_bg)

    return rgbs, depths


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
    d_cam /= jnp.linalg.norm(d_cam, axis=-1, keepdims=True)

    # [H*W, 3]
    o_world = jnp.broadcast_to(transform_cw.translation, d_cam.shape)
    # [H*W, 3]
    d_world = d_cam @ transform_cw.rotation.T
    d_world /= jnp.linalg.norm(d_world, axis=-1, keepdims=True)

    return o_world, d_world


@jit_jaxfn_with(static_argnames=["H", "W", "chunk_size"])
def get_indices_chunks(
        K: jran.KeyArray,
        H: int,
        W: int,
        chunk_size: int,
    ):
    n_pixels = H * W
    n_chunks = (n_pixels + chunk_size - 1) // chunk_size

    # randomize ray order
    K, key = jran.split(K, 2)
    indices = jran.permutation(key, H * W)

    # xys has sorted order
    _xs = jnp.mod(jnp.arange(H * W), W)
    _ys = jnp.floor_divide(jnp.arange(H * W), H)
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
        K: jran.KeyArray,
        aabb: AABB,
        camera: PinholeCamera,
        transform_cw: RigidTransformation,
        options: RenderingOptions,
        raymarch_options: RayMarchingOptions,
        param_dict: FrozenVariableDict,
        nerf_fn: Callable[[FrozenVariableDict, jax.Array, jax.Array], DensityAndRGB],
    ) -> Tuple[jax.Array, jax.Array]:
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

    K, key = jran.split(K, 2)
    xys, indices = get_indices_chunks(key, camera.H, camera.W, options.ray_chunk_size)

    image_array = jnp.empty((camera.H, camera.W, 3), dtype=jnp.uint8)
    depth_array = jnp.empty((camera.H, camera.W), dtype=jnp.uint8)
    for idcs in tqdm(indices, desc="| rendering {}x{} image".format(camera.W, camera.H), bar_format=tqdm_format):
        # rgbs = raymarcher(o_world[idcs], d_world[idcs], param_dict)
        K, key = jran.split(K, 2)
        rgbs, depths = march_rays(
            key,
            o_world[idcs],
            d_world[idcs],
            aabb,
            options.use_white_bg,
            raymarch_options,
            param_dict,
            nerf_fn,
        )
        depths = depths / 8.
        image_array = set_pixels(image_array, xys, idcs, rgbs)
        depth_array = set_pixels(depth_array, xys, idcs, depths)

    # print("returning from fn ... ", end="")
    return image_array, depth_array


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
    K = jran.PRNGKey(0xccff)

    # o = jran.normal(key, (8, 3)) + 0.5
    # d = 99 * jran.normal(keyy, (8, 3))
    # d /= jnp.linalg.norm(d, axis=-1, keepdims=True)
    camera = PinholeCamera(W=W, H=H, focal=focal)
    # ndc_o, ndc_d = make_ndc_rays(o, d, camera)
    raymarch_options = RayMarchingOptions(
        steps=2**10,
        stratified=False,
        n_importance=0,
    )
    bound = 1.5
    aabb = [[-bound, bound]] * 3
    render_options = RenderingOptions(
        ray_chunk_size=2**13,
        use_white_bg=True
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
        K, key = jran.split(K, 2)
        img, depth = render_image(
            K=key,
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
        dep = Image.fromarray(np.asarray(depth), mode="L")
        dep.save("depth-{:03d}.png".format(i))
    # np.savetxt("pts.xyz", np.asarray(pts.reshape(-1, 3)))


if __name__ == "__main__":
    main()
