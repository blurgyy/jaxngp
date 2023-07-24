from dataclasses import dataclass
import math
from typing import Callable

from flax.core.scope import FrozenVariableDict
import jax
import jax.numpy as jnp
import jax.random as jran
from volrendjax import (
    integrate_rays,
    integrate_rays_inference,
    march_rays,
    march_rays_inference,
)

from utils.common import jit_jaxfn_with
from utils.data import f32_to_u8
from utils.types import Camera, CameraOverrideOptions, NeRFState, RigidTransformation


@jax.jit
def make_rays_worldspace(
    camera: Camera,
    transform_cw: RigidTransformation,
):
    """
    Generate world-space rays for each pixel in the given camera's projection plane.

    Inputs:
        camera: the camera model in use
        transform_cw[rotation, translation]: camera to world transformation
            rotation [3, 3]: rotation matrix
            translation [3]: translation vector

    Returns:
        o_world [H*W, 3]: ray origins, in world-space
        d_world [H*W, 3]: ray directions, in world-space
    """
    # [H*W, 1]
    d_cam_idcs = jnp.arange(camera.n_pixels)
    x, y = (
        jnp.mod(d_cam_idcs, camera.width),
        jnp.floor_divide(d_cam_idcs, camera.width),
    )
    # [H*W, 3]
    d_cam = camera.make_ray_directions_from_pixel_coordinates(x, y, use_pixel_center=True)

    # [H*W, 3]
    o_world = jnp.broadcast_to(transform_cw.translation, d_cam.shape)
    # [H*W, 3]
    d_world = d_cam @ transform_cw.rotation.T

    return o_world, d_world


@jax.jit
def make_near_far_from_bound(
    bound: float,
    o: jax.Array,  # [n_rays, 3]
    d: jax.Array,  # [n_rays, 3]
):
    "Calculates near and far intersections with the bounding box [-bound, bound]^3 for each ray."

    # avoid d[j] being zero
    eps = 1e-15
    d = jnp.where(
        jnp.signbit(d),  # True for negatives, False for non-negatives
        jnp.clip(d, None, -eps * jnp.ones_like(d)),  # if negative, upper-bound is -eps
        jnp.clip(d, eps * jnp.ones_like(d)),  # if non-negative, lower-bound is eps
    )

    # [n_rays]
    tx0, tx1 = (
        (-bound - o[:, 0]) / d[:, 0],
        (bound - o[:, 0]) / d[:, 0],
    )
    ty0, ty1 = (
        (-bound - o[:, 1]) / d[:, 1],
        (bound - o[:, 1]) / d[:, 1],
    )
    tz0, tz1 = (
        (-bound - o[:, 2]) / d[:, 2],
        (bound - o[:, 2]) / d[:, 2],
    )
    tx_start, tx_end = jnp.minimum(tx0, tx1), jnp.maximum(tx0, tx1)
    ty_start, ty_end = jnp.minimum(ty0, ty1), jnp.maximum(ty0, ty1)
    tz_start, tz_end = jnp.minimum(tz0, tz1), jnp.maximum(tz0, tz1)

    # when t_start<0, or t_start>t_end, ray does not intersect with aabb, these cases are handled in
    # the `march_rays` implementation
    t_start = jnp.maximum(jnp.maximum(tx_start, ty_start), tz_start)  # last axis that gose inside the bbox
    t_end = jnp.minimum(jnp.minimum(tx_end, ty_end), tz_end)  # first axis that goes out of the bbox

    t_start = jnp.maximum(0., t_start)

    # [n_rays], [n_rays]
    return t_start, t_end


@jit_jaxfn_with(static_argnames=["total_samples"])
def render_rays_train(
    KEY: jran.KeyArray,
    o_world: jax.Array,
    d_world: jax.Array,
    appearance_embeddings: jax.Array,
    bg: jax.Array,
    total_samples: int,
    state: NeRFState,
):
    # skip the empty space between camera and scene bbox
    # [n_rays], [n_rays]
    t_starts, t_ends = make_near_far_from_bound(
        bound=state.scene_meta.bound,
        o=o_world,
        d=d_world,
    )

    if state.raymarch.perturb:
        KEY, key = jran.split(KEY, 2)
        noises = jran.uniform(key, shape=t_starts.shape, dtype=t_starts.dtype, minval=0., maxval=1.)
    else:
        noises = 0.
    measured_batch_size_before_compaction, n_valid_rays, rays_n_samples, rays_sample_startidx, ray_idcs, xyzs, dirs, dss, z_vals = march_rays(
        total_samples=total_samples,
        diagonal_n_steps=state.raymarch.diagonal_n_steps,
        K=state.scene_meta.cascades,
        G=state.raymarch.density_grid_res,
        bound=state.scene_meta.bound,
        stepsize_portion=state.scene_meta.stepsize_portion,
        rays_o=o_world,
        rays_d=d_world,
        t_starts=t_starts.ravel(),
        t_ends=t_ends.ravel(),
        noises=noises,
        occupancy_bitfield=state.ogrid.occupancy,
    )

    drgbs, tv = state.nerf_fn(
        {"params": state.params["nerf"]},
        xyzs,
        dirs,
        appearance_embeddings[ray_idcs],
    )

    effective_samples, final_rgbds, final_opacities = integrate_rays(
        near_distance=state.scene_meta.camera.near,
        rays_sample_startidx=rays_sample_startidx,
        rays_n_samples=rays_n_samples,
        bgs=bg,
        dss=dss,
        z_vals=z_vals,
        drgbs=drgbs,
    )

    batch_metrics = {
        "n_valid_rays": n_valid_rays,
        "ray_is_valid": rays_n_samples > 0,
        "measured_batch_size_before_compaction": measured_batch_size_before_compaction,
        "measured_batch_size": jnp.where(effective_samples > 0, effective_samples, 0).sum(),
    }

    return batch_metrics, final_rgbds, tv


@dataclass(frozen=True, kw_only=True)
class MarchAndIntegrateInferencePayload:
    march_steps_cap: int
    diagonal_n_steps: int
    cascades: int
    density_grid_res: int
    bound: float
    stepsize_portion: float
    nerf_fn: Callable


@jit_jaxfn_with(static_argnames=["payload"])
def march_and_integrate_inference(
    payload: MarchAndIntegrateInferencePayload,
    locked_nerf_params: FrozenVariableDict,
    appearance_embedding: jax.Array,

    next_ray_index: jax.Array,
    rays_o: jax.Array,
    rays_d: jax.Array,
    t_starts: jax.Array,
    t_ends: jax.Array,
    occupancy_bitfield: jax.Array,
    terminated: jax.Array,
    indices: jax.Array,

    rays_bg: jax.Array,
    rays_rgbd: jax.Array,
    rays_T: jax.Array,
    rays_cost: jax.Array | None,
):
    next_ray_index, indices, n_samples, t_starts, xyzs, dss, z_vals = march_rays_inference(
        diagonal_n_steps=payload.diagonal_n_steps,
        K=payload.cascades,
        G=payload.density_grid_res,
        march_steps_cap=payload.march_steps_cap,
        bound=payload.bound,
        stepsize_portion=payload.stepsize_portion,
        rays_o=rays_o,
        rays_d=rays_d,
        t_starts=t_starts,
        t_ends=t_ends,
        occupancy_bitfield=occupancy_bitfield,
        next_ray_index_in=next_ray_index,
        terminated=terminated,
        indices=indices,
    )
    if rays_cost is not None:
        rays_cost = rays_cost.at[indices].set(rays_cost[indices] + n_samples)

    xyzs = jax.lax.stop_gradient(xyzs)
    drgbs, _ = payload.nerf_fn(
        {"params": locked_nerf_params},
        xyzs,
        jnp.broadcast_to(rays_d[indices, None, :], xyzs.shape),
        jax.lax.stop_gradient(appearance_embedding),
    )

    terminate_cnt, terminated, rays_rgbd, rays_T = integrate_rays_inference(
        rays_bg=rays_bg,
        rays_rgbd=rays_rgbd,
        rays_T=rays_T,

        n_samples=n_samples,
        indices=indices,
        dss=dss,
        z_vals=z_vals,
        drgbs=drgbs,
    )

    return terminate_cnt, terminated, next_ray_index, indices, t_starts, rays_rgbd, rays_T, rays_cost


def render_image_inference(
    KEY: jran.KeyArray,
    transform_cw: RigidTransformation,
    state: NeRFState,
    camera_override: None | CameraOverrideOptions=None,
    render_cost: bool=False,
    appearance_embedding_index: int=0,
):
    if isinstance(camera_override, Camera):
        state = state.replace(scene_meta=state.scene_meta.replace(camera=camera_override))
    elif isinstance(camera_override, CameraOverrideOptions):
        state = state.replace(
            scene_meta=state.scene_meta.replace(
                camera=camera_override.update_camera(state.scene_meta.camera),
            ),
        )
    elif camera_override is None:
        pass
    else:
        raise RuntimeError(
            "expected `camera_override` to be of type `Camera` or `CameraOverrideOptions`, got {}".format(
                type(camera_override)
            )
        )

    o_world, d_world = make_rays_worldspace(camera=state.scene_meta.camera, transform_cw=transform_cw)
    appearance_embedding = (
        state.locked_params["appearance_embeddings"][appearance_embedding_index]
            if "appearance_embeddings" in state.locked_params
            else jnp.empty(0)
    )
    t_starts, t_ends = make_near_far_from_bound(state.scene_meta.bound, o_world, d_world)
    rays_rgbd = jnp.zeros((state.scene_meta.camera.n_pixels, 4), dtype=jnp.float32)
    rays_T = jnp.ones(state.scene_meta.camera.n_pixels, dtype=jnp.float32)
    if render_cost:
        rays_cost = jnp.zeros(state.scene_meta.camera.n_pixels, dtype=jnp.uint32)
    else:
        rays_cost = None
    if state.use_background_model:
        bg = state.bg_fn(
            {"params": state.locked_params["bg"]},
            o_world,
            d_world,
            appearance_embedding,
        )
    elif state.render.random_bg:
        KEY, key = jran.split(KEY, 2)
        bg = jran.uniform(key, (3,), dtype=jnp.float32, minval=0, maxval=1)
    else:
        bg = state.render.bg
    rays_bg = jnp.broadcast_to(jnp.asarray(bg), (state.scene_meta.camera.n_pixels, 3))

    o_world, d_world, t_starts, t_ends, rays_bg, rays_rgbd, rays_T = jax.lax.stop_gradient((
        o_world, d_world, t_starts, t_ends, rays_bg, rays_rgbd, rays_T
    ))

    march_steps_cap = 8
    n_rays = min(8192, state.scene_meta.camera.n_pixels)

    next_ray_index = jnp.zeros(1, dtype=jnp.uint32)
    terminated = jnp.ones(n_rays, dtype=jnp.bool_)  # all rays are terminated at the beginning
    indices = jnp.zeros(n_rays, dtype=jnp.uint32)
    n_rendered_rays = 0

    while n_rendered_rays < state.scene_meta.camera.n_pixels:
        iters = max(1, (state.scene_meta.camera.n_pixels - n_rendered_rays) // n_rays)
        iters = 2 ** int(math.log2(iters) + 1)

        for _ in range(iters):
            terminate_cnt, terminated, next_ray_index, indices, t_starts, rays_rgbd, rays_T, rays_cost = march_and_integrate_inference(
                payload=MarchAndIntegrateInferencePayload(
                    march_steps_cap=march_steps_cap,
                    diagonal_n_steps=state.raymarch.diagonal_n_steps,
                    cascades=state.scene_meta.cascades,
                    density_grid_res=state.raymarch.density_grid_res,
                    bound=state.scene_meta.bound,
                    stepsize_portion=state.scene_meta.stepsize_portion,
                    nerf_fn=state.nerf_fn,
                ),
                locked_nerf_params=state.locked_params["nerf"],
                appearance_embedding=appearance_embedding,

                next_ray_index=next_ray_index,
                rays_o=o_world,
                rays_d=d_world,
                t_starts=t_starts,
                t_ends=t_ends,
                occupancy_bitfield=state.ogrid.occupancy,
                terminated=terminated,
                indices=indices,

                rays_bg=rays_bg,
                rays_rgbd=rays_rgbd,
                rays_T=rays_T,
                rays_cost=rays_cost,
            )
            n_rendered_rays += terminate_cnt

    bg_array_f32 = rays_bg.reshape((state.scene_meta.camera.height, state.scene_meta.camera.width, 3))
    rays_rgb, rays_depth = jnp.array_split(rays_rgbd, [3], axis=-1)
    image_array_u8 = f32_to_u8(rays_rgb).reshape((state.scene_meta.camera.height, state.scene_meta.camera.width, 3))
    disparity_array_u8 = f32_to_u8(1. / (rays_depth + 1e-15)).reshape((state.scene_meta.camera.height, state.scene_meta.camera.width))
    if render_cost:
        cost_array_u8 = f32_to_u8(rays_cost.astype(jnp.float32) / (rays_cost.astype(jnp.float32).max() + 1.)).reshape((state.scene_meta.camera.height, state.scene_meta.camera.width))
    else:
        cost_array_u8 = None

    return bg_array_f32, image_array_u8, disparity_array_u8, cost_array_u8
