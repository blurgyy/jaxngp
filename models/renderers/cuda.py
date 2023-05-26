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
from utils.types import (
    NeRFState,
    PinholeCamera,
    RigidTransformation,
)

from ._utils import make_rays_worldspace


def make_near_far_from_bound(
    bound: float,
    o: jax.Array,  # [n_rays, 3]
    d: jax.Array,  # [n_rays, 3]
):
    "Calculates near and far intersections with the bounding box [-bound, bound]^3 for each ray."

    # make sure d is normalized
    d /= jnp.linalg.norm(d, axis=-1, keepdims=True) + 1e-15

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
    bg: jax.Array,
    total_samples: int,
    state: NeRFState,
):
    # make sure d_world is normalized
    d_world /= jnp.linalg.norm(d_world, axis=-1, keepdims=True) + 1e-15
    # skip the empty space between camera and scene bbox
    # [n_rays], [n_rays]
    t_starts, t_ends = make_near_far_from_bound(
        bound=state.scene_meta.bound,
        o=o_world,
        d=d_world
    )

    if state.raymarch.perturb:
        KEY, key = jran.split(KEY, 2)
        noises = jran.uniform(key, shape=t_starts.shape, dtype=t_starts.dtype, minval=0., maxval=1.)
    else:
        noises = 0.
    measured_batch_size_before_compaction, rays_n_samples, rays_sample_startidx, ray_pts, ray_dirs, dss, z_vals = march_rays(
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

    drgbs = state.nerf_fn(
        {"params": state.params["nerf"]},
        ray_pts,
        ray_dirs,
    )

    effective_samples, final_rgbds = integrate_rays(
        rays_sample_startidx=rays_sample_startidx,
        rays_n_samples=rays_n_samples,
        bgs=bg,
        dss=dss,
        z_vals=z_vals,
        drgbs=drgbs,
    )

    batch_metrics = {
        "measured_batch_size_before_compaction": measured_batch_size_before_compaction,
        "measured_batch_size": jnp.where(effective_samples > 0, effective_samples, 0).sum(),
    }

    return batch_metrics, final_rgbds


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

    counter: jax.Array,
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
    counter, indices, n_samples, t_starts, xyzs, dss, z_vals = march_rays_inference(
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
        counter=counter,
        terminated=terminated,
        indices=indices,
    )
    if rays_cost is not None:
        rays_cost = rays_cost.at[indices].set(rays_cost[indices] + n_samples)

    xyzs = jax.lax.stop_gradient(xyzs)
    drgbs = payload.nerf_fn(
        {"params": locked_nerf_params},
        xyzs,
        jnp.broadcast_to(rays_d[indices, None, :], xyzs.shape),
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

    return terminate_cnt, terminated, counter, indices, t_starts, rays_rgbd, rays_T, rays_cost


def render_image_inference(
    KEY: jran.KeyArray,
    transform_cw: RigidTransformation,
    state: NeRFState,
    camera_override: None | PinholeCamera=None,
    render_cost: bool=False,
):
    if camera_override is not None:
        state = state.replace(scene_meta=state.scene_meta.replace(camera=camera_override))

    o_world, d_world = make_rays_worldspace(camera=state.scene_meta.camera, transform_cw=transform_cw)
    t_starts, t_ends = make_near_far_from_bound(state.scene_meta.bound, o_world, d_world)
    rays_rgbd = jnp.zeros((state.scene_meta.camera.n_pixels, 4), dtype=jnp.float32)
    rays_T = jnp.ones(state.scene_meta.camera.n_pixels, dtype=jnp.float32)
    if render_cost:
        rays_cost = jnp.zeros(state.scene_meta.camera.n_pixels, dtype=jnp.uint32)
    else:
        rays_cost = None
    if state.use_background_model:
        bg = state.bg_fn({"params": state.locked_params["bg"]}, o_world, d_world)
    elif state.render.random_bg:
        KEY, key = jran.split(KEY, 2)
        bg = jran.uniform(key, (3,), dtype=jnp.float32, minval=0, maxval=1)
    else:
        bg = state.render.bg
    rays_bg = jnp.broadcast_to(jnp.asarray(bg), (state.scene_meta.camera.n_pixels, 3))

    o_world, d_world, t_starts, t_ends, rays_bg, rays_rgbd, rays_T = map(
        jax.lax.stop_gradient,
        [o_world, d_world, t_starts, t_ends, rays_bg, rays_rgbd, rays_T],
    )

    if state.batch_config.mean_effective_samples_per_ray > 7:
        march_steps_cap = max(4, min(state.batch_config.mean_effective_samples_per_ray // 2 + 1, 8))
    else:
        march_steps_cap = min(4, state.batch_config.mean_effective_samples_per_ray)
    march_steps_cap = int(march_steps_cap)
    n_rays = min(65536 // march_steps_cap, o_world.shape[0])

    counter = jnp.zeros(1, dtype=jnp.uint32)
    terminated = jnp.ones(n_rays, dtype=jnp.bool_)  # all rays are terminated at the beginning
    indices = jnp.zeros(n_rays, dtype=jnp.uint32)
    n_rendered_rays = 0

    while n_rendered_rays < state.scene_meta.camera.n_pixels:
        iters = max(1, (state.scene_meta.camera.n_pixels - n_rendered_rays) // n_rays)
        iters = 2 ** int(math.log2(iters) + 1)

        for _ in range(iters):
            terminate_cnt, terminated, counter, indices, t_starts, rays_rgbd, rays_T, rays_cost = march_and_integrate_inference(
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

                counter=counter,
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

    bg_array_f32 = rays_bg.reshape((state.scene_meta.camera.H, state.scene_meta.camera.W, 3))
    rays_rgb, rays_depth = jnp.array_split(rays_rgbd, [3], axis=-1)
    image_array_u8 = f32_to_u8(rays_rgb).reshape((state.scene_meta.camera.H, state.scene_meta.camera.W, 3))
    rays_depth_f32 = rays_depth / (state.scene_meta.bound * 2 + jnp.linalg.norm(transform_cw.translation))
    depth_array_u8 = f32_to_u8(rays_depth_f32).reshape((state.scene_meta.camera.H, state.scene_meta.camera.W))
    if render_cost:
        cost_array_u8 = f32_to_u8(rays_cost.astype(jnp.float32) / (rays_cost.astype(jnp.float32).max() + 1.)).reshape((state.scene_meta.camera.H, state.scene_meta.camera.W))
    else:
        cost_array_u8 = None

    return bg_array_f32, image_array_u8, depth_array_u8, cost_array_u8
