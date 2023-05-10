import math

import jax
import jax.numpy as jnp
import jax.random as jran
from volrendjax import (
    integrate_rays,
    integrate_rays_inference,
    march_rays,
    march_rays_inference,
    morton3d_invert,
    packbits,
)

from utils.common import jit_jaxfn_with
from utils.types import (
    NeRFState,
    OccupancyDensityGrid,
    RigidTransformation,
)

from ._utils import make_rays_worldspace


@jit_jaxfn_with(static_argnames=["update_all"])
def update_ogrid(
    KEY: jran.KeyArray,
    update_all: bool,
    state: NeRFState,
) -> NeRFState:
    # (1) decay the density value in each grid cell by a factor of 0.95.
    decay = .95
    density_grid = state.ogrid.density * decay

    # (2) randomly sample 𝑀 candidate cells, and set their value to the maximum of their current
    # value and the density component of the NeRF model at a random location within the cell.
    G3 = state.raymarch.density_grid_res ** 3
    for cas in range(state.scene_meta.cascades):
        if update_all:
            # During the first 256 training steps, we sample 𝑀 = 𝐾 · 128^{3} cells uniformly without
            # repetition.
            M = G3
            indices = jnp.arange(M, dtype=jnp.uint32)
        else:
            # For subsequent training steps we set 𝑀 = 𝐾 · 128^{3}/2 which we partition into two
            # sets.
            M = G3 // 2
            # The first 𝑀/2 cells are sampled uniformly among all cells.
            KEY, key_firsthalf, key_secondhalf = jran.split(KEY, 3)
            indices_firsthalf = jran.choice(key=key_firsthalf, a=jnp.arange(G3, dtype=jnp.uint32), shape=(M//2,), replace=True)  # allow duplicated choices
            # Rejection sampling is used for the remaining samples to restrict selection to cells
            # that are currently occupied.
            # NOTE: Below is just uniformly sampling the occupied cells, not rejection sampling.
            cas_occ_mask = state.ogrid.occ_mask[cas*G3:(cas+1)*G3]
            p = cas_occ_mask.astype(jnp.float32)
            indices_secondhalf = jran.choice(key=key_secondhalf, a=jnp.arange(G3, dtype=jnp.uint32), shape=(M//2,), replace=True, p=p)
            indices = jnp.concatenate([indices_firsthalf, indices_secondhalf])

        coordinates = morton3d_invert(indices).astype(jnp.float32)
        coordinates = coordinates / (state.raymarch.density_grid_res - 1) * 2 - 1  # in [-1, 1]
        mip_bound = min(state.scene_meta.bound, 2**cas)
        half_cell_width = mip_bound / state.raymarch.density_grid_res
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

        new_densities = state.nerf_fn(
            {"params": state.locked_params["nerf"]},
            jax.lax.stop_gradient(coordinates),
            None,
        )

        # set their value to the maximum of their current value and the density component of the
        # NeRF model at a random location within the cell.
        density_grid = density_grid.at[indices + cas*G3].set(
            jnp.maximum(density_grid[indices + cas*G3], new_densities.ravel())
        )

    # (3) update the occupancy bits by thresholding each cell’s density with 𝑡 = 0.01 · 1024/√3,
    # which corresponds to thresholding the opacity of a minimal ray marching step by 1 − exp(−0.01)
    # ≈ 0.01.
    density_threshold = .01 * state.raymarch.diagonal_n_steps / (2 * min(state.scene_meta.bound, 1) * 3**.5)
    mean_density = jnp.sum(jnp.where(density_grid > 0, density_grid, 0)) / jnp.sum(jnp.where(density_grid > 0, 1, 0))
    density_threshold = jnp.minimum(density_threshold, mean_density)
    # density_threshold = 1e-2
    occupied_mask, occupancy_bitfield = packbits(
        density_threshold=density_threshold,
        density_grid=density_grid,
    )

    return state.replace(ogrid=OccupancyDensityGrid(
        density=density_grid,
        occ_mask=occupied_mask,
        occupancy=occupancy_bitfield,
    ))


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

    densities, rgbs = state.nerf_fn(
        {"params": state.params["nerf"]},
        ray_pts,
        ray_dirs,
    )

    effective_samples, opacities, final_rgbs, depths = integrate_rays(
        rays_sample_startidx=rays_sample_startidx,
        rays_n_samples=rays_n_samples,
        bgs=bg,
        dss=dss,
        z_vals=z_vals,
        densities=densities,
        rgbs=rgbs,
    )

    batch_metrics = {
        "measured_batch_size_before_compaction": measured_batch_size_before_compaction,
        "measured_batch_size": jnp.where(effective_samples > 0, effective_samples, 0).sum(),
    }

    return batch_metrics, opacities, final_rgbs, depths


@jit_jaxfn_with(static_argnames=["march_steps_cap"])
def march_and_integrate_inference(
    march_steps_cap: int,
    state: NeRFState,

    counter: jax.Array,
    rays_o: jax.Array,
    rays_d: jax.Array,
    t_starts: jax.Array,
    t_ends: jax.Array,
    occupancy_bitfield: jax.Array,
    terminated: jax.Array,
    indices: jax.Array,

    rays_bg: jax.Array,
    rays_rgb: jax.Array,
    rays_T: jax.Array,
    rays_depth: jax.Array,
):
    counter, indices, n_samples, t_starts, xyzdirs, dss, z_vals = march_rays_inference(
        diagonal_n_steps=state.raymarch.diagonal_n_steps,
        K=state.scene_meta.cascades,
        G=state.raymarch.density_grid_res,
        march_steps_cap=march_steps_cap,
        bound=state.scene_meta.bound,
        stepsize_portion=state.scene_meta.stepsize_portion,
        rays_o=rays_o,
        rays_d=rays_d,
        t_starts=t_starts,
        t_ends=t_ends,
        occupancy_bitfield=occupancy_bitfield,
        counter=counter,
        terminated=terminated,
        indices=indices,
    )

    xyzdirs = jax.lax.stop_gradient(xyzdirs)
    densities, rgbs = state.nerf_fn(
        {"params": state.locked_params["nerf"]},
        xyzdirs[..., :3],
        xyzdirs[..., 3:],
    )

    terminate_cnt, terminated, rays_rgb, rays_T, rays_depth = integrate_rays_inference(
        rays_bg=rays_bg,
        rays_rgb=rays_rgb,
        rays_T=rays_T,
        rays_depth=rays_depth,

        n_samples=n_samples,
        indices=indices,
        dss=dss,
        z_vals=z_vals,
        densities=densities,
        rgbs=rgbs,
    )

    return terminate_cnt, terminated, counter, indices, t_starts, rays_rgb, rays_T, rays_depth


def render_image_inference(
    KEY: jran.KeyArray,
    transform_cw: RigidTransformation,
    state: NeRFState,
):
    o_world, d_world = make_rays_worldspace(camera=state.scene_meta.camera, transform_cw=transform_cw)
    t_starts, t_ends = make_near_far_from_bound(state.scene_meta.bound, o_world, d_world)
    rays_rgb = jnp.zeros((state.scene_meta.camera.n_pixels, 3), dtype=jnp.float32)
    rays_T = jnp.ones(state.scene_meta.camera.n_pixels, dtype=jnp.float32)
    rays_depth = jnp.zeros(state.scene_meta.camera.n_pixels, dtype=jnp.float32)
    if state.use_background_model:
        bg = state.bg_fn({"params": state.locked_params["bg"]}, o_world, d_world)
    elif state.render.random_bg:
        KEY, key = jran.split(KEY, 2)
        bg = jran.uniform(key, (3,), dtype=jnp.float32, minval=0, maxval=1)
    else:
        bg = state.render.bg
    rays_bg = jnp.broadcast_to(jnp.asarray(bg), rays_rgb.shape)

    o_world, d_world, t_starts, t_ends, rays_bg, rays_rgb, rays_T, rays_depth = map(
        jax.lax.stop_gradient,
        [o_world, d_world, t_starts, t_ends, rays_bg, rays_rgb, rays_T, rays_depth],
    )

    if state.batch_config.mean_effective_samples_per_ray > 7:
        march_rays_cap = max(4, min(state.batch_config.mean_effective_samples_per_ray // 2 + 1, 8))
    else:
        march_rays_cap = min(4, state.batch_config.mean_effective_samples_per_ray)
    march_rays_cap = int(march_rays_cap)
    n_rays = min(65536, o_world.shape[0]) // march_rays_cap

    counter = jnp.zeros(1, dtype=jnp.uint32)
    terminated = jnp.ones(n_rays, dtype=jnp.bool_)  # all rays are terminated at the beginning
    indices = jnp.zeros(n_rays, dtype=jnp.uint32)
    n_rendered_rays = 0

    while n_rendered_rays < state.scene_meta.camera.n_pixels:
        iters = max(1, (state.scene_meta.camera.n_pixels - n_rendered_rays) // n_rays)
        iters = 2 ** int(math.log2(iters))

        terminate_cnt = 0
        for _ in range(iters):
            iter_terminate_cnt, terminated, counter, indices, t_starts, rays_rgb, rays_T, rays_depth = march_and_integrate_inference(
                march_steps_cap=march_rays_cap,
                state=state,

                counter=counter,
                rays_o=o_world,
                rays_d=d_world,
                t_starts=t_starts,
                t_ends=t_ends,
                occupancy_bitfield=state.ogrid.occupancy,
                terminated=terminated,
                indices=indices,

                rays_bg=rays_bg,
                rays_rgb=rays_rgb,
                rays_T=rays_T,
                rays_depth=rays_depth,
            )
            terminate_cnt += iter_terminate_cnt
        n_rendered_rays += terminate_cnt

    bg_array_f32 = rays_bg.reshape((state.scene_meta.camera.H, state.scene_meta.camera.W, 3))
    image_array_u8 = jnp.clip(jnp.round(rays_rgb * 255), 0, 255).astype(jnp.uint8).reshape((state.scene_meta.camera.H, state.scene_meta.camera.W, 3))
    rays_depth_f32 = rays_depth / (state.scene_meta.bound * 2 + jnp.linalg.norm(transform_cw.translation))
    depth_array_u8 = jnp.clip(jnp.round(rays_depth_f32 * 255), 0, 255).astype(jnp.uint8).reshape((state.scene_meta.camera.H, state.scene_meta.camera.W))

    return bg_array_f32, image_array_u8, depth_array_u8
