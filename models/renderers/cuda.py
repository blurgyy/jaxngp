from collections.abc import Callable

import chex
from flax.core.scope import FrozenVariableDict
from icecream import ic
import jax
import jax.numpy as jnp
import jax.random as jran
from tqdm import tqdm
from volrendjax import integrate_rays, march_rays, morton3d_invert, packbits

from utils.common import jit_jaxfn_with, tqdm_format
from utils.data import blend_alpha_channel, cascades_from_bound, set_pixels
from utils.types import (
    DensityAndRGB,
    NeRFBatchConfig,
    NeRFTrainState,
    OccupancyDensityGrid,
    PinholeCamera,
    RayMarchingOptions,
    RenderingOptions,
    RigidTransformation,
)

from ._utils import get_indices_chunks, make_rays_worldspace


@jit_jaxfn_with(static_argnames=["update_all", "bound", "raymarch"])
def update_ogrid(
    KEY: jran.KeyArray,
    update_all: bool,
    bound: float,
    raymarch: RayMarchingOptions,
    state: NeRFTrainState,
) -> NeRFTrainState:
    # (1) decay the density value in each grid cell by a factor of 0.95.
    decay = .95
    density_grid = state.ogrid.density * decay

    # (2) randomly sample 𝑀 candidate cells, and set their value to the maximum of their current
    # value and the density component of the NeRF model at a random location within the cell.
    G3 = raymarch.density_grid_res ** 3
    cascades = cascades_from_bound(bound)
    for cas in range(cascades):
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
            KEY, key = jran.split(KEY, 2)
            indices_firsthalf = jran.choice(key=key, a=jnp.arange(G3, dtype=jnp.uint32), shape=(M//2,), replace=True)  # allow duplicated choices
            # Rejection sampling is used for the remaining samples to restrict selection to cells
            # that are currently occupied.
            # NOTE: Below is just uniformly sampling the occupied cells, not rejection sampling.
            cas_occ_mask = state.ogrid.occ_mask[cas*G3:(cas+1)*G3]
            p = cas_occ_mask.astype(jnp.float32)
            KEY, key = jran.split(KEY, 2)
            indices_secondhalf = jran.choice(key=key, a=jnp.arange(G3, dtype=jnp.uint32), shape=(M//2,), replace=True, p=p)
            indices = jnp.concatenate([indices_firsthalf, indices_secondhalf])

        coordinates = morton3d_invert(indices).astype(jnp.float32)
        coordinates = coordinates / (raymarch.density_grid_res - 1) * 2 - 1  # in [-1, 1]
        bound = min(bound, 2**cas)
        half_cell_width = bound / raymarch.density_grid_res
        coordinates *= bound - half_cell_width  # in [-bound+half_cell_width, bound-half_cell_width]
        # random point inside grid cells
        KEY, key = jran.split(KEY, 2)
        coordinates += jran.uniform(
            key,
            coordinates.shape,
            coordinates.dtype,
            minval=-half_cell_width,
            maxval=half_cell_width,
        )

        new_densities = state.apply_fn(
            {"params": state.params},
            coordinates,
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
    density_threshold = .01 * raymarch.max_steps / (2 * 3**.5)
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
    "Finds a smallest non-negative `t` for each ray, such that o+td is inside the given aabb."

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

    # [n_rays], [n_rays]
    return t_start, t_end


@jit_jaxfn_with(static_argnames=["bound", "options", "max_n_samples", "nerf_fn"])
def render_rays(
    KEY: jran.KeyArray,
    o_world: jax.Array,
    d_world: jax.Array,
    bound: float,
    ogrid: OccupancyDensityGrid,
    options: RayMarchingOptions,
    max_n_samples: int,
    param_dict: FrozenVariableDict,
    nerf_fn: Callable[[FrozenVariableDict, jax.Array, jax.Array], DensityAndRGB],
):
    if options.n_importance > 0:
        raise NotImplementedError(
            "CUDA-based raymarching is not designed for importance sampling.\n"
            "Set n_importance to 0 to use CUDA-based raymarching"
        )

    # make sure d_world is normalized
    d_world /= jnp.linalg.norm(d_world, axis=-1, keepdims=True) + 1e-15
    # skip the empty space between camera and scene bbox
    # [n_rays], [n_rays]
    t_starts, t_ends = make_near_far_from_bound(
        bound=bound,
        o=o_world,
        d=d_world
    )

    if options.perturb:
        KEY, key = jran.split(KEY, 2)
        noises = jran.uniform(key, shape=t_starts.shape, dtype=t_starts.dtype, minval=0., maxval=1.)
    else:
        noises = 0.
    rays_n_samples, ray_pts, ray_dirs, dss, z_vals = march_rays(
        max_n_samples=max_n_samples,
        max_steps=options.max_steps,
        K=cascades_from_bound(bound),
        G=options.density_grid_res,
        bound=bound,
        stepsize_portion=options.stepsize_portion,
        rays_o=o_world,
        rays_d=d_world,
        t_starts=t_starts.ravel(),
        t_ends=t_ends.ravel(),
        noises=noises,
        occupancy_bitfield=ogrid.occupancy,
    )

    densities, rgbs = nerf_fn(
        param_dict,
        ray_pts,
        ray_dirs,
    )

    rays_sample_startidx = jnp.concatenate(
        [jnp.zeros_like(rays_n_samples[:1]), jnp.cumsum(max_n_samples * jnp.ones_like(rays_n_samples[:-1]))],
        axis=-1,
    )

    effective_samples, opacities, final_rgbs, depths = integrate_rays(
        1e-4,
        rays_sample_startidx,
        rays_n_samples,
        dss,
        z_vals,
        densities,
        rgbs,
    )

    return effective_samples, opacities, final_rgbs, depths


def render_image(
    KEY: jran.KeyArray,
    bound: float,
    camera: PinholeCamera,
    transform_cw: RigidTransformation,
    options: RenderingOptions,
    raymarch_options: RayMarchingOptions,
    batch_config: NeRFBatchConfig,
    ogrid: OccupancyDensityGrid,
    param_dict: FrozenVariableDict,
    nerf_fn: Callable[[FrozenVariableDict, jax.Array, jax.Array], DensityAndRGB],
):
    chex.assert_shape([transform_cw.rotation, transform_cw.translation], [(3, 3), (3,)])

    o_world, d_world = make_rays_worldspace(camera=camera, transform_cw=transform_cw)

    KEY, key = jran.split(KEY, 2)
    xys, indices = get_indices_chunks(key, camera.H, camera.W, int(batch_config.n_rays))

    image_array = jnp.empty((camera.H, camera.W, 3), dtype=jnp.uint8)
    depth_array = jnp.empty((camera.H, camera.W), dtype=jnp.uint8)
    for idcs in tqdm(indices, desc="| rendering {}x{} image".format(camera.W, camera.H), bar_format=tqdm_format):
        KEY, key = jran.split(KEY, 2)
        _, opacities, rgbs, depths = render_rays(
            key,
            o_world[idcs],
            d_world[idcs],
            bound,
            ogrid,
            raymarch_options,
            int(batch_config.n_samples_per_ray),
            param_dict,
            nerf_fn,
        )
        if options.random_bg:
            KEY, key = jran.split(KEY, 2)
            bg = jran.uniform(key, rgbs.shape, rgbs.dtype, minval=0, maxval=1)
        else:
            bg = options.bg
        rgbs = blend_alpha_channel(jnp.concatenate([rgbs, opacities[:, None]], axis=-1), bg=bg)
        depths = depths / (bound * 2 + jnp.linalg.norm(transform_cw.translation))
        image_array = set_pixels(image_array, xys, idcs, rgbs)
        depth_array = set_pixels(depth_array, xys, idcs, depths)

    return image_array, depth_array
