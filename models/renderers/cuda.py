from collections.abc import Callable

from flax.core.scope import FrozenVariableDict
from flax.struct import dataclass
from icecream import ic
import jax
import jax.numpy as jnp
import jax.random as jran
from volrendjax import integrate_rays, march_rays

from utils.common import jit_jaxfn_with
from utils.types import AABB, DensityAndRGB, RayMarchingOptions


@dataclass
class OccupancyDensityGrid:
    # number of cascades
    # paper:
    #   𝐾 = 1 for all synthetic NeRF scenes (single grid) and 𝐾 ∈ [1, 5] for larger real-world
    #   scenes (up to 5 grids, depending on scene size)
    K: int
    # uint8, each bit is an occupancy value of a grid cell
    occupancy: jax.Array
    # float32, full-precision density values
    density: jax.Array

    @classmethod
    def create(cls, cascades: int, grid_resolution: int=128):
        """
        Example usage:
            ogrid = OccupancyDensityGrid.create(cascades=5, grid_resolution=128)
        """
        occupancy = jnp.zeros(
            shape=(cascades*grid_resolution**3 // 8,),  # every bit is an occupancy value
            dtype=jnp.uint8,
        )
        density = jnp.zeros(
            shape=(cascades*grid_resolution**3,),
            dtype=jnp.float32,
        )
        return cls(K=cascades, occupancy=occupancy, density=density)


def make_near_far_from_aabb(
        aabb: AABB,  # [3, 2]
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

    # [n_rays, 1]
    tx0, tx1 = (
        (aabb[0][0] - o[:, 0:1]) / d[:, 0:1],
        (aabb[0][1] - o[:, 0:1]) / d[:, 0:1],
    )
    ty0, ty1 = (
        (aabb[1][0] - o[:, 1:2]) / d[:, 1:2],
        (aabb[1][1] - o[:, 1:2]) / d[:, 1:2],
    )
    tz0, tz1 = (
        (aabb[2][0] - o[:, 2:3]) / d[:, 2:3],
        (aabb[2][1] - o[:, 2:3]) / d[:, 2:3],
    )
    tx_start, tx_end = jnp.minimum(tx0, tx1), jnp.maximum(tx0, tx1)
    ty_start, ty_end = jnp.minimum(ty0, ty1), jnp.maximum(ty0, ty1)
    tz_start, tz_end = jnp.minimum(tz0, tz1), jnp.maximum(tz0, tz1)

    # when t_start<0, or t_start>t_end, ray does not intersect with aabb, these cases are handled in
    # the `march_rays` implementation
    t_start = jnp.maximum(jnp.maximum(tx_start, ty_start), tz_start)  # last axis that gose inside the bbox
    t_end = jnp.minimum(jnp.minimum(tx_end, ty_end), tz_end)  # first axis that goes out of the bbox

    # [n_rays, 1], [n_rays, 1]
    return t_start, t_end


@jit_jaxfn_with(static_argnames=["options", "nerf_fn"])
def render_rays(
    K: jran.KeyArray,
    o_world: jax.Array,
    d_world: jax.Array,
    aabb: AABB,
    options: RayMarchingOptions,
    param_dict: FrozenVariableDict,
    nerf_fn: Callable[[FrozenVariableDict, jax.Array, jax.Array], DensityAndRGB],
):
    """
    Given a pack of rays, do:
        1. generate samples on each ray
        2. predict densities and RGBs on each sample
        3. composite the predicted values along each ray to get rays' opacities, final colors, and
           estimated depths

    Inputs:
        o_world [n_rays, 3]: ray origins, in world space
        d_world [n_rays, 3]: ray directions (unit vectors), in world space
        aabb [3, 2]: scene bounds on each of x, y, z axes
        options: see :class:`RayMarchingOptions`
        param_dict: :class:`NeRF` model params
        nerf_fn: function that takes the param_dict, xyz, and viewing directions as inputs, and
                 outputs densities and rgbs.

    Returns:
        opacities [n_rays]: accumulated opacities along each ray
        rgbs [n_rays, 3]: rendered colors
        depths [n_rays]: rays' expected termination depths
    """
    if options.n_importance > 0:
        raise NotImplementedError(
            "CUDA-based raymarching is not designed for importance sampling.\n"
            "Set n_importance to 0 to use CUDA-based raymarching"
        )

    # make sure d_world is normalized
    d_world /= jnp.linalg.norm(d_world, axis=-1, keepdims=True) + 1e-15
    # skip the empty space between camera and scene bbox
    # [n_rays], [n_rays]
    t_starts, t_ends = make_near_far_from_aabb(
        aabb=aabb,
        o=o_world,
        d=d_world
    )

    rays_n_samples, ray_pts, ray_dirs, dss, z_vals = march_rays(
        max_n_samples=options.steps,
        K=1,
        G=128,
        bound=1,
        stepsize_portion=1/256,
        rays_o=o_world,
        rays_d=d_world,
        t_starts=t_starts.ravel(),
        t_ends=t_ends.ravel(),
        occupancy_bitfield=jnp.ones((128**3)//8, dtype=jnp.uint8) * 255,
    )

    densities, rgbs = nerf_fn(
        param_dict,
        ray_pts,
        ray_dirs,
    )

    rays_sample_startidx = jnp.concatenate(
        [jnp.zeros_like(rays_n_samples[:1]), jnp.cumsum(options.steps * jnp.ones_like(rays_n_samples[:-1]))],
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

    return opacities, final_rgbs, depths
