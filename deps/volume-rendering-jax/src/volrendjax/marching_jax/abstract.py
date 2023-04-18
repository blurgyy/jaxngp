import chex
import jax
import jax.numpy as jnp


# jit rules
def march_rays_abstract(
    # arrays
    rays_o: jax.ShapedArray,
    rays_d: jax.ShapedArray,
    t_starts: jax.ShapedArray,
    t_ends: jax.ShapedArray,
    noises: jax.ShapedArray,
    occupancy_bitfield: jax.ShapedArray,

    # static args
    total_samples: int,
    diagonal_n_steps: int,
    K: int,
    G: int,
    bound: float,
    stepsize_portion: float,
):
    n_rays, _ = rays_o.shape

    chex.assert_shape([rays_o, rays_d], (n_rays, 3))
    chex.assert_shape([t_starts, t_ends, noises], (n_rays,))

    chex.assert_shape(occupancy_bitfield, (K*G*G*G//8,))
    chex.assert_type(occupancy_bitfield, jnp.uint8)

    dtype = jax.dtypes.canonicalize_dtype(rays_o.dtype)
    if dtype != jnp.float32:
        raise NotImplementedError(
            "march_rays is only implemented for input coordinates of `jnp.float32` type, got {}".format(
                dtype,
            )
        )

    shapes = {
        "helper.counter": (1,),

        "out.rays_n_samples": (n_rays,),
        "out.rays_sample_startidx": (n_rays,),
        "out.xyzs": (total_samples, 3),
        "out.dirs": (total_samples, 3),
        "out.dss": (total_samples,),
        "out.z_vals": (total_samples,),
    }

    return (
        jax.ShapedArray(shape=shapes["helper.counter"], dtype=jnp.uint32),
        jax.ShapedArray(shape=shapes["out.rays_n_samples"], dtype=jnp.uint32),
        jax.ShapedArray(shape=shapes["out.rays_sample_startidx"], dtype=jnp.uint32),
        jax.ShapedArray(shape=shapes["out.xyzs"], dtype=jnp.float32),
        jax.ShapedArray(shape=shapes["out.dirs"], dtype=jnp.float32),
        jax.ShapedArray(shape=shapes["out.dss"], dtype=jnp.float32),
        jax.ShapedArray(shape=shapes["out.z_vals"], dtype=jnp.float32),
    )
