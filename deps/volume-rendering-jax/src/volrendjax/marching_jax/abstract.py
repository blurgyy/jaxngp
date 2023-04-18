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
    max_steps: int,
    K: int,
    G: int,
    bound: float,
    stepsize_portion: float,
):
    n_rays, _ = rays_o.shape

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
