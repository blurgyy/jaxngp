import jax
import jax.numpy as jnp


# jit rules
def march_rays_abstract(
    # arrays
    rays_o: jax.ShapedArray,
    rays_d: jax.ShapedArray,
    t_starts: jax.ShapedArray,
    t_ends: jax.ShapedArray,
    occupancy_bitfield: jax.ShapedArray,

    # static args
    max_n_samples: int,
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

    out_shapes = {
        "rays_n_samples": (n_rays,),
        "valid_mask": (n_rays * max_n_samples,),
        "xyzs": (n_rays * max_n_samples, 3),
        "dirs": (n_rays * max_n_samples, 3),
        "dss": (n_rays * max_n_samples,),
        "z_vals": (n_rays * max_n_samples,),
    }

    return (
        jax.ShapedArray(shape=out_shapes["rays_n_samples"], dtype=jnp.uint32),
        jax.ShapedArray(shape=out_shapes["valid_mask"], dtype=jnp.bool_),
        jax.ShapedArray(shape=out_shapes["xyzs"], dtype=jnp.float32),
        jax.ShapedArray(shape=out_shapes["dirs"], dtype=jnp.float32),
        jax.ShapedArray(shape=out_shapes["dss"], dtype=jnp.float32),
        jax.ShapedArray(shape=out_shapes["z_vals"], dtype=jnp.float32),
    )
