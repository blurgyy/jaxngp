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

    chex.assert_scalar_positive(total_samples)
    chex.assert_scalar_positive(diagonal_n_steps)
    chex.assert_scalar_positive(K)
    chex.assert_scalar_positive(G)
    chex.assert_scalar_positive(bound)
    chex.assert_scalar_non_negative(stepsize_portion)

    dtype = jax.dtypes.canonicalize_dtype(rays_o.dtype)
    if dtype != jnp.float32:
        raise NotImplementedError(
            "march_rays is only implemented for input coordinates of `jnp.float32` type, got {}".format(
                dtype,
            )
        )

    shapes = {
        "helper.next_sample_write_location": (1,),
        "helper.n_valid_rays": (1,),

        "out.rays_n_samples": (n_rays,),
        "out.rays_sample_startidx": (n_rays,),
        "out.idcs": (total_samples,),
        "out.xyzs": (total_samples, 3),
        "out.dirs": (total_samples, 3),
        "out.dss": (total_samples,),
        "out.z_vals": (total_samples,),
    }

    return (
        jax.ShapedArray(shape=shapes["helper.next_sample_write_location"], dtype=jnp.uint32),
        jax.ShapedArray(shape=shapes["helper.n_valid_rays"], dtype=jnp.uint32),
        jax.ShapedArray(shape=shapes["out.rays_n_samples"], dtype=jnp.uint32),
        jax.ShapedArray(shape=shapes["out.rays_sample_startidx"], dtype=jnp.uint32),
        jax.ShapedArray(shape=shapes["out.idcs"], dtype=jnp.uint32),
        jax.ShapedArray(shape=shapes["out.xyzs"], dtype=jnp.float32),
        jax.ShapedArray(shape=shapes["out.dirs"], dtype=jnp.float32),
        jax.ShapedArray(shape=shapes["out.dss"], dtype=jnp.float32),
        jax.ShapedArray(shape=shapes["out.z_vals"], dtype=jnp.float32),
    )


def march_rays_inference_abstract(
    # arrays
    rays_o: jax.ShapedArray,
    rays_d: jax.ShapedArray,
    t_starts: jax.ShapedArray,
    t_ends: jax.ShapedArray,
    occupancy_bitfield: jax.ShapedArray,
    next_ray_index: jax.ShapedArray,
    terminated: jax.ShapedArray,
    indices_in: jax.ShapedArray,

    # static args
    diagonal_n_steps: int,
    K: int,
    G: int,
    march_steps_cap: int,
    bound: float,
    stepsize_portion: float,
):
    (n_total_rays, _), (n_rays,) = rays_o.shape, terminated.shape

    chex.assert_shape([rays_o, rays_d], (n_total_rays, 3))
    chex.assert_shape([t_starts, t_ends], (n_total_rays,))
    chex.assert_shape(occupancy_bitfield, (K*G*G*G//8,))
    chex.assert_type(occupancy_bitfield, jnp.uint8)
    chex.assert_shape(next_ray_index, (1,))
    chex.assert_shape([terminated, indices_in], (n_rays,))

    out_shapes = {
        "next_ray_index": (1,),
        "indices_out": (n_rays,),
        "n_samples": (n_rays,),
        "t_starts": (n_rays,),
        "xyzs": (n_rays, march_steps_cap, 3),
        "dss": (n_rays, march_steps_cap),
        "z_vals": (n_rays, march_steps_cap),
    }

    return (
        jax.ShapedArray(shape=out_shapes["next_ray_index"], dtype=jnp.uint32),
        jax.ShapedArray(shape=out_shapes["indices_out"], dtype=jnp.uint32),
        jax.ShapedArray(shape=out_shapes["n_samples"], dtype=jnp.uint32),
        jax.ShapedArray(shape=out_shapes["t_starts"], dtype=jnp.float32),
        jax.ShapedArray(shape=out_shapes["xyzs"], dtype=jnp.float32),
        jax.ShapedArray(shape=out_shapes["dss"], dtype=jnp.float32),
        jax.ShapedArray(shape=out_shapes["z_vals"], dtype=jnp.float32),
    )
