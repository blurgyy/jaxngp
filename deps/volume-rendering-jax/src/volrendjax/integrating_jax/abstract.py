import chex
import jax
import jax.numpy as jnp


# jit rules
def integrate_rays_abstract(
    transmittance_threshold: jax.Array,

    rays_sample_startidx: jax.Array,
    rays_n_samples: jax.Array,

    dss: jax.Array,
    z_vals: jax.Array,
    densities: jax.Array,
    rgbs: jax.Array,
):
    (n_rays,), (total_samples,) = transmittance_threshold.shape, dss.shape

    chex.assert_shape([rays_sample_startidx, rays_n_samples], (n_rays,))
    chex.assert_shape(z_vals, (total_samples,))
    chex.assert_shape(densities, (total_samples, 1))
    chex.assert_shape(rgbs, (total_samples, 3))

    dtype = jax.dtypes.canonicalize_dtype(rgbs.dtype)
    if dtype != jnp.float32:
        raise NotImplementedError(
            "integrate_rays is only implemented for input color of `jnp.float32` type, got {}".format(
                dtype,
            )
        )

    out_shapes = {
        "effective_samples": (n_rays,),
        "opacities": (n_rays,),
        "final_rgbs": (n_rays, 3),
        "depths": (n_rays,),
    }

    return (
        jax.ShapedArray(shape=out_shapes["effective_samples"], dtype=jnp.int32),  # effective_samples
        jax.ShapedArray(shape=out_shapes["opacities"], dtype=jnp.float32),  # opacities
        jax.ShapedArray(shape=out_shapes["final_rgbs"], dtype=jnp.float32),  # final_rgbs
        jax.ShapedArray(shape=out_shapes["depths"], dtype=jnp.float32),  # depths
    )

def integrate_rays_backward_abstract(
    transmittance_threshold: jax.Array,

    rays_sample_startidx: jax.Array,
    rays_n_samples: jax.Array,

    # original inputs
    dss: jax.Array,
    z_vals: jax.Array,
    densities: jax.Array,
    rgbs: jax.Array,

    # original outputs
    opacities: jax.Array,
    final_rgbs: jax.Array,
    depths: jax.Array,

    # gradient inputs
    dL_dopacities: jax.Array,
    dL_dfinal_rgbs: jax.Array,
    dL_ddepths: jax.Array,
):
    (n_rays,), (total_samples,) = transmittance_threshold.shape, dss.shape

    chex.assert_shape([rays_sample_startidx, rays_n_samples], (n_rays,))
    chex.assert_shape(z_vals, (total_samples,))
    chex.assert_shape(densities, (total_samples, 1))
    chex.assert_shape(rgbs, (total_samples, 3))
    chex.assert_shape([opacities, dL_dopacities], (n_rays,))
    chex.assert_shape([final_rgbs, dL_dfinal_rgbs], (n_rays, 3))
    chex.assert_shape([depths, dL_ddepths], (n_rays,))

    dtype = jax.dtypes.canonicalize_dtype(rgbs.dtype)
    if dtype != jnp.float32:
        raise NotImplementedError(
            "integrate_rays is only implemented for input color of `jnp.float32` type, got {}".format(
                dtype,
            )
        )

    out_shapes = {
        "dL_dz_vals": (total_samples,),
        "dL_ddensities": (total_samples, 1),
        "dL_drgbs": (total_samples, 3),
    }

    return (
        jax.ShapedArray(shape=out_shapes["dL_dz_vals"], dtype=jnp.float32),  # dL_dz_vals
        jax.ShapedArray(shape=out_shapes["dL_ddensities"], dtype=jnp.float32),  # dL_ddensities
        jax.ShapedArray(shape=out_shapes["dL_drgbs"], dtype=jnp.float32),  # dL_drgbs
    )
