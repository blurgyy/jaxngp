import chex
import jax
import jax.numpy as jnp


# jit rules
def integrate_rays_abstract(
    rays_sample_startidx: jax.Array,
    rays_n_samples: jax.Array,

    bgs: jax.Array,
    dss: jax.Array,
    z_vals: jax.Array,
    drgbs: jax.Array,
):
    (n_rays,), (total_samples,) = rays_sample_startidx.shape, dss.shape

    chex.assert_shape([rays_sample_startidx, rays_n_samples], (n_rays,))
    chex.assert_shape(bgs, (n_rays, 3))
    chex.assert_shape(z_vals, (total_samples,))
    chex.assert_shape(drgbs, (total_samples, 4))

    dtype = jax.dtypes.canonicalize_dtype(drgbs.dtype)
    if dtype != jnp.float32:
        raise NotImplementedError(
            "integrate_rays is only implemented for input prediction (density, color) of `jnp.float32` type, got {}".format(
                dtype,
            )
        )

    shapes = {
        "helper.counter": (1,),

        "out.final_rgbs": (n_rays, 3),
        "out.depths": (n_rays,),
    }

    return (
        jax.ShapedArray(shape=shapes["helper.counter"], dtype=jnp.uint32),

        jax.ShapedArray(shape=shapes["out.final_rgbs"], dtype=jnp.float32),
        jax.ShapedArray(shape=shapes["out.depths"], dtype=jnp.float32),
    )

def integrate_rays_backward_abstract(
    rays_sample_startidx: jax.Array,
    rays_n_samples: jax.Array,

    # original inputs
    bgs: jax.Array,
    dss: jax.Array,
    z_vals: jax.Array,
    drgbs: jax.Array,

    # original outputs
    final_rgbs: jax.Array,
    depths: jax.Array,

    # gradient inputs
    dL_dfinal_rgbs: jax.Array,
    dL_ddepths: jax.Array,
):
    (n_rays,), (total_samples,) = rays_sample_startidx.shape, dss.shape

    chex.assert_shape([rays_sample_startidx, rays_n_samples], (n_rays,))
    chex.assert_shape(bgs, (n_rays, 3))
    chex.assert_shape(z_vals, (total_samples,))
    chex.assert_shape(drgbs, (total_samples, 4))
    chex.assert_shape([final_rgbs, dL_dfinal_rgbs], (n_rays, 3))
    chex.assert_shape([depths, dL_ddepths], (n_rays,))

    dtype = jax.dtypes.canonicalize_dtype(drgbs.dtype)
    if dtype != jnp.float32:
        raise NotImplementedError(
            "integrate_rays is only implemented for input color of `jnp.float32` type, got {}".format(
                dtype,
            )
        )

    out_shapes = {
        "dL_dbgs": (n_rays, 3),
        "dL_dz_vals": (total_samples,),
        "dL_ddrgbs": (total_samples, 4),
    }

    return (
        jax.ShapedArray(shape=out_shapes["dL_dbgs"], dtype=jnp.float32),
        jax.ShapedArray(shape=out_shapes["dL_dz_vals"], dtype=jnp.float32),
        jax.ShapedArray(shape=out_shapes["dL_ddrgbs"], dtype=jnp.float32),
    )


def integrate_rays_inference_abstract(
    rays_bg: jax.ShapedArray,
    rays_rgb: jax.ShapedArray,
    rays_T: jax.ShapedArray,
    rays_depth: jax.ShapedArray,

    n_samples: jax.ShapedArray,
    indices: jax.ShapedArray,
    dss: jax.ShapedArray,
    z_vals: jax.ShapedArray,
    drgbs: jax.ShapedArray,
):
    (n_total_rays, _), (n_rays, march_steps_cap) = rays_rgb.shape, dss.shape

    chex.assert_shape([rays_bg, rays_rgb], (n_total_rays, 3))
    chex.assert_shape([rays_T, rays_depth], (n_total_rays,))
    chex.assert_shape([n_samples, indices], (n_rays,))
    chex.assert_shape([dss, z_vals], (n_rays, march_steps_cap))
    chex.assert_shape(drgbs, (n_rays, march_steps_cap, 4))

    out_shapes = {
        "terminate_cnt": (1,),
        "terminated": (n_rays,),
        "rays_rgb": (n_rays, 3),
        "rays_T": (n_rays,),
        "rays_depth": (n_rays,),
    }

    return (
        jax.ShapedArray(shape=out_shapes["terminate_cnt"], dtype=jnp.uint32),
        jax.ShapedArray(shape=out_shapes["terminated"], dtype=jnp.bool_),
        jax.ShapedArray(shape=out_shapes["rays_rgb"], dtype=jnp.float32),
        jax.ShapedArray(shape=out_shapes["rays_T"], dtype=jnp.float32),
        jax.ShapedArray(shape=out_shapes["rays_depth"], dtype=jnp.float32),
    )
