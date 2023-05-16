from typing import Tuple

import jax

from . import impl


# this function is a wrapper on top of `__integrate_rays` which has custom vjp (wrapping the
# `__integrate_rays` function because the @jax.custom_vjp decorator makes the decorated function's
# docstring invisible to LSPs).
def integrate_rays(
    rays_sample_startidx: jax.Array,
    rays_n_samples: jax.Array,
    bgs: jax.Array,
    dss: jax.Array,
    z_vals: jax.Array,
    densities: jax.Array,
    rgbs: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Inputs:
        rays_sample_startidx `[n_rays]`: i-th element is the index of the first sample in z_vals,
                                         densities, and rgbs of the i-th ray
        rays_n_samples `[n_rays]`: i-th element is the number of samples for the i-th ray

        bgs `[n_rays, 3]`: background colors of each ray
        dss [total_samples]: it means `ds`s, the notation `ds` comes from the article "Local and
                             global illumination in the volume rendering integral" written by Nelson
                             Max and Min Chen, 2005.  The product of `ds[i]` and `densities[i]`
                             represents the probability of the ray terminates anywhere between
                             `z_vals[i]` and `z_vals[i]+ds[i]`.
                             Note that `ds[i]` is _not_ the same as `z_vals[i+1]-z_vals[i]` (though
                             they may equal), because: (1) if empty spaces are skipped during ray
                             marching, `z_vals[i+1]-z_vals[i]` may be very large, in which case it's
                             no longer appropriate to assume the density is constant along this
                             large segment; (2) `z_vals[i+1]` is not defined for the last sample.
        z_vals [total_samples]: z_vals[i] is the distance of the i-th sample from the camera
        densities [total_samples, 1]: density values along a ray
        rgbs [total_samples, 3]: rgb values along a ray

    Returns:
        measured_batch_size `uint`: total number of samples that got composited into output
        final_rgbs `[n_rays, 3]~: integrated ray colors according to input densities and rgbs.
        depths `[n_rays]`: estimated termination depth of each ray
    """
    counter, final_rgbs, depths = impl.__integrate_rays(
        rays_sample_startidx,
        rays_n_samples,
        bgs,
        dss,
        z_vals,
        densities,
        rgbs,
    )

    return counter[0], final_rgbs, depths


def integrate_rays_inference(
    rays_bg: jax.Array,
    rays_rgb: jax.Array,
    rays_T: jax.Array,
    rays_depth: jax.Array,

    n_samples: jax.Array,
    indices: jax.Array,
    dss: jax.Array,
    z_vals: jax.Array,
    densities: jax.Array,
    rgbs: jax.Array,
):
    """
    Inputs:
        rays_bg `float` `[n_total_rays, 3]`: normalized background color of each ray in question
        rays_rgb `float` `[n_total_rays, 3]`: target array to write rendered colors to
        rays_T `float` `[n_total_rays]`: accumulated transmittance of each ray
        rays_depth `float` `[n_total_rays]`: accumulated depth of each ray

        n_samples `uint32` `[n_rays]`: output of ray marching, specifies how many samples are
                                        generated for this ray at this iteration
        indices `uint32` `[n_rays]`: values are in range [0, n_total_rays), specifies the location
                                     in `rays_bg`, `rays_rgb`, `rays_T`, and `rays_depth`
                                     corresponding to this ray
        dss `float` `[n_rays, march_steps_cap]`: each sample's `ds`
        z_vals `float` `[n_rays, march_steps_cap]`: each sample's distance to its ray origin
        densities `float` `[n_rays, march_steps_cap]`: predicted density values from a NeRF model
        rgbs `float` `[n_rays, march_steps_cap, 3]`: predicted RGB values from a NeRF model

    Returns:
        terminate_cnt `uint32`: number of rays that terminated this iteration
        terminated `bool` `[n_rays]`: a binary mask, the i-th location being True means the i-th ray
                                       has terminated
        rays_rgb `float` `[n_total_rays, 3]`: the input `rays_rgb` with ray colors updated
        rays_T `float` `[n_total_rays]`: the input `rays_rgb` with transmittance values updated
        rays_depth `float``[n_total_rays]`: the input `rays_rgb` with depth values updated
    """
    terminate_cnt, terminated, rays_rgb_out, rays_T_out, rays_depth_out = impl.integrate_rays_inference_p.bind(
        rays_bg,
        rays_rgb,
        rays_T,
        rays_depth,

        n_samples,
        indices,
        dss,
        z_vals,
        densities,
        rgbs,
    )
    rays_rgb = rays_rgb.at[indices].set(rays_rgb_out)
    rays_T = rays_T.at[indices].set(rays_T_out)
    rays_depth = rays_depth.at[indices].set(rays_depth_out)
    return terminate_cnt[0], terminated, rays_rgb, rays_T, rays_depth
