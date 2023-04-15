from typing import Tuple

import jax

from . import impl


# this function is a wrapper on top of `__integrate_rays` which has custom vjp (wrapping the
# `__integrate_rays` function because the @jax.custom_vjp decorator makes the decorated function's
# docstring invisible to LSPs).
def integrate_rays(
    transmittance_threshold: jax.Array,
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
        transmittance_threshold `[n_rays]`: the i-th ray will stop compositing color once its
                                            accumulated transmittance is below this threshold
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
        opacities `[n_rays]`: accumulated opacities along each ray
        final_rgbs `[n_rays, 3]~: integrated ray colors according to input densities and rgbs.
        depths `[n_rays]`: estimated termination depth of each ray
    """
    counter, reached_bg, opacities, final_rgbs, depths = impl.__integrate_rays(
        transmittance_threshold,
        rays_sample_startidx,
        rays_n_samples,
        bgs,
        dss,
        z_vals,
        densities,
        rgbs,
    )

    return counter[0], opacities, final_rgbs, depths
