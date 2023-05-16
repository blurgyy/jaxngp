import functools
from typing import Tuple

import jax
from jax.interpreters import mlir, xla
from jax.lib import xla_client

from . import abstract, lowering
from .. import volrendutils_cuda


# register GPU XLA custom calls
for name, value in volrendutils_cuda.get_integrating_registrations().items():
    xla_client.register_custom_call_target(name, value, platform="gpu")


# primitives
integrate_rays_p = jax.core.Primitive("integrate_rays🎨")
integrate_rays_p.multiple_results = True
integrate_rays_p.def_impl(functools.partial(xla.apply_primitive, integrate_rays_p))
integrate_rays_p.def_abstract_eval(abstract.integrate_rays_abstract)

integrate_rays_bwd_p = jax.core.Primitive("integrate_rays🎨backward")
integrate_rays_bwd_p.multiple_results = True
integrate_rays_bwd_p.def_impl(functools.partial(xla.apply_primitive, integrate_rays_bwd_p))
integrate_rays_bwd_p.def_abstract_eval(abstract.integrate_rays_backward_abstract)

integrate_rays_inference_p = jax.core.Primitive("integrate_rays🎨inference")
integrate_rays_inference_p.multiple_results = True
integrate_rays_inference_p.def_impl(functools.partial(xla.apply_primitive, integrate_rays_inference_p))
integrate_rays_inference_p.def_abstract_eval(abstract.integrate_rays_inference_abstract)

# register mlir lowering rules
mlir.register_lowering(
    prim=integrate_rays_p,
    rule=lowering.integrate_rays_lowering_rule,
    platform="gpu",
)
mlir.register_lowering(
    prim=integrate_rays_bwd_p,
    rule=lowering.integrate_rays_backward_lowring_rule,
    platform="gpu",
)
mlir.register_lowering(
    prim=integrate_rays_inference_p,
    rule=lowering.integrate_rays_inference_lowering_rule,
    platform="gpu",
)

@jax.custom_vjp
def __integrate_rays(
    rays_sample_startidx: jax.Array,
    rays_n_samples: jax.Array,
    bgs: jax.Array,
    dss: jax.Array,
    z_vals: jax.Array,
    densities: jax.Array,
    rgbs: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    bgs = jax.numpy.broadcast_to(bgs, (rays_sample_startidx.shape[0], 3))

    counter, final_rgbs, depths = integrate_rays_p.bind(
        rays_sample_startidx,
        rays_n_samples,
        bgs,
        dss,
        z_vals,
        densities,
        rgbs,
    )

    return counter, final_rgbs, depths

def __fwd_integrate_rays(
    rays_sample_startidx: jax.Array,
    rays_n_samples: jax.Array,
    bgs: jax.Array,
    dss: jax.Array,
    z_vals: jax.Array,
    densities: jax.Array,
    rgbs: jax.Array,
):
    bgs = jax.numpy.broadcast_to(bgs, (rays_sample_startidx.shape[0], 3))

    primal_outputs = __integrate_rays(
        rays_sample_startidx,
        rays_n_samples,
        bgs,
        dss,
        z_vals,
        densities,
        rgbs,
    )
    counter, final_rgbs, depths = primal_outputs
    aux = {
        "in.rays_sample_startidx": rays_sample_startidx,
        "in.rays_n_samples": rays_n_samples,
        "in.bgs": bgs,
        "in.dss": dss,
        "in.z_vals": z_vals,
        "in.densities": densities,
        "in.rgbs": rgbs,

        "out.counter": counter,
        "out.final_rgbs": final_rgbs,
        "out.depths": depths,
    }
    return primal_outputs, aux

def __bwd_integrate_rays(aux, grads):
    _, dL_dfinal_rgbs, dL_ddepths = grads
    dL_dbgs, dL_dz_vals, dL_ddensities, dL_drgbs = integrate_rays_bwd_p.bind(
        aux["in.rays_sample_startidx"],
        aux["in.rays_n_samples"],
        aux["in.bgs"],
        aux["in.dss"],
        aux["in.z_vals"],
        aux["in.densities"],
        aux["in.rgbs"],

        aux["out.final_rgbs"],
        aux["out.depths"],

        dL_dfinal_rgbs,
        dL_ddepths,
    )
    return (
        # First 2 primal inputs are integer-valued arrays (`rays_sample_startidx`,
        # `rays_n_samples`), return no gradient for them.
        # REF:
        #   <https://jax.readthedocs.io/en/latest/jep/4008-custom-vjp-update.html#what-to-update>:
        #   Wherever we used to use nondiff_argnums for array values, we should just pass those as
        #   regular arguments.  In the bwd rule, we need to produce values for them, but we can just
        #   produce `None` values to indicate there’s no corresponding gradient value.
        None, None,
        # 4-th primal input is `dss`, no gradient
        None,
        # gradients for background colors, z_vals and model predictions (densites and rgbs)
        dL_dbgs, dL_dz_vals, dL_ddensities, dL_drgbs
    )

__integrate_rays.defvjp(
    fwd=__fwd_integrate_rays,
    bwd=__bwd_integrate_rays,
)
