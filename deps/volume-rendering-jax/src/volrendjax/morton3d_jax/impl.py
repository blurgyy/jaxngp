import functools

import jax
from jax.interpreters import mlir, xla
from jax.lib import xla_client

from . import abstract, lowering
from .. import volrendutils_cuda


# register GPU XLA custom calls
for name, value in volrendutils_cuda.get_morton3d_registrations().items():
    xla_client.register_custom_call_target(name, value, platform="gpu")

morton3d_p = jax.core.Primitive("morton3d⚡")
morton3d_p.multiple_results = False
morton3d_p.def_impl(functools.partial(xla.apply_primitive, morton3d_p))
morton3d_p.def_abstract_eval(abstract.morton3d_abstract)

morton3d_invert_p = jax.core.Primitive("morton3d⚡invert")
morton3d_invert_p.multiple_results = False
morton3d_invert_p.def_impl(functools.partial(xla.apply_primitive, morton3d_invert_p))
morton3d_invert_p.def_abstract_eval(abstract.morton3d_invert_abstract)

# register mlir lowering rules
mlir.register_lowering(
    prim=morton3d_p,
    rule=lowering.morton3d_lowering_rule,
    platform="gpu",
)
mlir.register_lowering(
    prim=morton3d_invert_p,
    rule=lowering.morton3d_invert_lowering_rule,
    platform="gpu",
)
