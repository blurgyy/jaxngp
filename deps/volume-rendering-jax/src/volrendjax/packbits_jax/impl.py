import functools

import jax
from jax.interpreters import mlir, xla
from jax.lib import xla_client

from . import abstract, lowering
from .. import volrendutils_cuda


# register GPU XLA custom calls
for name, value in volrendutils_cuda.get_packbits_registrations().items():
    xla_client.register_custom_call_target(name, value, platform="gpu")

packbits_p = jax.core.Primitive("packbits🎱")
packbits_p.multiple_results = True
packbits_p.def_impl(functools.partial(xla.apply_primitive, packbits_p))
packbits_p.def_abstract_eval(abstract.pack_density_into_bits_abstract)

mlir.register_lowering(
    prim=packbits_p,
    rule=lowering.packbits_lowering_rule,
    platform="gpu",
)
