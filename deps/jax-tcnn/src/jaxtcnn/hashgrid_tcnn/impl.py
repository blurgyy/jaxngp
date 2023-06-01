import dataclasses
import functools
from typing import Tuple

import chex
import jax
from jax.interpreters import mlir, xla
from jax.lib import xla_client

from . import abstract, lowering
from .. import tcnnutils


# register GPU XLA custom calls
for name, value in tcnnutils.get_hashgrid_registrations().items():
    xla_client.register_custom_call_target(name, value, platform="gpu")


# primitives
hashgrid_encode_p = jax.core.Primitive("hashgrid🏁")
hashgrid_encode_p.multiple_results = True
hashgrid_encode_p.def_impl(functools.partial(xla.apply_primitive, hashgrid_encode_p))
hashgrid_encode_p.def_abstract_eval(abstract.hashgrid_encode_abstract)

hashgrid_encode_backward_p = jax.core.Primitive("hashgrid🏁backward")
hashgrid_encode_backward_p.multiple_results = True
hashgrid_encode_backward_p.def_impl(functools.partial(xla.apply_primitive, hashgrid_encode_backward_p))
hashgrid_encode_backward_p.def_abstract_eval(abstract.hashgrid_encode_backward_abstract)

hashgrid_encode_inference_p = jax.core.Primitive("hashgrid🏁inference")
hashgrid_encode_inference_p.multiple_results = True
hashgrid_encode_inference_p.def_impl(functools.partial(xla.apply_primitive, hashgrid_encode_inference_p))
hashgrid_encode_inference_p.def_abstract_eval(abstract.hashgrid_encode_inference_abstract)


# lowering rules
mlir.register_lowering(
    prim=hashgrid_encode_p,
    rule=lowering.hashgrid_encode_lowering_rule,
    platform="gpu",
)
mlir.register_lowering(
    prim=hashgrid_encode_backward_p,
    rule=lowering.hashgrid_encode_backward_lowering_rule,
    platform="gpu",
)
mlir.register_lowering(
    prim=hashgrid_encode_inference_p,
    rule=lowering.hashgrid_encode_inference_lowering_rule,
    platform="gpu",
)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, kw_only=True)
class HashGridMetadata:
    # number of levels, "n_levels" in tcnn
    L: int

    # number of features that each level should output, "n_features_per_level" in tcnn
    F: int

    # coarsest resolution, "base_resolution" in tcnn
    N_min: int

    # scale factor between consecutive levels
    per_level_scale: float

    def tree_flatten(self):
        children = ()
        aux = (self.L, self.F, self.N_min, self.per_level_scale)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        L, F, N_min, per_level_scale = aux
        return cls(
            L=L,
            F=F,
            N_min=N_min,
            per_level_scale=per_level_scale,
        )


@functools.partial(jax.custom_vjp, nondiff_argnums=[0])
def __hashgrid_encode(
    desc: HashGridMetadata,
    offset_table_data: jax.Array,
    coords_rm: jax.Array,
    params: jax.Array,
):
    encoded_coords_rm, dy_dcoords_rm = hashgrid_encode_p.bind(
        offset_table_data,
        coords_rm,
        params,
        L=desc.L,
        F=desc.F,
        N_min=desc.N_min,
        per_level_scale=desc.per_level_scale,
    )
    return encoded_coords_rm, dy_dcoords_rm


def __hashgrid_encode_fwd(
    desc: HashGridMetadata,
    offset_table_data: jax.Array,
    coords_rm: jax.Array,
    params: jax.Array,
):
    primal_outputs = hashgrid_encode_p.bind(
        offset_table_data,
        coords_rm,
        params,
        L=desc.L,
        F=desc.F,
        N_min=desc.N_min,
        per_level_scale=desc.per_level_scale,
    )
    encoded_coords_rm, dy_dcoords_rm = primal_outputs
    aux = {
        "in.offset_table_data": offset_table_data,
        "in.coords_rm": coords_rm,
        "in.params": params,
        "out.dy_dcoords_rm": dy_dcoords_rm,
    }
    return primal_outputs, aux


def __hashgrid_encode_bwd(desc: HashGridMetadata, aux, grads):
    dL_dy_rm, _ = grads
    dL_dparams, dL_dcoords_rm = hashgrid_encode_backward_p.bind(
        aux["in.offset_table_data"],
        aux["in.coords_rm"],
        aux["in.params"],
        dL_dy_rm,
        aux["out.dy_dcoords_rm"],

        L=desc.L,
        F=desc.F,
        N_min=desc.N_min,
        per_level_scale=desc.per_level_scale,
    )
    return None, dL_dcoords_rm, dL_dparams


__hashgrid_encode.defvjp(
    fwd=__hashgrid_encode_fwd,
    bwd=__hashgrid_encode_bwd,
)
