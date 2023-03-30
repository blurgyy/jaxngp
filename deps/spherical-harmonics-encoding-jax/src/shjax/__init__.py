import functools

import chex
import jax
from jax.abstract_arrays import ShapedArray
from jax.interpreters import batching, mlir, xla
from jax.interpreters.mlir import ir
from jax.lib import xla_client
import jax.numpy as jnp

from . import cudaops

try:
    from jaxlib.mhlo_helpers import custom_call
except ModuleNotFoundError:
    # A more recent jaxlib would have `hlo_helpers` instead of `mhlo_helpers`
    # <https://github.com/google/jax/commit/b8ae8e3fa10f9abe998459fac1513915acee776d#diff-50658d597212b4ce070b8bd8c1fc522deeee1845ba387a0a5b507c446e8ea12a>
    from jaxlib.hlo_helpers import custom_call


# register GPU XLA custom calls
for name, value in cudaops.get_registrations().items():
    xla_client.register_custom_call_target(name, value, platform="gpu")

# jit rules, infer returned shape according to input
def _spherical_harmonics_encoding_abstract(coord: jax.Array, hint: jax.Array):
    """
    Inputs:
        coord [..., 3] float: input coordinates
        hint [degree]: an array with shape [L], this is used to hint the function with the desired
                  spherical harmonics degrees
    """
    (n, _), dtype = coord.shape, coord.dtype
    degree, = hint.shape
    dtype = jax.dtypes.canonicalize_dtype(coord.dtype)
    return ShapedArray(shape=(n, degree * degree), dtype=dtype)

# register the primitive
sh_enc_p = jax.core.Primitive("spherical_harmonics_encoding🌐")
sh_enc_p.multiple_results = False
sh_enc_p.def_impl(functools.partial(xla.apply_primitive, sh_enc_p))
sh_enc_p.def_abstract_eval(_spherical_harmonics_encoding_abstract)

# helper function for mapping given shapes to their default mlir layouts
def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]

# mlir lowering rule
def _spherical_harmonics_encoding_lowering_cuda(
        ctx: mlir.LoweringRuleContext,
        coord: ir.Value,
        hint: ir.Value,
    ):
    coord_type = ir.RankedTensorType(coord.type)
    coord_shape = coord_type.shape

    n, _ = coord_shape
    degree, = ir.RankedTensorType(hint.type).shape

    result_shape = (n, degree * degree)

    opaque = cudaops.make_spherical_harmonics_encoding_descriptor(n, degree)

    # Documentation says directly return the `custom_call` would suffice, but directly returning
    # here results in error "Output of translation rule must be iterable: ...", so let's make it
    # iterable.
    # NOTE:
    #   A newer jaxlib (current 0.3.22) may require this to be a single custom_call(...), instead of
    #   an iterable, as documentation suggests.
    # REF:
    #   documentation: <https://jax.readthedocs.io/en/latest/Custom_Operation_for_GPUs.html#lowering-to-mlir-custom-call>
    #   tutorial: <https://github.com/dfm/extending-jax/blob/1cb4c39c524bccb5e3068c5a7f57a425ab0426a2/src/kepler_jax/kepler_jax.py#L113>
    return [custom_call(
        "spherical_harmonics_encoding_cuda_f32",  # the name of the registered XLA custom call at the top of this script
        out_types=[
            ir.RankedTensorType.get(result_shape, coord_type.element_type),
        ],
        operands=[coord],
        backend_config=opaque,
        operand_layouts=default_layouts(coord_shape),
        result_layouts=default_layouts(result_shape),
    )]

mlir.register_lowering(
    prim=sh_enc_p,
    rule=_spherical_harmonics_encoding_lowering_cuda,
    platform="gpu",
)

# vmap support. REF: <https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html#batching>
def spherical_harmonics_encoding_batch(args, axes):
    """
    Inputs:
        args: Passed to def_impl, contains two tensors: `coord` and `hint`, where only `coord` is
              batched.  args is (coord, hint)
        axes: The axes that are being batched, one value for each arg, value is an integer if the
              arg is batched, value is None if the arg is not batched.  In this case,
              coord.shape[axes[0]] = B, and axes[1] = None.
    """
    coord, hint = args
    assert coord.shape[-1] == 3, "spatial coordinates must be the last dimension"

    # reshape the value so that the custom operation can consume it
    enc = sh_enc_p.bind(coord.reshape(-1, 3), hint)
    # or:
    # ret = spherical_harmonics_encoding(
    #     coord=coord.reshape(B*n, 3),
    #     degree=hint.shape[0],
    # )

    # reshape back so that the spatial dimension (last dimension) is replaced with the encodings
    enc = enc.reshape(*coord.shape[:-1], -1)

    # return the result, and the result axis that was batched
    return enc, axes[0]

batching.primitive_batchers[sh_enc_p] = spherical_harmonics_encoding_batch

# the only exposed function
def spherical_harmonics_encoding(coord: jax.Array, degree: int) -> jax.Array:
    """
    Spherical harmonics encoding with GPU acceleration, expects unit vectors as input.

    Inputs:
        coord [..., 3] float: input 3D coordinates
        degree int: highest degree used in spherical harmonics

    Returns:
        outputs [..., degree**2] float: encoded coordinates
    """
    chex.assert_rank(coord, 2)
    chex.assert_axis_dimension(coord, -1, 3)
    return sh_enc_p.bind(coord, jnp.empty((degree,)))
