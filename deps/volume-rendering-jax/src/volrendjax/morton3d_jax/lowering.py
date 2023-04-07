from jax.interpreters import mlir
from jax.interpreters.mlir import ir

from .. import volrendutils_cuda

try:
    from jaxlib.mhlo_helpers import custom_call
except ModuleNotFoundError:
    # A more recent jaxlib would have `hlo_helpers` instead of `mhlo_helpers`
    # <https://github.com/google/jax/commit/b8ae8e3fa10f9abe998459fac1513915acee776d#diff-50658d597212b4ce070b8bd8c1fc522deeee1845ba387a0a5b507c446e8ea12a>
    from jaxlib.hlo_helpers import custom_call


# helper function for mapping given shapes to their default mlir layouts
def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]


def morton3d_lowering_rule(
    ctx: mlir.LoweringRule,

    # input array
    xyzs: ir.Value,
):
    length, _ = ir.RankedTensorType(xyzs.type).shape

    opaque = volrendutils_cuda.make_morton3d_descriptor(length)

    shapes = {
        "in.xyzs": (length, 3),

        "out.idcs": (length,),
    }

    return [custom_call(
        call_target_name="morton3d",
        out_types=[
            ir.RankedTensorType.get(shapes["out.idcs"], ir.IntegerType.get_unsigned(32)),
        ],
        operands=[
            xyzs,
        ],
        backend_config=opaque,
        operand_layouts=default_layouts(
            shapes["in.xyzs"],
        ),
        result_layouts=default_layouts(
            shapes["out.idcs"],
        ),
    )]


def morton3d_invert_lowering_rule(
    ctx: mlir.LoweringRule,

    # input array
    idcs: ir.Value,
):
    length, = ir.RankedTensorType(idcs.type).shape

    opaque = volrendutils_cuda.make_morton3d_descriptor(length)

    shapes = {
        "in.idcs": (length,),

        "out.xyzs": (length, 3),
    }

    return [custom_call(
        call_target_name="morton3d_invert",
        out_types=[
            ir.RankedTensorType.get(shapes["out.xyzs"], ir.IntegerType.get_unsigned(32)),
        ],
        operands=[
            idcs,
        ],
        backend_config=opaque,
        operand_layouts=default_layouts(
            shapes["in.idcs"],
        ),
        result_layouts=default_layouts(
            shapes["out.xyzs"],
        ),
    )]
