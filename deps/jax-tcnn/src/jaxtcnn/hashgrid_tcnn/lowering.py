from jax.interpreters import mlir
from jax.interpreters.mlir import ir

from .. import tcnnutils

try:
    from jaxlib.mhlo_helpers import custom_call
except ModuleNotFoundError:
    # A more recent jaxlib would have `hlo_helpers` instead of `mhlo_helpers`
    # <https://github.com/google/jax/commit/b8ae8e3fa10f9abe998459fac1513915acee776d#diff-50658d597212b4ce070b8bd8c1fc522deeee1845ba387a0a5b507c446e8ea12a>
    from jaxlib.hlo_helpers import custom_call


# helper function for mapping given shapes to their default mlir layouts
def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]


def hashgrid_encode_lowering_rule(
    ctx: mlir.LoweringRule,

    # arrays
    offset_table_data: ir.Value,
    coords_rm: ir.Value,
    params: ir.Value,

    # static args
    L: int,
    F: int,
    N_min: int,
    per_level_scale: float,
):
    dim, n_coords = ir.RankedTensorType(coords_rm.type).shape
    n_params, _ = ir.RankedTensorType(params.type).shape

    opaque = tcnnutils.make_hashgrid_descriptor(
        n_coords,
        L,
        F,
        N_min,
        per_level_scale,
    )

    shapes = {
        "in.offset_table_data": (L + 1,),
        "in.coords_rm": (dim, n_coords),
        "in.params": (n_params, F),

        "out.encoded_coords_rm": (L * F, n_coords),
        "out.dy_dcoords_rm": (dim * L * F, n_coords),
    }

    return custom_call(
        call_target_name="hashgrid_encode",
        out_types=[
            ir.RankedTensorType.get(shapes["out.encoded_coords_rm"], ir.F32Type.get()),
            ir.RankedTensorType.get(shapes["out.dy_dcoords_rm"], ir.F32Type.get()),
        ],
        operands=[
            offset_table_data,
            coords_rm,
            params,
        ],
        backend_config=opaque,
        operand_layouts=default_layouts(
            shapes["in.offset_table_data"],
            shapes["in.coords_rm"],
            shapes["in.params"],
        ),
        result_layouts=default_layouts(
            shapes["out.encoded_coords_rm"],
            shapes["out.dy_dcoords_rm"],
        ),
    )


def hashgrid_encode_backward_lowering_rule(
    ctx: mlir.LoweringRule,

    offset_table_data: ir.Value,
    coords_rm: ir.Value,
    params: ir.Value,  # only for determining shape of dL_dparams
    dL_dy_rm: ir.Value,
    dy_dcoords_rm: ir.Value,

    # static args
    L: int,
    F: int,
    N_min: int,
    per_level_scale: float,
):
    dim, n_coords = ir.RankedTensorType(coords_rm.type).shape
    n_params, _ = ir.RankedTensorType(params.type).shape

    opaque = tcnnutils.make_hashgrid_descriptor(
        n_coords,
        L,
        F,
        N_min,
        per_level_scale,
    )

    shapes = {
        "in.offset_table_data": (L + 1,),
        "in.coords_rm": (dim, n_coords),
        # "in.params": (n_params, F),
        "in.dL_dy_rm": (L * F, n_coords),
        "in.dy_dcoords_rm": (dim * L * F, n_coords),

        "out.dL_dparams": (n_params, F),
        "out.dL_dcoords_rm": (dim, n_coords),
    }

    return custom_call(
        call_target_name="hashgrid_encode_backward",
        out_types=[
            ir.RankedTensorType.get(shapes["out.dL_dparams"], ir.F32Type.get()),
            ir.RankedTensorType.get(shapes["out.dL_dcoords_rm"], ir.F32Type.get()),
        ],
        operands=[
            offset_table_data,
            coords_rm,
            dL_dy_rm,
            dy_dcoords_rm
        ],
        backend_config=opaque,
        result_layouts=default_layouts(
            shapes["out.dL_dparams"],
            shapes["out.dL_dcoords_rm"],
        ),
        operand_layouts=default_layouts(
            shapes["in.offset_table_data"],
            shapes["in.coords_rm"],
            shapes["in.dL_dy_rm"],
            shapes["in.dy_dcoords_rm"],
        ),
    )


def hashgrid_encode_inference_lowering_rule(
    ctx: mlir.LoweringRule,

    # arrays
    offset_table_data: ir.Value,
    coords_rm: ir.Value,
    params: ir.Value,

    # static args
    L: int,
    F: int,
    N_min: int,
    per_level_scale: float,
):
    dim, n_coords = ir.RankedTensorType(coords_rm.type).shape
    n_params, _ = ir.RankedTensorType(params.type).shape

    opaque = tcnnutils.make_hashgrid_descriptor(
        n_coords,
        L,
        F,
        N_min,
        per_level_scale,
    )

    shapes = {
        "in.offset_table_data": (L + 1,),
        "in.coords_rm": (dim, n_coords),
        "in.params": (n_params, F),

        "out.encoded_coords_rm": (L * F, n_coords),
        "out.dy_dcoords_rm": (dim * L * F, n_coords),
    }

    return custom_call(
        call_target_name="hashgrid_encode",
        out_types=[
            ir.RankedTensorType.get(shapes["out.encoded_coords_rm"], ir.F16Type.get()),  # fp16
            ir.RankedTensorType.get(shapes["out.dy_dcoords_rm"], ir.F32Type.get()),
        ],
        operands=[
            offset_table_data,
            coords_rm,
            params,
        ],
        backend_config=opaque,
        operand_layouts=default_layouts(
            shapes["in.offset_table_data"],
            shapes["in.coords_rm"],
            shapes["in.params"],
        ),
        result_layouts=default_layouts(
            shapes["out.encoded_coords_rm"],
            shapes["out.dy_dcoords_rm"],
        ),
    )
