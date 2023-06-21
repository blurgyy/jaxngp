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


def integrate_rays_lowering_rule(
    ctx: mlir.LoweringRuleContext,

    rays_sample_startidx: ir.Value,
    rays_n_samples: ir.Value,

    bgs: ir.Value,
    dss: ir.Value,
    z_vals: ir.Value,
    drgbs: ir.Value,
):
    n_rays, = ir.RankedTensorType(rays_sample_startidx.type).shape
    total_samples, = ir.RankedTensorType(z_vals.type).shape

    opaque = volrendutils_cuda.make_integrating_descriptor(n_rays, total_samples)

    shapes = {
        "in.rays_sample_startidx": (n_rays,),
        "in.rays_n_samples": (n_rays,),

        "in.bgs": (n_rays, 3),
        "in.dss": (total_samples,),
        "in.z_vals": (total_samples,),
        "in.drgbs": (total_samples, 4),

        "helper.counter": (1,),

        "out.final_rgbds": (n_rays, 4),
        "out.final_opacities": (n_rays,),
    }

    return custom_call(
        call_target_name="integrate_rays",
        out_types=[
            ir.RankedTensorType.get(shapes["helper.counter"], ir.IntegerType.get_unsigned(32)),
            ir.RankedTensorType.get(shapes["out.final_rgbds"], ir.F32Type.get()),
            ir.RankedTensorType.get(shapes["out.final_opacities"], ir.F32Type.get()),
        ],
        operands=[
            rays_sample_startidx,
            rays_n_samples,
            bgs,
            dss,
            z_vals,
            drgbs,
        ],
        backend_config=opaque,
        operand_layouts=default_layouts(
            shapes["in.rays_sample_startidx"],
            shapes["in.rays_n_samples"],
            shapes["in.bgs"],
            shapes["in.dss"],
            shapes["in.z_vals"],
            shapes["in.drgbs"],
        ),
        result_layouts=default_layouts(
            shapes["helper.counter"],
            shapes["out.final_rgbds"],
            shapes["out.final_opacities"],
        ),
    )


def integrate_rays_backward_lowring_rule(
    ctx: mlir.LoweringRuleContext,

    rays_sample_startidx: ir.Value,
    rays_n_samples: ir.Value,

    # original inputs
    bgs: ir.Value,
    dss: ir.Value,
    z_vals: ir.Value,
    drgbs: ir.Value,

    # original outputs
    final_rgbds: ir.Value,
    final_opacities: ir.Value,

    # gradient inputs
    dL_dfinal_rgbds: ir.Value,

    # static argument
    near_distance: float,
):
    n_rays, = ir.RankedTensorType(rays_sample_startidx.type).shape
    total_samples, = ir.RankedTensorType(z_vals.type).shape

    opaque = volrendutils_cuda.make_integrating_backward_descriptor(n_rays, total_samples, near_distance)

    shapes = {
        "in.rays_sample_startidx": (n_rays,),
        "in.rays_n_samples": (n_rays,),

        "in.bgs": (n_rays, 3),
        "in.dss": (total_samples,),
        "in.z_vals": (total_samples,),
        "in.drgbs": (total_samples, 4),

        "in.final_rgbds": (n_rays, 4),
        "in.final_opacities": (n_rays,),

        "in.dL_dfinal_rgbds": (n_rays, 4),

        "out.dL_dbgs": (n_rays, 3),
        "out.dL_dz_vals": (total_samples,),
        "out.dL_ddrgbs": (total_samples, 4),
    }

    return custom_call(
        call_target_name="integrate_rays_backward",
        out_types=[
            ir.RankedTensorType.get(shapes["out.dL_dbgs"], ir.F32Type.get()),
            ir.RankedTensorType.get(shapes["out.dL_dz_vals"], ir.F32Type.get()),
            ir.RankedTensorType.get(shapes["out.dL_ddrgbs"], ir.F32Type.get()),
        ],
        operands=[
            rays_sample_startidx,
            rays_n_samples,

            bgs,
            dss,
            z_vals,
            drgbs,

            final_rgbds,
            final_opacities,

            dL_dfinal_rgbds,
        ],
        backend_config=opaque,
        operand_layouts=default_layouts(
            shapes["in.rays_sample_startidx"],
            shapes["in.rays_n_samples"],
            shapes["in.bgs"],
            shapes["in.dss"],
            shapes["in.z_vals"],
            shapes["in.drgbs"],

            shapes["in.final_rgbds"],
            shapes["in.final_opacities"],

            shapes["in.dL_dfinal_rgbds"],
        ),
        result_layouts=default_layouts(
            shapes["out.dL_dbgs"],
            shapes["out.dL_dz_vals"],
            shapes["out.dL_ddrgbs"],
        ),
    )


def integrate_rays_inference_lowering_rule(
    ctx: mlir.LoweringRuleContext,

    rays_bg: ir.Value,
    rays_rgbd: ir.Value,
    rays_T: ir.Value,

    n_samples: ir.Value,
    indices: ir.Value,
    dss: ir.Value,
    z_vals: ir.Value,
    drgbs: ir.Value,
):
    (n_total_rays, _) = ir.RankedTensorType(rays_rgbd.type).shape
    (n_rays, march_steps_cap) = ir.RankedTensorType(dss.type).shape

    opaque = volrendutils_cuda.make_integrating_inference_descriptor(n_total_rays, n_rays, march_steps_cap)

    shapes = {
        "in.rays_bg": (n_total_rays, 3),
        "in.rays_rgbd": (n_total_rays, 4),
        "in.rays_T": (n_total_rays,),

        "in.n_samples": (n_rays,),
        "in.indices": (n_rays,),
        "in.dss": (n_rays, march_steps_cap),
        "in.z_vals": (n_rays, march_steps_cap),
        "in.drgbs": (n_rays, march_steps_cap, 4),

        "out.terminate_cnt": (1,),
        "out.terminated": (n_rays,),
        "out.rays_rgbd": (n_rays, 4),
        "out.rays_T": (n_rays,),
    }

    return custom_call(
        call_target_name="integrate_rays_inference",
        out_types=[
            ir.RankedTensorType.get(shapes["out.terminate_cnt"], ir.IntegerType.get_unsigned(32)),
            ir.RankedTensorType.get(shapes["out.terminated"], ir.IntegerType.get_signless(1)),
            ir.RankedTensorType.get(shapes["out.rays_rgbd"], ir.F32Type.get()),
            ir.RankedTensorType.get(shapes["out.rays_T"], ir.F32Type.get()),
        ],
        operands=[
            rays_bg,
            rays_rgbd,
            rays_T,

            n_samples,
            indices,
            dss,
            z_vals,
            drgbs,
        ],
        backend_config=opaque,
        operand_layouts=default_layouts(
            shapes["in.rays_bg"],
            shapes["in.rays_rgbd"],
            shapes["in.rays_T"],

            shapes["in.n_samples"],
            shapes["in.indices"],
            shapes["in.dss"],
            shapes["in.z_vals"],
            shapes["in.drgbs"],
        ),
        result_layouts=default_layouts(
            shapes["out.terminate_cnt"],
            shapes["out.terminated"],
            shapes["out.rays_rgbd"],
            shapes["out.rays_T"],
        ),
    )
