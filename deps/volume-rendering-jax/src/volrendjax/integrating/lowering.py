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
    densities: ir.Value,
    rgbs: ir.Value,
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
        "in.densities": (total_samples, 1),
        "in.rgbs": (total_samples, 3),

        "helper.counter": (1,),

        "out.opacities": (n_rays,),
        "out.final_rgbs": (n_rays, 3),
        "out.depths": (n_rays,),
    }

    return custom_call(
        call_target_name="integrate_rays",
        out_types=[
            ir.RankedTensorType.get(shapes["helper.counter"], ir.IntegerType.get_unsigned(32)),
            ir.RankedTensorType.get(shapes["out.opacities"], ir.F32Type.get()),
            ir.RankedTensorType.get(shapes["out.final_rgbs"], ir.F32Type.get()),
            ir.RankedTensorType.get(shapes["out.depths"], ir.F32Type.get()),
        ],
        operands=[
            rays_sample_startidx,
            rays_n_samples,
            bgs,
            dss,
            z_vals,
            densities,
            rgbs,
        ],
        backend_config=opaque,
        operand_layouts=default_layouts(
            shapes["in.rays_sample_startidx"],
            shapes["in.rays_n_samples"],
            shapes["in.bgs"],
            shapes["in.dss"],
            shapes["in.z_vals"],
            shapes["in.densities"],
            shapes["in.rgbs"],
        ),
        result_layouts=default_layouts(
            shapes["helper.counter"],
            shapes["out.opacities"],
            shapes["out.final_rgbs"],
            shapes["out.depths"],
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
    densities: ir.Value,
    rgbs: ir.Value,

    # original outputs
    opacities: ir.Value,
    final_rgbs: ir.Value,
    depths: ir.Value,

    # gradient inputs
    dL_dopacities: ir.Value,
    dL_dfinal_rgbs: ir.Value,
    dL_ddepths: ir.Value,
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
        "in.densities": (total_samples, 1),
        "in.rgbs": (total_samples, 3),

        "in.opacities": (n_rays,),
        "in.final_rgbs": (n_rays, 3),
        "in.depths": (n_rays,),

        "in.dL_dopacities": (n_rays,),
        "in.dL_dfinal_rgbs": (n_rays, 3),
        "in.dL_ddepths": (n_rays,),

        "out.dL_dbgs": (n_rays, 3),
        "out.dL_dz_vals": (total_samples,),
        "out.dL_ddensities": (total_samples, 1),
        "out.dL_drgbs": (total_samples, 3),
    }

    return custom_call(
        call_target_name="integrate_rays_backward",
        out_types=[
            ir.RankedTensorType.get(shapes["out.dL_dbgs"], ir.F32Type.get()),
            ir.RankedTensorType.get(shapes["out.dL_dz_vals"], ir.F32Type.get()),
            ir.RankedTensorType.get(shapes["out.dL_ddensities"], ir.F32Type.get()),
            ir.RankedTensorType.get(shapes["out.dL_drgbs"], ir.F32Type.get()),
        ],
        operands=[
            rays_sample_startidx,
            rays_n_samples,

            bgs,
            dss,
            z_vals,
            densities,
            rgbs,

            opacities,
            final_rgbs,
            depths,

            dL_dopacities,
            dL_dfinal_rgbs,
            dL_ddepths
        ],
        backend_config=opaque,
        operand_layouts=default_layouts(
            shapes["in.rays_sample_startidx"],
            shapes["in.rays_n_samples"],
            shapes["in.bgs"],
            shapes["in.dss"],
            shapes["in.z_vals"],
            shapes["in.densities"],
            shapes["in.rgbs"],

            shapes["in.opacities"],
            shapes["in.final_rgbs"],
            shapes["in.depths"],

            shapes["in.dL_dopacities"],
            shapes["in.dL_dfinal_rgbs"],
            shapes["in.dL_ddepths"],
        ),
        result_layouts=default_layouts(
            shapes["out.dL_dbgs"],
            shapes["out.dL_dz_vals"],
            shapes["out.dL_ddensities"],
            shapes["out.dL_drgbs"],
        ),
    )


def integrate_rays_inference_lowering_rule(
    ctx: mlir.LoweringRuleContext,

    rays_bg: ir.Value,
    rays_rgb: ir.Value,
    rays_T: ir.Value,
    rays_depth: ir.Value,

    n_samples: ir.Value,
    indices: ir.Value,
    dss: ir.Value,
    z_vals: ir.Value,
    densities: ir.Value,
    rgbs: ir.Value,
):
    (n_total_rays, _) = ir.RankedTensorType(rays_rgb.type).shape
    (n_rays, march_steps_cap) = ir.RankedTensorType(dss.type).shape

    opaque = volrendutils_cuda.make_integrating_inference_descriptor(n_total_rays, n_rays, march_steps_cap)

    shapes = {
        "in.rays_bg": (n_total_rays, 3),
        "in.rays_rgb": (n_total_rays, 3),
        "in.rays_T": (n_total_rays,),
        "in.rays_depth": (n_total_rays,),

        "in.n_samples": (n_rays,),
        "in.indices": (n_rays,),
        "in.dss": (n_rays, march_steps_cap),
        "in.z_vals": (n_rays, march_steps_cap),
        "in.densities": (n_rays, march_steps_cap, 1),
        "in.rgbs": (n_rays, march_steps_cap, 3),

        "out.terminate_cnt": (1,),
        "out.terminated": (n_rays,),
        "out.rays_rgb": (n_rays, 3),
        "out.rays_T": (n_rays,),
        "out.rays_depth": (n_rays,),
    }

    return custom_call(
        call_target_name="integrate_rays_inference",
        out_types=[
            ir.RankedTensorType.get(shapes["out.terminate_cnt"], ir.IntegerType.get_unsigned(32)),
            ir.RankedTensorType.get(shapes["out.terminated"], ir.IntegerType.get_signless(1)),
            ir.RankedTensorType.get(shapes["out.rays_rgb"], ir.F32Type.get()),
            ir.RankedTensorType.get(shapes["out.rays_T"], ir.F32Type.get()),
            ir.RankedTensorType.get(shapes["out.rays_depth"], ir.F32Type.get()),
        ],
        operands=[
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
        ],
        backend_config=opaque,
        operand_layouts=default_layouts(
            shapes["in.rays_bg"],
            shapes["in.rays_rgb"],
            shapes["in.rays_T"],
            shapes["in.rays_depth"],

            shapes["in.n_samples"],
            shapes["in.indices"],
            shapes["in.dss"],
            shapes["in.z_vals"],
            shapes["in.densities"],
            shapes["in.rgbs"],
        ),
        result_layouts=default_layouts(
            shapes["out.terminate_cnt"],
            shapes["out.terminated"],
            shapes["out.rays_rgb"],
            shapes["out.rays_T"],
            shapes["out.rays_depth"],
        ),
    )
