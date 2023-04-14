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


def march_rays_lowering_rule(
    ctx: mlir.LoweringRule,

    # arrays
    rays_o: ir.Value,
    rays_d: ir.Value,
    t_starts: ir.Value,
    t_ends: ir.Value,
    noises: ir.Value,
    occupancy_bitfield: ir.Value,

    # static args
    max_n_samples_per_ray: int,  # int
    total_samples: int,  # int
    max_steps: int,  # int
    K: int,  # int
    G: int,  # int
    bound: float,  # float
    stepsize_portion: float,  # float
):
    n_rays, _ = ir.RankedTensorType(rays_o.type).shape

    opaque = volrendutils_cuda.make_marching_descriptor(
        n_rays,
        max_n_samples_per_ray,
        total_samples,
        max_steps,
        K,
        G,
        bound,
        stepsize_portion,
    )

    shapes = {
        "in.rays_o": (n_rays, 3),
        "in.rays_d": (n_rays, 3),
        "in.t_starts": (n_rays,),
        "in.t_ends": (n_rays,),
        "in.noises": (n_rays,),
        "in.occupancy_bitfield": (K*G*G*G//8,),

        "helper.counter": (1,),

        "out.rays_n_samples": (n_rays,),
        "out.rays_sample_startidx": (n_rays,),
        "out.xyzs": (total_samples, 3),
        "out.dirs": (total_samples, 3),
        "out.dss": (total_samples,),
        "out.z_vals": (total_samples,),
    }

    return custom_call(
        call_target_name="march_rays",
        out_types=[
            ir.RankedTensorType.get(shapes["helper.counter"], ir.IntegerType.get_unsigned(32)),
            ir.RankedTensorType.get(shapes["out.rays_n_samples"], ir.IntegerType.get_unsigned(32)),
            ir.RankedTensorType.get(shapes["out.rays_sample_startidx"], ir.IntegerType.get_unsigned(32)),
            ir.RankedTensorType.get(shapes["out.xyzs"], ir.F32Type.get()),
            ir.RankedTensorType.get(shapes["out.dirs"], ir.F32Type.get()),
            ir.RankedTensorType.get(shapes["out.dss"], ir.F32Type.get()),
            ir.RankedTensorType.get(shapes["out.z_vals"], ir.F32Type.get()),
        ],
        operands=[
            rays_o,
            rays_d,
            t_starts,
            t_ends,
            noises,
            occupancy_bitfield,
        ],
        backend_config=opaque,
        operand_layouts=default_layouts(
            shapes["in.rays_o"],
            shapes["in.rays_d"],
            shapes["in.t_starts"],
            shapes["in.t_ends"],
            shapes["in.noises"],
            shapes["in.occupancy_bitfield"],
        ),
        result_layouts=default_layouts(
            shapes["helper.counter"],
            shapes["out.rays_n_samples"],
            shapes["out.rays_sample_startidx"],
            shapes["out.xyzs"],
            shapes["out.dirs"],
            shapes["out.dss"],
            shapes["out.z_vals"],
        ),
    )
