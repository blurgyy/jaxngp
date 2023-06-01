import jax

from .impl import HashGridMetadata, __hashgrid_encode, hashgrid_encode_inference_p


def hashgrid_encode(
    desc: HashGridMetadata,

    offset_table_data: jax.Array,
    coords_rm: jax.Array,
    params: jax.Array,
):
    encoded_coords_rm, dy_dcoords_rm = __hashgrid_encode(
        desc,
        offset_table_data,
        coords_rm,
        params,
    )
    return encoded_coords_rm

def hashgrid_encode_inference(
    desc: HashGridMetadata,

    offset_table_data: jax.Array,
    coords_rm: jax.Array,
    params: jax.Array,
):
    encoded_coords_rm, _ = hashgrid_encode_inference_p.bind(
        offset_table_data,
        coords_rm,
        params,

        L=desc.L,
        F=desc.F,
        N_min=desc.N_min,
        per_level_scale=desc.per_level_scale,
    )
    return encoded_coords_rm
