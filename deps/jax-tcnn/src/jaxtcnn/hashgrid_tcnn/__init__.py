import jax

from .impl import HashGridMetadata, __hashgrid_encode


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
