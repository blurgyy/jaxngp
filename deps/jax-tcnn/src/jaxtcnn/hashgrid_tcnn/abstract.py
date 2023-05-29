import chex
import jax
import jax.numpy as jnp


def hashgrid_encode_abstract(
    # arrays
    offset_table_data: jax.Array,
    coords_rm: jax.Array,
    params: jax.Array,

    # static args
    L: int,
    F: int,
    N_min: int,
    per_level_scale: float,
):
    dim, n_coords = coords_rm.shape
    if dim != 3:
        raise NotImplementedError(
            "hashgrid encoding is only implemented for 3D coordinates, expected input coordinates to have shape ({}, n_coords), but got shape {}".format(
                dim, coords_rm.shape
            )
        )

    n_params, _ = params.shape

    chex.assert_shape(offset_table_data, (L + 1,))
    chex.assert_shape(coords_rm, (dim, n_coords))
    chex.assert_shape(params, (n_params, F))

    chex.assert_scalar(L)
    chex.assert_scalar(F)
    chex.assert_scalar(N_min)
    chex.assert_scalar(per_level_scale)
    chex.assert_type([L, F, N_min], int)
    chex.assert_type(per_level_scale, float)

    offset_dtype = jax.dtypes.canonicalize_dtype(offset_table_data.dtype)
    if offset_dtype != jnp.uint32:
        raise RuntimeError(
            "hashgrid encoding expects `offset_table_data` (a prefix sum of the hash table sizes of each level) to be of type uint32, got {}".format(offset_dtype)
        )

    coord_dtype = jax.dtypes.canonicalize_dtype(coords_rm.dtype)
    if coord_dtype != jnp.float32:
        raise NotImplementedError(
            "hashgrid encoding is only implemented for input coordinates of type float32, got {}".format(
                coord_dtype
            )
        )

    param_dtype = jax.dtypes.canonicalize_dtype(params.dtype)
    if param_dtype != jnp.float32:
        raise NotImplementedError(
            "hashgrid encoding is only implemented for parameters of type float32, got {}".format(
                param_dtype
            )
        )

    out_shapes = {
        "encoded_coords_rm": (L * F, n_coords),
        "dy_dcoords_rm": (dim * L * F, n_coords),
    }

    return (
        jax.ShapedArray(shape=out_shapes["encoded_coords_rm"], dtype=jnp.float32),
        jax.ShapedArray(shape=out_shapes["dy_dcoords_rm"], dtype=jnp.float32),
    )


def hashgrid_encode_backward_abstract(
    offset_table_data: jax.ShapedArray,
    coords_rm: jax.ShapedArray,
    params: jax.ShapedArray,  # only for determining shape of dL_dparams
    dL_dy_rm: jax.ShapedArray,
    dy_dcoords_rm: jax.ShapedArray,

    # static args
    L: int,
    F: int,
    N_min: int,
    per_level_scale: float,
):
    dim, n_coords = coords_rm.shape
    if dim != 3:
        raise NotImplementedError(
            "hashgrid encoding is only implemented for 3D coordinates, expected input coordinates to have shape ({}, n_coords), but got shape {}".format(
                dim, coords_rm.shape
            )
        )

    n_params, _ = params.shape

    chex.assert_shape(offset_table_data, (L + 1,))
    chex.assert_shape(coords_rm, (dim, n_coords))
    chex.assert_shape(params, (n_params, F))
    chex.assert_shape(dL_dy_rm, (L*F, n_coords))
    chex.assert_shape(dy_dcoords_rm, (dim*L*F, n_coords))

    chex.assert_scalar(L)
    chex.assert_scalar(F)
    chex.assert_scalar(N_min)
    chex.assert_scalar(per_level_scale)
    chex.assert_type([L, F, N_min], int)
    chex.assert_type(per_level_scale, float)

    offset_dtype = jax.dtypes.canonicalize_dtype(offset_table_data.dtype)
    if offset_dtype != jnp.uint32:
        raise RuntimeError(
            "hashgrid encoding expects `offset_table_data` (a prefix sum of the hash table sizes of each level) to be of type uint32, got {}".format(offset_dtype)
        )

    coord_dtype = jax.dtypes.canonicalize_dtype(coords_rm.dtype)
    if coord_dtype != jnp.float32:
        raise NotImplementedError(
            "hashgrid encoding is only implemented for input coordinates of type float32, got {}".format(
                coord_dtype
            )
        )

    param_dtype = jax.dtypes.canonicalize_dtype(params.dtype)
    if param_dtype != jnp.float32:
        raise NotImplementedError(
            "hashgrid encoding is only implemented for parameters of type float32, got {}".format(
                param_dtype
            )
        )

    out_shapes = {
        "dL_dparams": (n_params, F),
        "dL_dcoords_rm": (dim, n_coords),
    }

    return (
        jax.ShapedArray(shape=out_shapes["dL_dparams"], dtype=jnp.float32),
        jax.ShapedArray(shape=out_shapes["dL_dcoords_rm"], dtype=jnp.float32),
    )
