import jax
import jax.numpy as jnp


# jit rules
def morton3d_abstract(
    # input array
    xyzs: jax.ShapedArray,
):
    length, _ = xyzs.shape

    dtype = jax.dtypes.canonicalize_dtype(xyzs.dtype)
    if dtype != jnp.uint32:
        raise NotImplementedError(
            "morton3d is only implemented for input coordinates of type `jnp.uint32`, got {}".format(
                dtype,
            )
        )

    out_shapes = {
        "idcs": (length,),
    }

    return jax.ShapedArray(shape=out_shapes["idcs"], dtype=jnp.uint32)


def morton3d_invert_abstract(
    # input array
    idcs: jax.ShapedArray,
):
    length, = idcs.shape

    dtype = jax.dtypes.canonicalize_dtype(idcs.dtype)
    if dtype != jnp.uint32:
        raise NotImplementedError(
            "morton3d_invert is only implemented for input indices of type `jnp.uint32`, got {}".format(
                dtype,
            )
        )

    out_shapes = {
        "xyzs": (length, 3),
    }

    return jax.ShapedArray(shape=out_shapes["xyzs"], dtype=jnp.uint32)
