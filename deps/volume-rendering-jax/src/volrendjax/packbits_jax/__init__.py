from typing import Tuple
import jax
import jax.numpy as jnp

from . import impl


def packbits(
    density_threshold: float,
    density_grid: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """
    Pack the given `density_grid` into a compact representation of type uint8, where each bit is
    high if its corresponding density grid cell's density is larger than `density_threshold`, low
    otherwise.

    Inputs:
        density_threshold `broadcastable to [N]`
        density_grid `[N]`

    Returns:
        occ_mask `[N] bool`: boolean mask that indicates whether this grid is occupied
        occ_bitfield `[N//8]`
    """
    return impl.packbits_p.bind(
        jnp.broadcast_to(density_threshold, density_grid.shape),
        density_grid,
    )
