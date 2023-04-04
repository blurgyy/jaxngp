import jax

from . import impl


def packbits(
    density_threshold: float,
    density_grid: jax.Array,
) -> jax.Array:
    """
    Pack the given `density_grid` into a compact representation of type uint8, where each bit is
    high if its corresponding density grid cell's density is larger than `density_threshold`, low
    otherwise.
    """
    return impl.packbits_p.bind(
        density_grid,
        density_threshold=density_threshold,
    )
