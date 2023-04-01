import jax
import jax.numpy as jnp
from flax.struct import dataclass


@dataclass
class OccupancyDensityGrid:
    # number of cascades
    # paper:
    #   𝐾 = 1 for all synthetic NeRF scenes (single grid) and 𝐾 ∈ [1, 5] for larger real-world
    #   scenes (up to 5 grids, depending on scene size)
    K: int
    # uint8, each bit is an occupancy value of a grid cell
    occupancy: jax.Array
    # float32, full-precision density values
    density: jax.Array

    @classmethod
    def create(cls, cascades: int, grid_resolution: int=128):
        """
        Example usage:
            ogrid = OccupancyDensityGrid.create(cascades=5, grid_resolution=128)
        """
        occupancy = jnp.zeros(
            shape=(cascades*grid_resolution**3 // 8,),  # every bit is an occupancy value
            dtype=jnp.uint8,
        )
        density = jnp.zeros(
            shape=(cascades*grid_resolution**3,),
            dtype=jnp.float32,
        )
        return cls(K=cascades, occupancy=occupancy, density=density)
