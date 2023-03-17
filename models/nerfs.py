from typing import Callable, List, Tuple

import flax.linen as nn
import jax
from jax.nn.initializers import Initializer
import jax.numpy as jnp

from models.encoders import (
    Encoder,
    FrequencyEncoder,
    HashGridEncoder,
    SphericalHarmonicsEncoder,
)
from utils.common import (
    ActivationType,
    DirectionalEncodingType,
    PositionalEncodingType,
    find_smallest_prime_larger_or_equal_than,
    mkValueError,
)


class NeRF(nn.Module):
    position_encoder: Encoder
    direction_encoder: Encoder

    density_mlp: nn.Module
    rgb_mlp: nn.Module

    density_activation: Callable
    rgb_activation: Callable

    @nn.compact
    def __call__(self, xyz: jax.Array, dir: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Inputs:
            xyz [..., 3]: coordinates in $\R^3$.
            dirs [..., 3]: **unit** vectors, representing viewing directions.

        Returns:
            density [..., 1]: density (ray terminating probability) of each query points
            rgb [..., 3]: predicted color for each query point
        """
        # [..., D_pos]
        pos_enc = self.position_encoder(xyz)
        # [..., D_dir]
        dir_enc = self.direction_encoder(dir)

        x = self.density_mlp(pos_enc)
        # [...]
        density = x[..., 0]

        # [..., 3]
        # rgb = self.rgb_mlp(jnp.concatenate([x[..., 1:], dir_enc], axis=-1)) 
        rgb = self.rgb_mlp(jnp.concatenate([x, dir_enc], axis=-1))

        return self.density_activation(density), self.rgb_activation(rgb)


class PositionBasedMLP(nn.Module):
    "Position-based MLP"

    # hidden layer widths
    Ds: List[int]
    out_dim: int
    skip_in_layers: List[int]

    # as described in the paper
    kernel_init: Initializer=nn.initializers.glorot_uniform()
    bias_init: Initializer=nn.initializers.zeros

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        in_x = x
        for i, d in enumerate(self.Ds):
            if i in self.skip_in_layers:
                x = jnp.concatenate([in_x, x], axis=-1)
            x = nn.Dense(d)(x)
        return nn.Dense(self.out_dim)(x)


def make_activation(act: ActivationType):
    if act == "sigmoid":
        return nn.sigmoid
    elif act == "exponential":
        return jnp.exp
    else:
        raise mkValueError(
            desc="activation",
            value=act,
            type=ActivationType,
        )


def make_nerf(
        # encodings
        pos_enc: PositionalEncodingType,
        dir_enc: DirectionalEncodingType,

        # layer widths
        density_Ds: List[int],
        rgb_Ds: List[int],

        # output dimensions
        density_out_dim: int,
        rgb_out_dim: int,

        # skip connections
        density_skip_in_layers: List[int],
        rgb_skip_in_layers: List[int],

        # activations
        density_act: ActivationType,
        rgb_act: ActivationType,
    ) -> NeRF:
    if pos_enc == "frequency":
        raise NotImplementedError("Frequency encoding for NeRF is not tuned")
        position_encoder = FrequencyEncoder(dim=3, L=10)
    elif pos_enc == "hashgrid":
        position_encoder = HashGridEncoder(
            dim=3,
            L=16,
            T=find_smallest_prime_larger_or_equal_than(2**20),
            F=2,
            N_min=16,
            N_max=2**19,
            param_dtype=jnp.float32,
        )
    else:
        raise mkValueError(
            desc="positional encoding",
            value=pos_enc,
            type=PositionalEncodingType,
        )

    if dir_enc == "sh":
        direction_encoder = SphericalHarmonicsEncoder(L=4)
    else:
        raise mkValueError(
            desc="directional encoding",
            value=dir_enc,
            type=DirectionalEncodingType,
        )

    density_mlp = PositionBasedMLP(
        Ds=density_Ds,
        out_dim=density_out_dim,
        skip_in_layers=density_skip_in_layers
    )
    rgb_mlp = PositionBasedMLP(
        Ds=rgb_Ds,
        out_dim=rgb_out_dim,
        skip_in_layers=rgb_skip_in_layers
    )

    density_activation = make_activation(density_act)
    rgb_activation = make_activation(rgb_act)

    model = NeRF(
        position_encoder=position_encoder,
        direction_encoder=direction_encoder,

        density_mlp=density_mlp,
        rgb_mlp=rgb_mlp,

        density_activation=density_activation,
        rgb_activation=rgb_activation,
    )

    return model


def make_nerf_ngp() -> NeRF:
    return make_nerf(
        pos_enc="hashgrid",
        dir_enc="sh",

        density_Ds=[64],
        density_out_dim=16,
        density_skip_in_layers=[],
        density_act="exponential",

        rgb_Ds=[64, 64],
        rgb_out_dim=3,
        rgb_skip_in_layers=[],
        rgb_act="sigmoid",
    )


def make_test_cube(width: int) -> NeRF:
    @jax.jit
    @jax.vmap
    def cube_density_fn(x: jax.Array) -> jax.Array:
        density = (abs(x).max(axis=-1, keepdims=True) <= width/2).astype(float)
        # concatenate input xyz for use in later rgb querying
        return jnp.concatenate([density, x], axis=-1)

    @jax.jit
    @jax.vmap
    def cube_rgb_fn(density_xyz_dir: jax.Array) -> jax.Array:
        # density(1) + xyz(3) + dir(3), only take xyz to infer color
        x = density_xyz_dir[1:4]
        x = jnp.clip(x, -width/2, width/2)
        return x / width + .5

    return NeRF(
        position_encoder=lambda x: x,
        direction_encoder=lambda x: x,
        density_mlp=cube_density_fn,
        rgb_mlp=cube_rgb_fn,
        density_activation=lambda x: x,
        rgb_activation=lambda x: x,
    )


if __name__ == "__main__":
    import jax.numpy as jnp
    import jax.random as jran

    K = jran.PRNGKey(0)
    K, key = jran.split(K, 2)

    m = make_nerf_ngp()

    xyz = jnp.ones((100, 3))
    dir = jnp.ones((100, 2))
    params = m.init(key, xyz, dir)
    print(m.tabulate(key, xyz, dir))

    m = make_test_cube(2)
    # params = m.init(key, xyz, dir)
    # print(m.tabulate(key, xyz, dir))
    density, rgb = m.apply(
        {},
        xyz=jnp.asarray([[0, 0, 0], [1, 1, 1], [1.1, 0, 0], [0.6, 0.9, -0.5], [0.99, 0.99, 0.99]]),
        dir=jnp.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]),
    )
    print(density)
    print(rgb)
