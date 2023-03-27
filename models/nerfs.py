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
    mkValueError,
)
from utils.types import AABB


class NeRF(nn.Module):
    aabb: AABB

    position_encoder: Encoder
    direction_encoder: Encoder

    density_mlp: nn.Module
    rgb_mlp: nn.Module

    density_activation: Callable
    rgb_activation: Callable

    # TODO:
    #   * input "dir" does not need to be batched
    #   * use vmap
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
        def out_of_unit_cube(xyz: jax.Array, eps: float=1e-3):
            "below expression effectively tests if xyz is **outside** of the AABB [eps, 1-eps]^3"
            return jnp.signbit(xyz - eps) ^ jnp.signbit(1 - eps - xyz)

        # scale and translate xyz coordinates into unit cube
        bbox = jnp.asarray(self.aabb)
        xyz = (xyz - bbox[:, 0]) / (bbox[:, 1] - bbox[:, 0])
        # calculate mask for zeroing out-of-bound inputs, applying this mask (see below `jnp.where`
        # calls) gives a ~2x speed boost.
        oob = out_of_unit_cube(xyz)
        oob = oob[:, 0:1] | oob[:, 1:2] | oob[:, 2:3]

        # [..., D_pos]
        pos_enc = self.position_encoder(xyz)
        pos_enc = jnp.where(
            oob,
            jnp.zeros_like(pos_enc),
            pos_enc,
        )
        # [..., D_dir]
        dir_enc = self.direction_encoder(dir)
        dir_enc = jnp.where(
            oob,
            jnp.zeros_like(dir_enc),
            dir_enc,
        )

        x = self.density_mlp(pos_enc)
        # [..., 1], [..., density_MLP_out-1]
        density, x = jnp.split(x, [1], axis=-1)

        # paper:
        #   The color MLP adds view-dependent color variation. Its input is the concatenation of
        #   • the 16 output values of the density MLP, and
        #   • the view direction projected onto the first 16 coefficients of the spherical harmonics
        #     basis (i.e. up to degree 4). This is a natural frequency encoding over unit vectors.
        # x = nn.relu(x)

        # [..., 3]
        # rgb = self.rgb_mlp(jnp.concatenate([x[..., 1:], dir_enc], axis=-1)) 
        rgb = self.rgb_mlp(jnp.concatenate([x, dir_enc], axis=-1))

        density, rgb = self.density_activation(density), self.rgb_activation(rgb)

        return density, rgb


class CoordinateBasedMLP(nn.Module):
    "Coordinate-based MLP"

    # hidden layer widths
    Ds: List[int]
    out_dim: int
    skip_in_layers: List[int]

    # as described in the paper
    kernel_init: Initializer=nn.initializers.glorot_uniform()
    bias_init: Initializer=nn.initializers.uniform()

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        in_x = x
        for i, d in enumerate(self.Ds):
            if i in self.skip_in_layers:
                x = jnp.concatenate([in_x, x], axis=-1)
            x = nn.Dense(
                d,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )(x)
            x = nn.relu(x)
        x = nn.Dense(
            self.out_dim,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        return x


def make_activation(act: ActivationType):
    if act == "sigmoid":
        return nn.sigmoid
    elif act == "exponential":
        return jnp.exp
    elif act == "truncated_exponential":
        @jax.custom_vjp
        def trunc_exp(x):
            "Exponential function, except its gradient is capped with a maximum absolute value of 15.0"
            return jnp.exp(x)
        def __fwd_trunc_exp(x):
            y = trunc_exp(x)
            aux = jnp.exp(x)  # aux contains additional information that is useful in the backward pass
            return y, aux
        def __bwd_trunc_exp(aux, grad_y):
            grad_x = jnp.clip(aux * grad_y, -15, 15)
            return (grad_x, )
        trunc_exp.defvjp(
            fwd=__fwd_trunc_exp,
            bwd=__bwd_trunc_exp,
        )

        return trunc_exp
    elif act == "relu":
        return nn.relu
    else:
        raise mkValueError(
            desc="activation",
            value=act,
            type=ActivationType,
        )


def make_nerf(
        aabb: AABB,

        # encodings
        pos_enc: PositionalEncodingType,
        dir_enc: DirectionalEncodingType,

        # encoding levels
        pos_levels: int,
        dir_levels: int,

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
        bound_max = max(
            [
                aabb[0][1] - aabb[0][0],
                aabb[1][1] - aabb[1][0],
                aabb[2][1] - aabb[2][0],
            ]
        ) / 2
        position_encoder = HashGridEncoder(
            dim=3,
            L=pos_levels,
            T=2**19,
            F=2,
            N_min=2**4,
            N_max=int(2**11 * bound_max),
            param_dtype=jnp.float32,
        )
    else:
        raise mkValueError(
            desc="positional encoding",
            value=pos_enc,
            type=PositionalEncodingType,
        )

    if dir_enc == "sh":
        direction_encoder = SphericalHarmonicsEncoder(L=dir_levels)
    else:
        raise mkValueError(
            desc="directional encoding",
            value=dir_enc,
            type=DirectionalEncodingType,
        )

    density_mlp = CoordinateBasedMLP(
        Ds=density_Ds,
        out_dim=density_out_dim,
        skip_in_layers=density_skip_in_layers
    )
    rgb_mlp = CoordinateBasedMLP(
        Ds=rgb_Ds,
        out_dim=rgb_out_dim,
        skip_in_layers=rgb_skip_in_layers
    )

    density_activation = make_activation(density_act)
    rgb_activation = make_activation(rgb_act)

    model = NeRF(
        aabb=aabb,

        position_encoder=position_encoder,
        direction_encoder=direction_encoder,

        density_mlp=density_mlp,
        rgb_mlp=rgb_mlp,

        density_activation=density_activation,
        rgb_activation=rgb_activation,
    )

    return model


def make_nerf_ngp(aabb: AABB) -> NeRF:
    return make_nerf(
        aabb=aabb,

        pos_enc="hashgrid",
        dir_enc="sh",

        pos_levels=16,
        dir_levels=4,

        density_Ds=[64],
        density_out_dim=16,
        density_skip_in_layers=[],
        density_act="truncated_exponential",

        rgb_Ds=[64, 64],
        rgb_out_dim=3,
        rgb_skip_in_layers=[],
        rgb_act="sigmoid",
    )


def make_debug_nerf(aabb: AABB) -> NeRF:
    return NeRF(
        aabb=aabb,
        position_encoder=lambda x: x,
        direction_encoder=lambda x: x,
        density_mlp=CoordinateBasedMLP(
            Ds=[64],
            out_dim=16,
            skip_in_layers=[],
        ),
        rgb_mlp=CoordinateBasedMLP(
            Ds=[64, 64],
            out_dim=3,
            skip_in_layers=[],
        ),
        density_activation=lambda x: x,
        rgb_activation=lambda x: x,
    )


def make_test_cube(width: int, aabb: AABB, density: float=32) -> NeRF:
    @jax.jit
    @jax.vmap
    def cube_density_fn(x: jax.Array) -> jax.Array:
        # x is pre-normalized unit cube, we map it back to specified aabb.
        bbox = jnp.asarray(aabb)  # [3, 2]
        x = x * (bbox[:, 1] - bbox[:, 0]) + bbox[:, 0]
        mask = (abs(x).max(axis=-1, keepdims=True) <= width/2).astype(float)
        # concatenate input xyz for use in later rgb querying
        return jnp.concatenate([density * mask, x], axis=-1)

    @jax.jit
    @jax.vmap
    def cube_rgb_fn(density_xyz_dir: jax.Array) -> jax.Array:
        # density(1) + xyz(3) + dir(3), only take xyz to infer color
        x = density_xyz_dir[1:4]
        x = jnp.clip(x, -width/2, width/2)
        return x / width + .5

    return NeRF(
        aabb=aabb,
        position_encoder=lambda x: x,
        direction_encoder=lambda x: x,
        density_mlp=cube_density_fn,
        rgb_mlp=cube_rgb_fn,
        density_activation=lambda x: x,
        rgb_activation=lambda x: x,
    )


def main():
    import jax.numpy as jnp
    import jax.random as jran

    K = jran.PRNGKey(0)
    K, key = jran.split(K, 2)

    m = make_nerf_ngp()

    xyz = jnp.ones((100, 3))
    dir = jnp.ones((100, 2))
    params = m.init(key, xyz, dir)
    print(m.tabulate(key, xyz, dir))

    m = make_test_cube(
        width=2,
        aabb=[[-1, 1]] * 3,
        density=32,
    )
    # params = m.init(key, xyz, dir)
    # print(m.tabulate(key, xyz, dir))
    density, rgb = m.apply(
        {},
        jnp.asarray([[0, 0, 0], [1, 1, 1], [1.1, 0, 0], [0.6, 0.9, -0.5], [0.99, 0.99, 0.99]]),
        jnp.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]),
    )
    print(density)
    print(rgb)


if __name__ == "__main__":
    main()
