import functools
from typing import Callable, List, Optional, Tuple

import flax.linen as nn
import jax
from jax.nn.initializers import Initializer
import jax.numpy as jnp

from models.encoders import (
    Encoder,
    FrequencyEncoder,
    HashGridEncoder,
    SphericalHarmonicsEncoder,
    SphericalHarmonicsEncoderCuda,
)
from utils.common import mkValueError
from utils.types import ActivationType, DirectionalEncodingType, PositionalEncodingType


class NeRF(nn.Module):
    bound: float

    position_encoder: Encoder
    direction_encoder: Encoder

    density_mlp: nn.Module
    rgb_mlp: nn.Module

    density_activation: Callable
    rgb_activation: Callable

    @nn.compact
    def __call__(self, xyz: jax.Array, dir: Optional[jax.Array]) -> Tuple[jax.Array, jax.Array]:
        """
        Inputs:
            xyz [..., 3]: coordinates in $\R^3$.
            dir [..., 3]: **unit** vectors, representing viewing directions.  If `None`, only
                           return densities.

        Returns:
            density [..., 1]: density (ray terminating probability) of each query points
            rgb [..., 3]: predicted color for each query point
        """
        original_aux_shapes = xyz.shape[:-1]
        xyz = xyz.reshape(-1, 3)
        # scale and translate xyz coordinates into unit cube
        xyz = (xyz + self.bound) / (2 * self.bound)

        # [..., D_pos]
        pos_enc = self.position_encoder(xyz)

        x = self.density_mlp(pos_enc)
        # [..., 1], [..., density_MLP_out-1]
        density, _ = jnp.split(x, [1], axis=-1)

        if dir is None:
            return density.reshape(*original_aux_shapes, 1)
        dir = dir.reshape(-1, 3)

        # [..., D_dir]
        dir_enc = self.direction_encoder(dir)
        # [..., 3]
        rgb = self.rgb_mlp(jnp.concatenate([x, dir_enc], axis=-1))

        density, rgb = self.density_activation(density), self.rgb_activation(rgb)

        return density.reshape(*original_aux_shapes, 1), rgb.reshape(*original_aux_shapes, 3)


class CoordinateBasedMLP(nn.Module):
    "Coordinate-based MLP"

    # hidden layer widths
    Ds: List[int]
    out_dim: int
    skip_in_layers: List[int]

    # as described in the paper
    kernel_init: Initializer=nn.initializers.glorot_uniform()

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        in_x = x
        for i, d in enumerate(self.Ds):
            if i in self.skip_in_layers:
                x = jnp.concatenate([in_x, x], axis=-1)
            x = nn.Dense(
                d,
                use_bias=False,
                kernel_init=self.kernel_init,
            )(x)
            x = nn.relu(x)
        x = nn.Dense(
            self.out_dim,
            use_bias=False,
            kernel_init=self.kernel_init,
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
            "Exponential function, except its gradient calculation uses a truncated input value"
            return jnp.exp(x)
        def __fwd_trunc_exp(x):
            y = trunc_exp(x)
            aux = x  # aux contains additional information that is useful in the backward pass
            return y, aux
        def __bwd_trunc_exp(aux, grad_y):
            # REF: <https://github.com/NVlabs/instant-ngp/blob/d0d35d215c7c63c382a128676f905ecb676fa2b8/src/testbed_nerf.cu#L303>
            grad_x = jnp.exp(jnp.clip(aux, -15, 15)) * grad_y
            return (grad_x, )
        trunc_exp.defvjp(
            fwd=__fwd_trunc_exp,
            bwd=__bwd_trunc_exp,
        )
        return trunc_exp

    elif act == "thresholded_exponential":
        def thresh_exp(x, thresh):
            """
            Exponential function translated along -y direction by 1e-2, and thresholded to have
            non-negative values.
            """
            # paper:
            #   the occupancy grids ... is updated every 16 steps ... corresponds to thresholding
            #   the opacity of a minimal ray marching step by 1 − exp(−0.01) ≈ 0.01
            return nn.relu(jnp.exp(x) - thresh)
        return functools.partial(thresh_exp, thresh=1e-2)

    elif act == "truncated_thresholded_exponential":
        @jax.custom_vjp
        def trunc_thresh_exp(x, thresh):
            """
            Exponential, but value is translated along -y axis by value `thresh`, negative values
            are removed, and gradient is truncated.
            """
            return nn.relu(jnp.exp(x) - thresh)
        def __fwd_trunc_threash_exp(x, thresh):
            y = trunc_thresh_exp(x, thresh=thresh)
            aux = x, thresh  # aux contains additional information that is useful in the backward pass
            return y, aux
        def __bwd_trunc_threash_exp(aux, grad_y):
            x, thresh = aux
            grad_x = jnp.exp(jnp.clip(x, -15, 15)) * grad_y
            # clip gradient for values that has been thresholded by relu during forward pass
            grad_x = jnp.signbit(jnp.log(thresh) - x) * grad_x
            # first tuple element is gradient for input, second tuple element is gradient for the
            # `threshold` value.
            return (grad_x, 0)
        trunc_thresh_exp.defvjp(
            fwd=__fwd_trunc_threash_exp,
            bwd=__bwd_trunc_threash_exp,
        )
        return functools.partial(trunc_thresh_exp, thresh=1e-2)
    elif act == "relu":
        return nn.relu
    else:
        raise mkValueError(
            desc="activation",
            value=act,
            type=ActivationType,
        )


def make_nerf(
    bound: float,

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
    if pos_enc == "identity":
        position_encoder = lambda x: x
    elif pos_enc == "frequency":
        raise NotImplementedError("Frequency encoding for NeRF is not tuned")
        position_encoder = FrequencyEncoder(dim=3, L=10)
    elif pos_enc == "hashgrid":
        position_encoder = HashGridEncoder(
            dim=3,
            L=pos_levels,
            T=2**19,
            F=2,
            N_min=2**4,
            N_max=int(2**11 * bound),
            param_dtype=jnp.float32,
        )
    else:
        raise mkValueError(
            desc="positional encoding",
            value=pos_enc,
            type=PositionalEncodingType,
        )

    if dir_enc == "identity":
        direction_encoder = lambda x: x
    elif dir_enc == "sh":
        direction_encoder = SphericalHarmonicsEncoder(L=dir_levels)
    elif dir_enc == "shcuda":
        direction_encoder = SphericalHarmonicsEncoderCuda(L=dir_levels)
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
        bound=bound,

        position_encoder=position_encoder,
        direction_encoder=direction_encoder,

        density_mlp=density_mlp,
        rgb_mlp=rgb_mlp,

        density_activation=density_activation,
        rgb_activation=rgb_activation,
    )

    return model


def make_nerf_ngp(bound: float) -> NeRF:
    return make_nerf(
        bound=bound,

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


def make_debug_nerf(bound: float) -> NeRF:
    return NeRF(
        bound=bound,
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


def make_test_cube(width: int, bound: float, density: float=32) -> NeRF:
    @jax.jit
    @jax.vmap
    def cube_density_fn(x: jax.Array) -> jax.Array:
        # x is pre-normalized unit cube, we map it back to specified aabb.
        x = (x + bound) / (2 * bound)
        mask = (abs(x).max(axis=-1, keepdims=True) <= width/2).astype(float)
        # concatenate input xyz for use in later rgb querying
        return jnp.concatenate([density * mask, x], axis=-1)

    @jax.jit
    @jax.vmap
    def cube_rgb_fn(density_xyz_dir: jax.Array) -> jax.Array:
        # xyz(3) + dir(3), only take xyz to infer color
        x = density_xyz_dir[:3]
        x = jnp.clip(x, -width/2, width/2)
        return x / width + .5

    return NeRF(
        bound=bound,
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

    KEY = jran.PRNGKey(0)
    KEY, key = jran.split(KEY, 2)

    m = make_nerf_ngp()

    xyz = jnp.ones((100, 3))
    dir = jnp.ones((100, 2))
    params = m.init(key, xyz, dir)
    print(m.tabulate(key, xyz, dir))

    m = make_test_cube(
        width=2,
        bound=1,
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
