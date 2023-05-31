from typing import Literal

import chex
import flax.linen as nn
from flax.linen.dtypes import Dtype
import jax

from models.encoders import FrequencyEncoder, HashGridEncoder


class ImageFitter(nn.Module):
    encoding: Literal["hashgrid", "frequency"]
    encoding_dtype: Dtype

    @nn.compact
    def __call__(self, uv: jax.Array) -> jax.Array:
        """
        Inputs:
            uv [..., 2]: coordinates in $\\R^2$ (normalized in range [0, 1]).

        Returns:
            rgb [..., 3]: predicted color for each input uv coordinate (normalized in range [0, 1]).
        """
        chex.assert_axis_dimension(uv, -1, 2)

        if self.encoding == "hashgrid":
            # [..., L*F]
            x = HashGridEncoder(
                dim=2,
                L=16,
                # ~1Mi entries per level
                T=2**20,
                F=2,
                N_min=16,
                N_max=2**19,  # 524288
                param_dtype=self.encoding_dtype,
            )(uv)
        elif self.encoding == "frequency":
            # [..., dim*L]
            x = FrequencyEncoder(dim=2, L=10)(uv)
        else:
            raise ValueError("Unexpected encoding type '{}'".format(self.encoding))

        DenseLayer = lambda dim, name: nn.Dense(
                features=dim,
                name=name,
                # the paper uses glorot initialization, in practice glorot initialization converges
                # to a better result than kaiming initialization, though the gap is small.
                # TODO:
                #   experiment with initializers (or not)
                kernel_init=nn.initializers.lecun_normal(),
                bias_init=nn.initializers.zeros,
                param_dtype=x.dtype
            )
        # feed to the MLP
        x = DenseLayer(128, name="linear1")(x)
        x = nn.relu(x)
        x = DenseLayer(128, name="linear2")(x)
        x = nn.relu(x)

        if self.encoding == "frequency":
            x = nn.relu(DenseLayer(256, name="linear3")(x))
            x = nn.relu(DenseLayer(512, name="linear4")(x))
            x = nn.relu(DenseLayer(512, name="linear5")(x))
            x = nn.relu(DenseLayer(512, name="linear6")(x))
            x = nn.relu(DenseLayer(512, name="linear7")(x))

        x = nn.Dense(3, name="color_predictor", param_dtype=x.dtype)(x)
        rgb = nn.sigmoid(x)

        return rgb
