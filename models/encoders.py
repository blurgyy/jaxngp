import math

import chex
import flax.linen as nn
from flax.linen.dtypes import Dtype
import jax
import jax.numpy as jnp
import jax.random as jran

from utils.common import find_smallest_prime_larger_or_equal_than


vert_offsets_2d = jnp.asarray([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])


class HashGridEncoder2D(nn.Module):
    # Lets use the same notations as in the paper

    # Number of levels (16).
    L: int
    # Maximum entries per level (hash table size) (2**14 to 2**24).
    # T is better set to a prime number, because hashing operations requires taking its modulo.
    T: int
    # Number of feature dimensions per entry (2).
    F: int
    # Coarsest resolution (16).
    N_min: int
    # Finest resolution (512 to 524288).
    N_max: int

    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, uv: jax.Array):
        # Equation(3)
        # Essentially, it is $(n_max / n_min) ** (1/(L - 1))$
        b = math.exp((math.log(self.N_max) - math.log(self.N_min)) / (self.L - 1))

        resolutions, indexing_methods, offsets = [], [], [0]
        for i in range(self.L):
            res = int(self.N_min * (b**i))
            resolutions.append(res)

            n_entries = (res + 1) ** 2

            if n_entries <= self.T:
                indexing_methods.append(jax.vmap(self.onebyone, in_axes=(0,None, None)))
            else:
                indexing_methods.append(jax.vmap(self.hashing, in_axes=(0, None, None)))
                n_entries = self.T

            offsets.append(offsets[-1] + n_entries)

        latents = self.param(
            "latent codes stored on grid vertices",
            # paper:
            #   We initialize the hash table entries using the uniform distribution U(−10^{−4}, 10^{−4})
            #   to provide a small amount of randomness while encouraging initial predictions close
            #   to zero.
            lambda key, shape, dtype: jran.uniform(key, shape, dtype, -1e-4, 1e-4),
            (offsets[-1], self.F),
            self.param_dtype,
        )

        # [
        #   (indices_0, uv_scaled_0, vert_uv_0),
        #   (indices_1, uv_scaled_1, vert_uv_1),
        #   ...
        #   (indices_{L-1}, uv_scaled_{L-1}, vert_uv_{L-1})
        # ]
        indices_vertuvs_tuples = jax.tree_map(
            lambda index_fn, res: index_fn(uv, res, self.T),
            indexing_methods,
            resolutions,
        )

        # the indices and vertex coordinates of the enclosing grid's 4 vertices of each query point
        indices, uvs_scaled, vert_uvs = zip(*indices_vertuvs_tuples)

        # [L, uv.shape[0], 4]
        indices = jnp.asarray(indices)
        # [L, uv.shape[0], 2]
        uvs_scaled = jnp.asarray(uvs_scaled)
        # [L, uv.shape[0], 4, 2]
        vert_uvs = jnp.asarray(vert_uvs)
        # add offsets for each level
        indices += jnp.asarray(offsets[:-1]).reshape((self.L, 1, 1))

        # [L, uv.shape[0], 4, F]
        vert_latents = latents[indices]
        # [L, uv.shape[0], 4]
        # NOTE:
        #   need to set out_axes=1 to keep the output batch dimension on the 1-th axis
        vert_weights = jax.vmap(self.lerp_weights, in_axes=(1, 1), out_axes=1)(uvs_scaled, vert_uvs)

        # [L, uv.shape[0], F]
        encodings = (vert_latents * vert_weights[..., None]).sum(axis=-2)
        # [uv.shape[0], L*F]
        encodings = encodings.transpose(1, 0, 2).reshape(-1, self.L * self.F)
        return encodings


    @staticmethod
    @jax.jit
    def lerp_weights(uv_scaled, vert_uv):
        chex.assert_type([uv_scaled, vert_uv], [float, int])
        # uv_scaled: [L, ..., 2]
        # vert_uv: [L, ..., 4, 2]
        # [L, ..., 2]
        pos_offset = uv_scaled[..., None, :] - vert_uv[..., 0:1, :]
        # [L, ..., 4, 2]
        widths = jnp.clip(
            # vert_offsets_2d: [4, 2]
            (1 - vert_offsets_2d) + (2 * vert_offsets_2d - 1) * pos_offset,
            0,
            1,
        )
        # [L, ..., 4]
        return jnp.prod(widths, axis=-1)


    @staticmethod
    @jax.jit  # does this jit matter?
    def onebyone(uv: jax.Array, res: int, T: int):
        """
        Inputs:
            res int
            uv [..., 2]
        Returns:
            indices [..., 4]
            vert_uv [..., 4, 2]
        """
        chex.assert_type(uv, float)
        # [..., 2]
        uv_scaled = uv * res
        # [..., 2]
        uv_floored = jnp.floor(uv_scaled).astype(int)
        # [..., 4, 2]
        vert_uv = uv_floored[..., None, :] + vert_offsets_2d
        # [..., 4]
        return vert_uv[..., 0] * res + vert_uv[..., 1], uv_scaled, vert_uv


    @staticmethod
    @jax.jit  # does this jit matter?
    def hashing(uv: jax.Array, res: int, T: int):
        """
        Inputs:
            res int
            uv [..., 2]
        Returns:
            indices [..., 4]
            vert_uv [..., 4, 2]
        """
        chex.assert_type(uv, float)
        # [..., 2]
        uv_scaled = uv * res
        # [..., 2]
        uv_floored = jnp.floor(uv_scaled).astype(int)
        # [..., 4, 2]
        vert_uv = uv_floored[..., None, :] + vert_offsets_2d
        # NOTE:
        #   don't use too large primes (e.g. >= 2**20), because some larger images can easily have
        #   axes with more than 2**11(=2048) pixels, and jax represents integers as int32 by
        #   default, overflows could happen and increase hash collisions.
        primes = (1, find_smallest_prime_larger_or_equal_than(2**17))
        # [..., 4]
        indices = jnp.bitwise_xor(
            vert_uv[..., 0] * primes[0],
            vert_uv[..., 1] * primes[1],
        )
        return jnp.mod(indices, T), uv_scaled, vert_uv


def _frequency_encoding_2d(uv: jax.Array, L: int):
    """
    Frequency encoding from Equation(4) of the NeRF paper, except the encoded frequency orders are
    different.

    Inuts:
        uv [..., 2]: 2D coordinates to be frequency-encoded
        L [L]: number of frequencies

    Returns:
        encodings [..., 4*L]: frequency-encoded coordinates
    """
    # [..., 2, L]: 2^{l} * pi * p
    A = jnp.exp2(jnp.arange(L)) * jnp.pi * uv[..., None]
    # [..., 2, L], [..., 2, L]
    senc, cenc = jnp.sin(A), jnp.cos(A)
    # [..., 2*L], [..., 2*L]
    senc, cenc = senc.reshape(*A.shape[:-2], 2*L), cenc.reshape(*A.shape[:-2], 2*L)
    # [..., 4*L]
    encodings = jnp.concatenate([senc, cenc], axis=-1)
    return encodings

_frequency_encoding_2d = jax.jit(_frequency_encoding_2d, static_argnums=(1,))
frequency_encoding_2d = jax.vmap(_frequency_encoding_2d, in_axes=(0, None))
