import functools
import math

import chex
import flax.linen as nn
from flax.linen.dtypes import Dtype
import jax
import jax.numpy as jnp
import jax.random as jran

from utils.common import find_smallest_prime_larger_or_equal_than, jit_jaxfn_with, vmap_jaxfn_with


cell_vert_offsets = {
    2: jnp.asarray([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]),
    3: jnp.asarray([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]),
}


class Encoder(nn.Module):
    pass


# TODO: enforce types used in arrays
class HashGridEncoder(Encoder):
    dim: int
    # Let's use the same notations as in the paper

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

    @nn.jit
    @nn.compact
    def __call__(self, pos: jax.Array) -> jax.Array:
        chex.assert_axis_dimension(pos, -1, self.dim)

        # Equation(3)
        # Essentially, it is $(n_max / n_min) ** (1/(L - 1))$
        b = math.exp((math.log(self.N_max) - math.log(self.N_min)) / (self.L - 1))

        resolutions, indexing_methods, offsets = [], [], [0]
        for i in range(self.L):
            res = int(self.N_min * (b**i))
            resolutions.append(res)

            n_entries = (res + 1) ** 2

            if n_entries <= self.T:
                indexing_methods.append(self.onebyone)
            else:
                indexing_methods.append(self.hashing)
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

        # A list-like object, contents are: [
        #   (indices_0, pos_scaled_0, vert_pos_0),
        #   (indices_1, pos_scaled_1, vert_pos_1),
        #   ...
        #   (indices_{L-1}, pos_scaled_{L-1}, vert_pos_{L-1})
        # ], where:
        # indices [L, pos.shape[0], 2**dim] (int):
        #   indices in hash table
        # pos_scaled [L, pos.shape[0], 2**dim] (float):
        #   query points' coordinates scaled to each hierarchies (float)
        # vert_pos [L, pos.shape[0], 2**dim, dim] (int):
        #   vertex coordinates of the enclosing grid cell's vertices (2**dim in total) of each
        #   query point
        indices_posscaled_vertpos_tuples = jax.tree_map(
            lambda index_fn, res: index_fn(pos, self.dim, res, self.T),
            indexing_methods,
            resolutions,
        )

        indices, pos_scaled, vert_pos = map(jnp.asarray, zip(*indices_posscaled_vertpos_tuples))

        # add offsets for each hash grid level
        indices += jnp.asarray(offsets[:-1]).reshape((self.L, 1, 1))

        # [L, pos.shape[0], 2**dim, F]
        vert_latents = latents[indices]
        # [L, pos.shape[0], 2**dim]
        # NOTE:
        #   need to set out_axes=1 to keep the output batch dimension on the 1-th axis
        vert_weights = self.lerp_weights(pos_scaled, vert_pos, self.dim)

        # [L, pos.shape[0], F]
        encodings = (vert_latents * vert_weights[..., None]).sum(axis=-2)
        # [pos.shape[0], L*F]
        encodings = encodings.transpose(1, 0, 2).reshape(-1, self.L * self.F)
        return encodings


    # PERF:
    #   Make sure to jit the vmapped function, and NOT the other way around (always use jit as the
    #   outmost decorator).
    #   The @staticmethod decorator here doesn't degrade performance, but can only be add at top.
    @staticmethod
    @jit_jaxfn_with(static_argnames=["dim"])
    @vmap_jaxfn_with(in_axes=(1, 1, None), out_axes=1)
    def lerp_weights(pos_scaled: jax.Array, vert_pos: jax.Array, dim: int):
        """
        Inputs:
            pos_scaled [L, ..., dim]: coordinates of query points, scaled to the hierarchy in question
            vert_pos [L, ..., 2**dim, dim]: integer coordinates of the grid cells' vertices that enclose each of `pos_scaled`
            dim int: space dimension

        Returns:
            weights [L, ..., 2**dim]: linear interpolation weights w.r.t. each cell vertex
        """
        chex.assert_type([pos_scaled, vert_pos, dim], [float, int, int])
        chex.assert_axis_dimension(pos_scaled, -1, dim)
        chex.assert_axis_dimension(vert_pos, -2, 2**dim)
        chex.assert_axis_dimension(vert_pos, -1, dim)
        chex.assert_scalar(dim)
        # [L, ..., 1, dim]
        pos_offset = pos_scaled[..., None, :] - vert_pos[..., 0:1, :]
        # [L, ..., 2**dim, dim]
        widths = jnp.clip(
            # cell_vert_offsets: [2**dim, 2]
            (1 - cell_vert_offsets[dim]) + (2 * cell_vert_offsets[dim] - 1) * pos_offset,
            0,
            1,
        )
        # [L, ..., 2**dim]
        return jnp.prod(widths, axis=-1)


    @staticmethod
    @jit_jaxfn_with(static_argnames=["dim", "res", "T"])
    @vmap_jaxfn_with(in_axes=(0, None, None, None))
    def onebyone(pos: jax.Array, dim: int, res: int, T: int):
        """
        Inputs:
            res int: resolution of the hierarchy in question
            pos [..., dim]: spatial positions of the query points

        Returns:
            indices [..., 2**dim]: indices of the grid cell's vertices in the hash table
            vert_pos [..., 2**dim, dim]: positions of the grid cell's vertices in the input space
        """
        chex.assert_type([pos, dim, res, T], [float, int, int, int])
        chex.assert_axis_dimension(pos, -1, dim)
        chex.assert_scalar(dim)
        chex.assert_scalar(res)
        chex.assert_scalar(T)
        # [..., dim]
        pos_scaled = pos * res
        # [..., dim]
        pos_floored = jnp.floor(pos_scaled).astype(int)
        # [..., 2**dim, dim]
        vert_pos = pos_floored[..., None, :] + cell_vert_offsets[dim]
        # [..., 2**dim]
        indices = functools.reduce(
            lambda prev, d: prev * res + vert_pos[..., d],
            range(dim),
            1,
        )
        return indices, pos_scaled, vert_pos


    @staticmethod
    @jit_jaxfn_with(static_argnames=["dim", "res", "T"])
    @vmap_jaxfn_with(in_axes=(0, None, None, None))
    def hashing(pos: jax.Array, dim: int, res: int, T: int):
        """
        Inputs:
            res int: resolution of the hierarchy in question
            pos [..., dim]: spatial positions of the query points

        Returns:
            indices [..., 2**dim]: indices of the grid cell's vertices in the hash table
            vert_pos [..., 2**dim, dim]: positions of the grid cell's vertices in the input space
        """
        chex.assert_type([pos, dim, res, T], [float, int, int, int])
        chex.assert_axis_dimension(pos, -1, dim)
        chex.assert_scalar(dim)
        chex.assert_scalar(res)
        chex.assert_scalar(T)
        # [..., dim]
        pos_scaled = pos * res
        # [..., dim]
        pos_floored = jnp.floor(pos_scaled).astype(int)
        # [..., 2**dim, dim]
        vert_pos = pos_floored[..., None, :] + cell_vert_offsets[dim]
        # NOTE:
        #   don't use too large primes (e.g. >= 2**20), because some larger images can easily have
        #   axes with more than 2**11(=2048) pixels, and jax represents integers as int32 by
        #   default, overflows could happen and increase hash collisions.
        primes = (
            1,
            find_smallest_prime_larger_or_equal_than(1<<17),
            find_smallest_prime_larger_or_equal_than(1<<18),
        )
        # [..., 2**dim]
        indices = functools.reduce(
            lambda prev, d: jnp.bitwise_xor(prev, vert_pos[..., d] * primes[d]),
            range(dim),
            0,
        )
        return jnp.mod(indices, T), pos_scaled, vert_pos


class FrequencyEncoder(Encoder):
    """
        Frequency encoding from Equation(4) of the NeRF paper, except the encoded frequency orders
        are different.
    """

    # input dimension
    dim: int
    # number of frequencies
    L: int

    # NOTE:
    #   adding @nn.compact makes this not directly callable (CallCompactUnboundModuleError)
    #   See: <https://flax.readthedocs.io/en/latest/api_reference/flax.errors.html#flax.errors.CallCompactUnboundModuleError>
    #
    # @nn.compact
    # TODO:
    #   using a function for this (vmap, then jit the vmapped function) seems to be faster (~47ms vs
    #   ~57ms)
    @nn.jit  # use nn.jit instead of jax.jit because first argument is a Module
    def __call__(self, pos: jax.Array) -> jax.Array:
        """
        Inuts:
            pos [..., dim]: `dim`-d coordinates to be frequency-encoded

        Returns:
            encodings [..., 2*dim*L]: frequency-encoded coordinates
        """
        chex.assert_axis_dimension(pos, -1, self.dim)
        # [..., dim, L]: 2^{l} * pi * p
        A = jnp.exp2(jnp.arange(self.L)) * jnp.pi * pos[..., None]
        # [..., dim, L], [..., dim, L]
        senc, cenc = jnp.sin(A), jnp.cos(A)
        # [..., dim*L], [..., dim*L]
        senc, cenc = senc.reshape(*A.shape[:-2], self.dim*self.L), cenc.reshape(*A.shape[:-2], self.dim*self.L)
        # [..., 2*dim*L]
        encodings = jnp.concatenate([senc, cenc], axis=-1)
        return encodings
