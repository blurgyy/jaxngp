import jax.numpy as jnp
import jax.random as jran

from utils.common import jit_jaxfn_with
from utils.types import PinholeCamera, RigidTransformation


@jit_jaxfn_with(static_argnames=["camera"])
def make_rays_worldspace(
        camera: PinholeCamera,
        transform_cw: RigidTransformation,
    ):
    """
    Generate world-space rays for each pixel in the given camera's projection plane.

    Inputs:
        camera: camera model in-use
        transform_cw[rotation, translation]: camera to world transformation
            rotation [3, 3]: rotation matrix
            translation [3]: translation vector

    Returns:
        o_world [H*W, 3]: ray origins, in world-space
        d_world [H*W, 3]: ray directions, in world-space
    """
    H, W = camera.H, camera.W
    n_pixels = H * W
    # [H*W, 1]
    d_cam_idcs = jnp.arange(n_pixels).reshape(-1, 1)
    # [H*W, 1]
    d_cam_xs = jnp.mod(d_cam_idcs, camera.W)
    d_cam_xs = ((d_cam_xs + 0.5) - camera.W/2)
    # [H*W, 1]
    d_cam_ys = jnp.floor_divide(d_cam_idcs, camera.W)
    d_cam_ys = -((d_cam_ys + 0.5) - camera.H/2)  # NOTE: y axis indexes from bottom to top, so negate it
    # [H*W, 1]
    d_cam_zs = -camera.focal * jnp.ones_like(d_cam_idcs)
    # [H*W, 3]
    d_cam = jnp.concatenate([d_cam_xs, d_cam_ys, d_cam_zs], axis=-1)
    d_cam /= jnp.linalg.norm(d_cam, axis=-1, keepdims=True) + 1e-15

    # [H*W, 3]
    o_world = jnp.broadcast_to(transform_cw.translation, d_cam.shape)
    # [H*W, 3]
    d_world = d_cam @ transform_cw.rotation.T
    d_world /= jnp.linalg.norm(d_world, axis=-1, keepdims=True) + 1e-15

    return o_world, d_world


@jit_jaxfn_with(static_argnames=["H", "W", "chunk_size"])
def get_indices_chunks(
        KEY: jran.KeyArray,
        H: int,
        W: int,
        chunk_size: int,
    ):
    n_pixels = H * W
    n_chunks = (n_pixels + chunk_size - 1) // chunk_size

    # randomize ray order
    KEY, key = jran.split(KEY, 2)
    indices = jran.permutation(key, H * W)

    # xys has sorted order
    _xs = jnp.mod(jnp.arange(H * W), W)
    _ys = jnp.floor_divide(jnp.arange(H * W), H)
    xys = jnp.concatenate([_xs.reshape(-1, 1), _ys.reshape(-1, 1)], axis=-1)

    return xys, jnp.array_split(indices, n_chunks)
