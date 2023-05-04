from typing import Dict, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jran
import optax

from models.renderers import render_rays_train
from utils import common, data
from utils.types import NeRFState, PinholeCamera


__all__ = [
    "train_step",
]


@common.jit_jaxfn_with(static_argnames=["total_samples"])
def train_step(
    KEY: jran.KeyArray,
    state: NeRFState,
    total_samples: int,
    camera: PinholeCamera,
    all_xys: jax.Array,
    all_rgbas: jax.Array,
    all_transforms: jax.Array,
    perm: jax.Array,
) -> Tuple[NeRFState, Dict[str, Union[jax.Array, float]]]:
    # TODO:
    #   merge this and `models.renderers.make_rays_worldspace` as a single function
    def make_rays_worldspace() -> Tuple[jax.Array, jax.Array]:
        # [N, 2]
        xys = all_xys[perm]
        # [N, 3]
        xyzs = jnp.concatenate([xys, jnp.ones((xys.shape[0], 1))], axis=-1)
        # [N, 1]
        d_cam_xs = xyzs[:, 0:1]
        d_cam_xs = ((d_cam_xs + 0.5) - camera.cx) / camera.fx
        # [N, 1]
        d_cam_ys = xyzs[:, 1:2]
        d_cam_ys = -((d_cam_ys + 0.5) - camera.cy) / camera.fy
        # [N, 1]
        d_cam_zs = -xyzs[:, 2:3]
        # [N, 3]
        d_cam = jnp.concatenate([d_cam_xs, d_cam_ys, d_cam_zs], axis=-1)
        d_cam /= jnp.linalg.norm(d_cam, axis=-1, keepdims=True) + 1e-15

        # indices of views, used to retrieve transformation information for each ray
        view_idcs = perm // camera.n_pixels
        # [N, 3]
        o_world = all_transforms[view_idcs, -3:]  # WARN: using `perm` instead of `view_idcs` here
                                                  # will silently clip the out-of-bounds indices.
                                                  # REF:
                                                  #   <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#out-of-bounds-indexing>
        # [N, 3, 3]
        R_cws = all_transforms[view_idcs, :9].reshape(-1, 3, 3)
        # [N, 3]
        # equavalent to performing `d_cam[i] @ R_cws[i].T` for each i in [0, N)
        d_world = (d_cam[:, None, :] * R_cws).sum(-1)

        # d_cam was normalized already, normalize d_world just to be sure
        d_world /= jnp.linalg.norm(d_world, axis=-1, keepdims=True) + 1e-15

        return o_world, d_world

    def loss_fn(params, gt_rgba, KEY):
        o_world, d_world = make_rays_worldspace()
        if state.use_background_model:
            # NOTE: use `params` (from loss_fn's inputs) instead of `state.params` (from
            # train_step's inputs), as gradients are conly computed w.r.t. loss_fn's inputs.
            bg = state.bg_fn({"params": params["bg"]}, o_world, d_world)
        elif state.render.random_bg:
            KEY, key = jran.split(KEY, 2)
            bg = jran.uniform(key, shape=(o_world.shape[0], 3), dtype=jnp.float32, minval=0, maxval=1)
        else:
            bg = jnp.asarray(state.render.bg)
        KEY, key = jran.split(KEY, 2)
        batch_metrics, _, pred_rgbs, _ = render_rays_train(
            KEY=key,
            o_world=o_world,
            d_world=d_world,
            bg=bg,
            total_samples=total_samples,
            state=state.replace(params=params),
        )
        gt_rgbs = data.blend_rgba_image_array(imgarr=gt_rgba, bg=bg)
        # from NVlabs/instant-ngp/commit/d6c7241de9be5be1b6d85fe43e446d2eb042511b
        # Note: we divide the huber loss by a factor of 5 such that its L2 region near zero
        # matches with the L2 loss and error numbers become more comparable. This allows reading
        # off dB numbers of ~converged models and treating them as approximate PSNR to compare
        # with other NeRF methods. Self-normalizing optimizers such as Adam are agnostic to such
        # constant factors; optimization is therefore unaffected.
        loss = optax.huber_loss(pred_rgbs, gt_rgbs, delta=0.1).mean() / 5.0
        return loss, batch_metrics

    loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    KEY, key = jran.split(KEY, 2)
    (loss, batch_metrics), grads = loss_grad_fn(state.params, all_rgbas[perm], key)
    state = state.apply_gradients(grads=grads)
    metrics = {
        "loss": loss * perm.shape[0],
        **batch_metrics,
    }
    return state, metrics
