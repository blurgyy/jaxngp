from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as jran
import optax

from models.renderers import render_rays_train
from utils import common, data
from utils.types import NeRFState, SceneData


__all__ = [
    "train_step",
]


@common.jit_jaxfn_with(static_argnames=["total_samples"])
def train_step(
    KEY: jran.KeyArray,
    state: NeRFState,
    total_samples: int,
    scene: SceneData,
    perm: jax.Array,
) -> Tuple[NeRFState, Dict[str, jax.Array | float]]:
    # TODO:
    #   merge this and `models.renderers.make_rays_worldspace` as a single function
    def make_rays_worldspace() -> Tuple[jax.Array, jax.Array]:
        # [N, 1]
        d_cam_xs = jnp.mod(perm, scene.meta.camera.W)
        d_cam_xs = ((d_cam_xs + 0.5) - scene.meta.camera.cx) / scene.meta.camera.fx
        # [N, 1]
        d_cam_ys = jnp.mod(jnp.floor_divide(perm, scene.meta.camera.W), scene.meta.camera.H)
        d_cam_ys = -((d_cam_ys + 0.5) - scene.meta.camera.cy) / scene.meta.camera.fy
        # [N, 1]
        d_cam_zs = -jnp.ones_like(perm)
        # [N, 3]
        d_cam = jnp.stack([d_cam_xs, d_cam_ys, d_cam_zs]).T
        d_cam /= jnp.linalg.norm(d_cam, axis=-1, keepdims=True) + 1e-15

        # indices of views, used to retrieve transformation information for each ray
        view_idcs = perm // scene.meta.camera.n_pixels
        # [N, 3]
        o_world = scene.all_transforms[view_idcs, -3:]  # WARN: using `perm` instead of `view_idcs` here
                                                  # will silently clip the out-of-bounds indices.
                                                  # REF:
                                                  #   <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#out-of-bounds-indexing>
        # [N, 3, 3]
        R_cws = scene.all_transforms[view_idcs, :9].reshape(-1, 3, 3)
        # [N, 3]
        # equavalent to performing `d_cam[i] @ R_cws[i].T` for each i in [0, N)
        d_world = (d_cam[:, None, :] * R_cws).sum(-1)

        # d_cam was normalized already, normalize d_world just to be sure
        d_world /= jnp.linalg.norm(d_world, axis=-1, keepdims=True) + 1e-15

        o_world += scene.meta.camera.near * d_world

        return o_world, d_world

    def loss_fn(params, gt_rgba_f32, KEY):
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
        batch_metrics, pred_rgbds = render_rays_train(
            KEY=key,
            o_world=o_world,
            d_world=d_world,
            bg=bg,
            total_samples=total_samples,
            state=state.replace(params=params),
        )
        pred_rgbs, pred_depths = jnp.array_split(pred_rgbds, [3], axis=-1)
        gt_rgbs = data.blend_rgba_image_array(imgarr=gt_rgba_f32, bg=bg)
        # from NVLabs/instant-ngp/commit/d6c7241de9be5be1b6d85fe43e446d2eb042511b:
        #   Note: we divide the huber loss by a factor of 5 such that its L2 region near zero
        #   matches with the L2 loss and error numbers become more comparable. This allows reading
        #   off dB numbers of ~converged models and treating them as approximate PSNR to compare
        #   with other NeRF methods. Self-normalizing optimizers such as Adam are agnostic to such
        #   constant factors; optimization is therefore unaffected.
        # Multiplying by 2 here to match the loss scale of ~converged model as in
        # NVLabs/instant-ngp.
        loss = optax.huber_loss(pred_rgbs, gt_rgbs, delta=0.1).mean() * 2
        return loss, batch_metrics

    loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    KEY, key = jran.split(KEY, 2)
    (loss, batch_metrics), grads = loss_grad_fn(
        state.params,
        scene.all_rgbas_u8[perm].astype(jnp.float32) / 255,
        key,
    )
    state = state.apply_gradients(grads=grads)
    metrics = {
        "loss": loss * perm.shape[0],
        **batch_metrics,
    }
    return state, metrics

