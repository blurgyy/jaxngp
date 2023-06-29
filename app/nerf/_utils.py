from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as jran
import optax

from models.renderers import render_rays_train
from utils import common, data
from utils.types import NeRFState, SceneData


__all__ = [
    "make_optimizer",
    "train_step",
]


def make_optimizer(lr: float) -> optax.GradientTransformation:
    lr_sch = optax.exponential_decay(
        init_value=lr,
        transition_steps=10_000,
        decay_rate=1/3,  # decay to `1/3 * init_lr` after `transition_steps` steps
        staircase=True,  # use integer division to determine lr drop step
        transition_begin=10_000,  # hold the initial lr value for the initial 10k steps (but first lr drop happens at 20k steps because `staircase` is specified)
        end_value=lr / 100,  # stop decaying at `1/100 * init_lr`
    )
    optimizer_network = optax.adam(
        learning_rate=lr_sch,
        b1=0.9,
        b2=0.99,
        # paper:
        #   the small value of 𝜖 = 10^{−15} can significantly accelerate the convergence of the
        #   hash table entries when their gradients are sparse and weak.
        eps=1e-15,
        eps_root=1e-15,
    )
    optimizer_ae = optax.adam(
        learning_rate=1e-4,
        b1=.9,
        b2=.99,
        eps=1e-8,
        eps_root=0,
    )
    return optax.chain(
        optax.multi_transform(
            transforms={
                "network": optimizer_network,
                "ae": optimizer_ae,
            },
            param_labels={
                "nerf": "network",
                "bg": "network",
                "appearance_embeddings": "ae",
            },
        ),
        optax.add_decayed_weights(
            # In NeRF experiments, the network can converge to a reasonably low loss during the
            # first ~50k training steps (with 1024 rays per batch and 1024 samples per ray), but the
            # loss becomes NaN after about 50~150k training steps.
            # paper:
            #   To prevent divergence after long training periods, we apply a weak L2 regularization
            #   (factor 10^{−6}) to the neural network weights, ...
            weight_decay=1e-6,
            # paper:
            #   ... to the neural network weights, but not to the hash table entries.
            mask={
                "nerf": {
                    "density_mlp": True,
                    "rgb_mlp": True,
                    "position_encoder": False,
                },
                "bg": True,
                "appearance_embeddings": False,
            },
        ),
    )


@common.jit_jaxfn_with(static_argnames=["total_samples"])
def train_step(
    KEY: jran.KeyArray,
    state: NeRFState,
    total_samples: int,
    scene: SceneData,
    perm: jax.Array,
) -> Tuple[NeRFState, Dict[str, jax.Array | float]]:
    # indices of views and pixels
    view_idcs, pixel_idcs = scene.view_indices[perm], scene.pixel_indices[perm]

    # TODO:
    #   merge this and `models.renderers.make_rays_worldspace` as a single function
    def make_rays_worldspace() -> Tuple[jax.Array, jax.Array]:
        # [N], [N]
        x, y = (
            jnp.mod(pixel_idcs, scene.meta.camera.width),
            jnp.floor_divide(pixel_idcs, scene.meta.camera.width),
        )
        # [N, 3]
        d_cam = scene.meta.camera.make_ray_directions_from_pixel_coordinates(x, y, use_pixel_center=True)

        # [N, 3]
        o_world = scene.transforms[view_idcs, -3:]

        # [N, 3, 3]
        R_cws = scene.transforms[view_idcs, :9].reshape(-1, 3, 3)
        # [N, 3]
        # equavalent to performing `d_cam[i] @ R_cws[i].T` for each i in [0, N)
        d_world = (d_cam[:, None, :] * R_cws).sum(-1)

        return o_world, d_world

    # CAVEAT: gradient is only calculate w.r.t. the first parameter of this function
    # (`params_to_optimize here`), any parameters that need to be optimized should be taken from
    # this parameter, instead from the outer-scope `state.params`.
    def loss_fn(params_to_optimize, gt_rgba_f32, KEY):
        o_world, d_world = make_rays_worldspace()
        appearance_embeddings = (
            params_to_optimize["appearance_embeddings"][view_idcs]
                if "appearance_embeddings" in params_to_optimize
                else jnp.empty(0)
        )
        if state.use_background_model:
            bg = state.bg_fn(
                {"params": params_to_optimize["bg"]},
                o_world,
                d_world,
                appearance_embeddings,
            )
        elif state.render.random_bg:
            KEY, key = jran.split(KEY, 2)
            bg = jran.uniform(key, shape=(o_world.shape[0], 3), dtype=jnp.float32, minval=0, maxval=1)
        else:
            bg = jnp.asarray(state.render.bg)
        KEY, key = jran.split(KEY, 2)
        batch_metrics, pred_rgbds, tv = render_rays_train(
            KEY=key,
            o_world=o_world,
            d_world=d_world,
            appearance_embeddings=appearance_embeddings,
            bg=bg,
            total_samples=total_samples,
            state=state.replace(params=params_to_optimize),
        )
        pred_rgbs, pred_depths = jnp.array_split(pred_rgbds, [3], axis=-1)
        def get_valid_rgbs(rgbs):
            return jnp.where(
                jnp.arange(rgbs.shape[0])[:, None] < batch_metrics["n_valid_rays"],
                rgbs,
                0.
            )
        gt_rgbs = data.blend_rgba_image_array(imgarr=gt_rgba_f32, bg=bg)
        batch_metrics["loss"] = {
            # Scale huber loss by 2 here to match the loss scale of ~converged model so that the dB
            # number approximates PSNR.
            "rgb": optax.huber_loss(get_valid_rgbs(pred_rgbs), get_valid_rgbs(gt_rgbs), delta=0.1).mean() * 2,
            "total_variation": tv,
        }
        loss = jax.tree_util.tree_reduce(lambda x, y: x + y, batch_metrics["loss"])
        return loss, batch_metrics

    loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    KEY, key = jran.split(KEY, 2)
    (_, batch_metrics), grads = loss_grad_fn(
        state.params,
        scene.rgbas_u8[perm].astype(jnp.float32) / 255,
        key,
    )
    state = state.apply_gradients(grads=grads)
    return state, batch_metrics
