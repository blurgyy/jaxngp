import logging
from typing import Tuple

from PIL import Image
from flax.training import checkpoints
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import jax.random as jran
import numpy as np
import optax
from tqdm import tqdm
import tyro

from models.nerfs import make_nerf_ngp
from models.renderers import march_rays, render_image
from utils import common, data
from utils.args import NeRFTrainingArgs
from utils.types import (
    AABB,
    PinholeCamera,
    RayMarchingOptions,
    RenderingOptions,
    RigidTransformation,
)


@common.jit_jaxfn_with(static_argnames=["raymarch_options", "render_options"])
def train_step(
        K: jran.KeyArray,
        state: TrainState,
        aabb: AABB,
        camera: PinholeCamera,
        all_xys: jax.Array,
        all_rgbs: jax.Array,
        all_transforms: jax.Array,
        raymarch_options: RayMarchingOptions,
        render_options: RenderingOptions,
        perm: jax.Array
    ):
    # TODO:
    #   merge this and `models.renderers.make_rays_worldspace` as a single function
    def make_rays_worldspace() -> Tuple[jax.Array, jax.Array]:
        # [N, 2]
        xys = all_xys[perm]
        # [N, 3]
        xyzs = jnp.concatenate([xys, jnp.ones((xys.shape[0], 1))], axis=-1)
        # [N, 1]
        d_cam_xs = xyzs[:, 0:1]
        d_cam_xs = ((d_cam_xs + 0.5) - camera.W/2)
        # [N, 1]
        d_cam_ys = xyzs[:, 1:2]
        d_cam_ys = -((d_cam_ys + 0.5) - camera.H/2)
        # [N, 1]
        d_cam_zs = -camera.focal * xyzs[:, 2:3]
        # [N, 3]
        d_cam = jnp.concatenate([d_cam_xs, d_cam_ys, d_cam_zs], axis=-1)
        d_cam /= jnp.linalg.norm(d_cam, axis=-1, keepdims=True)

        # indices of views, used to retrieve transformation information for each ray
        view_idcs = perm // (camera.H * camera.W)
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
        d_world /= jnp.linalg.norm(d_world, axis=-1, keepdims=True)

        return o_world, d_world

    def loss(params, gt, K):
        o_world, d_world = make_rays_worldspace()
        K, key = jran.split(K, 2)
        weights, preds, _ = march_rays(
            key,
            o_world,
            d_world,
            aabb,
            raymarch_options,
            {"params": params},
            state.apply_fn,
        )
        if render_options.random_bg:
            K, key = jran.split(K, 2)
            bg = jran.uniform(key, preds.shape, dtype=preds.dtype, minval=0, maxval=1)
        else:
            bg = render_options.bg
        pred_rgbs = data.blend_alpha_channel(
            imgarr=jnp.concatenate([preds, weights.sum(axis=-1, keepdims=True)], axis=-1),
            bg=bg,
        )
        gt_rgbs = data.blend_alpha_channel(imgarr=gt, bg=bg)
        loss = jnp.square(pred_rgbs - gt_rgbs).mean()
        return loss

    loss_grad_fn = jax.value_and_grad(loss)

    K, key = jran.split(K, 2)
    loss, grads = loss_grad_fn(state.params, all_rgbs[perm], key)
    state = state.apply_gradients(grads=grads)
    metrics = {
        "loss": loss * perm.shape[0],
    }
    return state, metrics


def train_epoch(
        K: jran.KeyArray,
        aabb: AABB,
        scene_metadata: data.SceneMetadata,
        raymarch_options: RayMarchingOptions,
        render_options: RenderingOptions,
        permutation: data.Dataset,
        state: TrainState,
        total_batches: int,
        ep_log: int,
        total_epochs: int,
    ):
    loss, running_loss = 0, -1
    for perm in (pbar := tqdm(permutation, total=total_batches, desc="Training epoch#{:03d}/{:d}".format(ep_log, total_epochs), bar_format=common.tqdm_format)):
        K, key = jran.split(K)
        state, metrics = train_step(
            key,
            state,
            aabb,
            scene_metadata.camera,
            scene_metadata.all_xys,
            scene_metadata.all_rgbs,
            scene_metadata.all_transforms,
            raymarch_options,
            render_options,
            perm,
        )
        loss += metrics["loss"]
        loss_log = metrics["loss"] / perm.shape[0]
        if running_loss < 0:
            running_loss = loss_log
        else:
            running_loss = running_loss * 0.99 + 0.01 * loss_log
        pbar.set_description_str(
            desc="Training epoch#{:03d}/{:d} mse={:.3e} psnr={:.2f}".format(
                ep_log,
                total_epochs,
                running_loss,
                data.loss2psnr(running_loss, maxval=1)
            )
        )
    return loss, state


def train(args: NeRFTrainingArgs, logger: logging.Logger):
    if args.exp_dir.exists():
        logger.error("specified experiment directory '{}' already exists".format(args.exp_dir))
        exit(2)
    args.exp_dir.mkdir(parents=True)
    args.exp_dir.joinpath("config.yaml").write_text(tyro.to_yaml(args))
    logger.info("configurations saved to '{}'".format(args.exp_dir.joinpath("config.yaml")))

    dtype = getattr(jnp, "float{}".format(args.common.prec))
    logger.setLevel(args.common.logging.upper())

    # deterministic
    K = common.set_deterministic(args.common.seed)

    # model parameters
    aabb = [[-args.bound, args.bound]] * 3
    model, init_input = (
        make_nerf_ngp(aabb=aabb),
        (jnp.zeros((1, 3), dtype=dtype), jnp.zeros((1, 3), dtype=dtype))
    )
    K, key = jran.split(K, 2)
    variables = model.init(key, *init_input)
    if args.common.display_model_summary:
        print(model.tabulate(key, *init_input))

    lr_sch = optax.exponential_decay(
        init_value=args.train.lr,
        transition_steps=10_000,
        decay_rate=1/3,  # decay to `1/3 * init_lr` after `transition_steps` steps
        staircase=True,  # use integer division to determine lr drop step
        transition_begin=10_000,  # hold the initial lr value for the initial 10k steps (but first lr drop happens at 20k steps because `staircase` is specified)
        end_value=args.train.lr / 100,  # stop decaying at `1/100 * init_lr`
    )
    optimizer = optax.adamw(
        learning_rate=lr_sch,
        b1=0.9,
        b2=0.99,
        # paper:
        #   the small value of 𝜖 = 10^{−15} can significantly accelerate the convergence of the
        #   hash table entries when their gradients are sparse and weak.
        eps=1e-15,
        eps_root=1e-15,
        # In NeRF experiments, the network can converge to a reasonably low loss during the
        # frist ~50k training steps (with 1024 rays per batch and 1024 samples per ray), but the
        # loss becomes NaN after about 50~150k training steps.
        # paper:
        #   To prevent divergence after long training periods, we apply a weak L2 regularization
        #   (factor 10^{−6}) to the neural network weights, ...
        weight_decay=1e-6,
        # paper:
        #   ... to the neural network weights, but not to the hash table entries.
        mask={
            "density_mlp": True,
            "rgb_mlp": True,
            "position_encoder": False,
        },
    )

    # training state
    state = TrainState.create(
        apply_fn=model.apply,
        # unfreeze the frozen dict so that below weight_decay mask can apply, see:
        #   <https://github.com/deepmind/optax/issues/160>
        #   <https://github.com/google/flax/issues/1223>
        params=variables["params"].unfreeze(),
        tx=optimizer,
    )

    # data
    scene_metadata_train, _ = data.make_nerf_synthetic_scene_metadata(
        rootdir=args.data_root,
        split="train",
    )

    scene_metadata_val, val_views = data.make_nerf_synthetic_scene_metadata(
        rootdir=args.data_root,
        split="val",
    )

    logger.info("starting training")
    # training loop
    for ep in range(args.train.n_epochs):
        ep_log = ep + 1
        K, key = jran.split(K, 2)
        permutation = data.make_permutation_dataset(
            key,
            size=scene_metadata_train.all_xys.shape[0],
            shuffle=True
        )\
            .batch(args.render.ray_chunk_size, drop_remainder=True)\
            .repeat(args.train.data_loop)

        try:
            K, key = jran.split(K, 2)
            loss, state = train_epoch(
                K=key,
                aabb=aabb,
                scene_metadata=scene_metadata_train,
                raymarch_options=args.raymarch,
                render_options=args.render,
                permutation=permutation.take(args.train.n_batches).as_numpy_iterator(),
                state=state,
                total_batches=args.train.n_batches,
                ep_log=ep_log,
                total_epochs=args.train.n_epochs,
            )
        except KeyboardInterrupt:
            logger.warn("aborted at epoch {}".format(ep_log))
            logger.info("saving training state ... ")
            ckpt_name = checkpoints.save_checkpoint(args.exp_dir, state, step="ep{}aborted".format(ep_log), overwrite=True, keep=2**30)
            logger.info("training state of epoch {} saved to: {}".format(ep_log, ckpt_name))
            logger.info("exiting cleanly ...")
            exit()

        loss_log = loss / (args.train.n_batches * args.render.ray_chunk_size)
        logger.info("epoch#{:03d}: mse={:.2e} psnr={:.2f}".format(ep_log, loss_log, data.loss2psnr(loss_log, maxval=1)))

        logger.info("saving training state ... ")
        ckpt_name = checkpoints.save_checkpoint(
            args.exp_dir,
            state,
            step=ep_log * args.train.n_batches,
            overwrite=True,
            keep=2**30,
        )
        logger.info("training state of epoch {} saved to: {}".format(ep_log, ckpt_name))

        # validate on `args.val_num` random camera transforms
        K, key = jran.split(K, 2)
        for val_i in jran.choice(
                key,
                jnp.arange(len(val_views)),
                (min(args.val_num, len(val_views)),),
                replace=False,
            ):
            logger.debug("validating on {}".format(val_views[val_i].file))
            val_transform = RigidTransformation(
                rotation=scene_metadata_val.all_transforms[val_i, :9].reshape(3, 3),
                translation=scene_metadata_val.all_transforms[val_i, -3:].reshape(3),
            )
            K, key = jran.split(K, 2)
            rgb, depth = render_image(
                K=key,
                aabb=aabb,
                camera=scene_metadata_val.camera,
                transform_cw=val_transform,
                options=args.render_eval,
                raymarch_options=args.raymarch_eval,
                param_dict={"params": state.params},
                nerf_fn=state.apply_fn,
            )
            gt_image = Image.open(val_views[val_i].file)
            gt_image = np.asarray(gt_image)
            gt_image = data.blend_alpha_channel(gt_image, bg=args.render_eval.bg)
            logger.info("{}: psnr={}".format(val_views[val_i].file, data.psnr(gt_image, rgb)))
            dest = args.exp_dir\
                .joinpath("validataion")\
                .joinpath("ep{}".format(ep_log))
            dest.mkdir(parents=True, exist_ok=True)

            # rgb
            dest_rgb = dest.joinpath("{:03d}-rgb.png".format(val_i))
            logger.debug("saving predicted rgb image to {}".format(dest_rgb))
            Image.fromarray(np.asarray(rgb)).save(dest_rgb)

            # comparison image
            dest_comparison = dest.joinpath("{:03d}-comparison.png".format(val_i))
            logger.debug("saving comparison image to {}".format(dest_comparison))
            comparison_image_data = data.side_by_side(
                gt_image,
                rgb,
                H=scene_metadata_val.camera.H,
                W=scene_metadata_val.camera.W
            )
            comparison_image_data = data.add_border(comparison_image_data)
            Image.fromarray(np.asarray(comparison_image_data)).save(dest_comparison)

            # depth
            dest_depth = dest.joinpath("{:03d}-depth.png".format(val_i))
            logger.debug("saving predicted depth image to {}".format(dest_depth))
            Image.fromarray(np.asarray(depth)).save(dest_depth)
