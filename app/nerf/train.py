import logging
from typing import Tuple

from PIL import Image
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import jax.random as jran
import numpy as np
import optax
from tqdm import tqdm
import tyro

from models.nerfs import make_nerf_ngp
from models.renderers import render_image, render_rays, update_ogrid
from utils import common, data
from utils.args import NeRFTrainingArgs
from utils.types import (
    NeRFBatchConfig,
    NeRFTrainState,
    OccupancyDensityGrid,
    PinholeCamera,
    RayMarchingOptions,
    RenderingOptions,
    RigidTransformation,
)


@common.jit_jaxfn_with(static_argnames=["bound", "total_samples", "raymarch_options", "render_options"])
def train_step(
        KEY: jran.KeyArray,
        state: NeRFTrainState,
        bound: float,
        total_samples: int,
        camera: PinholeCamera,
        raymarch_options: RayMarchingOptions,
        render_options: RenderingOptions,
        all_xys: jax.Array,
        all_rgbs: jax.Array,
        all_transforms: jax.Array,
        perm: jax.Array,
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

    def loss_fn(params, gt, KEY):
        o_world, d_world = make_rays_worldspace()
        if render_options.random_bg:
            KEY, key = jran.split(KEY, 2)
            bg = jran.uniform(key, shape=(o_world.shape[0], 3), dtype=jnp.float32, minval=0, maxval=1)
        else:
            bg = jnp.asarray(render_options.bg)
        KEY, key = jran.split(KEY, 2)
        batch_metrics, _, pred_rgbs, _ = render_rays(
            KEY=key,
            o_world=o_world,
            d_world=d_world,
            bg=bg,
            bound=bound,
            total_samples=total_samples,
            ogrid=state.ogrid,
            options=raymarch_options,
            param_dict={"params": params},
            nerf_fn=state.apply_fn,
        )
        gt_rgbs = data.blend_rgba_image_array(imgarr=gt, bg=bg)
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
    (loss, batch_metrics), grads = loss_grad_fn(state.params, all_rgbs[perm], key)
    state = state.apply_gradients(grads=grads)
    metrics = {
        "loss": loss * perm.shape[0],
        **batch_metrics,
    }
    return state, metrics


def train_epoch(
        KEY: jran.KeyArray,
        bound: float,
        scene_metadata: data.SceneMetadata,
        raymarch_options: RayMarchingOptions,
        render_options: RenderingOptions,
        state: NeRFTrainState,
        permutation: jax.Array,
        n_batches: int,
        total_samples: int,
        ep_log: int,
        total_epochs: int,
    ):
    n_processed_rays = 0
    loss, running_loss = 0, -1
    running_mean_samp_per_ray = state.batch_config.mean_samples_per_ray

    beg_idx = 0
    for _ in (pbar := tqdm(range(n_batches), desc="Training epoch#{:03d}/{:d}".format(ep_log, total_epochs), bar_format=common.tqdm_format)):
        if beg_idx >= len(permutation):
            pbar.close()
            break
        KEY, key = jran.split(KEY, 2)
        perm = permutation[beg_idx:beg_idx+state.batch_config.n_rays]
        beg_idx += state.batch_config.n_rays
        state, metrics = train_step(
            KEY=key,
            state=state,
            bound=bound,
            total_samples=total_samples,
            camera=scene_metadata.camera,
            raymarch_options=raymarch_options,
            render_options=render_options,
            all_xys=scene_metadata.all_xys,
            all_rgbs=scene_metadata.all_rgbs,
            all_transforms=scene_metadata.all_transforms,
            perm=perm,
        )
        n_processed_rays += state.batch_config.n_rays
        loss += metrics["loss"]
        loss_log = metrics["loss"] / state.batch_config.n_rays
        if running_loss < 0:
            running_loss = loss_log
        else:
            running_loss = running_loss * 0.99 + 0.01 * loss_log
        running_mean_samp_per_ray = running_mean_samp_per_ray * .95 + .05 * metrics["measured_batch_size_before_compaction"] / state.batch_config.n_rays

        pbar.set_description_str(
            desc="Training epoch#{:03d}/{:d} batch_size={}/{} n_rays={} samp./ray={} loss={:.3e} psnr~{:.2f}dB".format(
                ep_log,
                total_epochs,
                metrics["measured_batch_size"],
                metrics["measured_batch_size_before_compaction"],
                state.batch_config.n_rays,
                state.batch_config.mean_samples_per_ray,
                running_loss,
                data.linear2psnr(running_loss, maxval=1)
            )
        )

        if state.should_call_update_ogrid:
            # update occupancy grid
            KEY, key = jran.split(KEY, 2)
            state = update_ogrid(
                KEY=key,
                update_all=state.should_update_all_ogrid_cells,
                bound=bound,
                raymarch=raymarch_options,
                state=state,
            )

        if state.should_update_batch_config:
            new_mean_samples_per_ray = int(running_mean_samp_per_ray + 1.5)
            new_n_rays = total_samples // new_mean_samples_per_ray
            state = state.replace(
                batch_config=NeRFBatchConfig(
                    mean_samples_per_ray=new_mean_samples_per_ray,
                    n_rays=new_n_rays,
                ),
            )

    return loss / n_processed_rays, state


def train(KEY: jran.KeyArray, args: NeRFTrainingArgs, logger: logging.Logger):
    if args.exp_dir.exists():
        logger.error("specified experiment directory '{}' already exists".format(args.exp_dir))
        exit(2)
    args.exp_dir.mkdir(parents=True)
    args.exp_dir.joinpath("config.yaml").write_text(tyro.to_yaml(args))
    logger.info("configurations saved to '{}'".format(args.exp_dir.joinpath("config.yaml")))

    dtype = getattr(jnp, "float{}".format(args.common.prec))
    logger.setLevel(args.common.logging.upper())

    # model parameters
    model, init_input = (
        make_nerf_ngp(bound=args.bound),
        (jnp.zeros((1, 3), dtype=dtype), jnp.zeros((1, 3), dtype=dtype))
    )
    KEY, key = jran.split(KEY, 2)
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
    state = NeRFTrainState.create(
        apply_fn=model.apply,
        # unfreeze the frozen dict so that below weight_decay mask can apply, see:
        #   <https://github.com/deepmind/optax/issues/160>
        #   <https://github.com/google/flax/issues/1223>
        params=variables["params"].unfreeze(),
        tx=optimizer,
        ogrid=OccupancyDensityGrid.create(
            cascades=data.cascades_from_bound(args.bound),
            grid_resolution=args.raymarch.density_grid_res,
        ),
        batch_config=NeRFBatchConfig(
            mean_samples_per_ray=args.raymarch.diagonal_n_steps,
            n_rays=args.train.bs // args.raymarch.diagonal_n_steps,
        ),
    )

    # data
    scene_metadata_train, _ = data.make_nerf_synthetic_scene_metadata(
        rootdir=args.data_root,
        split="train",
        scale=args.scale,
    )

    scene_metadata_val, val_views = data.make_nerf_synthetic_scene_metadata(
        rootdir=args.data_root,
        split="val",
        scale=args.scale,
    )

    logger.info("starting training")
    # training loop
    for ep in range(args.train.n_epochs):
        ep_log = ep + 1
        KEY, key = jran.split(KEY, 2)
        permutation = data.make_permutation(
            key,
            size=scene_metadata_train.all_xys.shape[0],
            loop=args.train.data_loop,
            shuffle=True
        )

        try:
            KEY, key = jran.split(KEY, 2)
            loss_log, state = train_epoch(
                KEY=key,
                bound=args.bound,
                scene_metadata=scene_metadata_train,
                raymarch_options=args.raymarch,
                render_options=args.render,
                state=state,
                permutation=permutation,
                n_batches=args.train.n_batches,
                total_samples=args.train.bs,
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

        logger.info("epoch#{:03d}: loss={:.2e} psnr~{:.2f}dB".format(ep_log, loss_log, data.linear2psnr(loss_log, maxval=1)))

        logger.info("saving training state ... ")
        ckpt_name = checkpoints.save_checkpoint(
            args.exp_dir,
            state,
            step=ep_log * args.train.n_batches,
            overwrite=True,
            keep=args.train.keep,
            keep_every_n_steps=args.train.keep_every_n_steps,
        )
        logger.info("training state of epoch {} saved to: {}".format(ep_log, ckpt_name))

        # validate on `args.val_num` random camera transforms
        KEY, key = jran.split(KEY, 2)
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
            KEY, key = jran.split(KEY, 2)
            rgb, depth = render_image(
                KEY=key,
                bound=args.bound,
                camera=scene_metadata_val.camera,
                transform_cw=val_transform,
                options=args.render_eval,
                raymarch_options=args.raymarch_eval,
                batch_config=state.batch_config,
                ogrid=state.ogrid,
                param_dict={"params": state.params},
                nerf_fn=state.apply_fn,
            )
            gt_image = Image.open(val_views[val_i].file)
            gt_image = np.asarray(gt_image)
            gt_image = data.blend_rgba_image_array(gt_image, bg=args.render_eval.bg)
            logger.info("{}: psnr={:.4f}dB".format(val_views[val_i].file, data.psnr(gt_image, rgb)))
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
