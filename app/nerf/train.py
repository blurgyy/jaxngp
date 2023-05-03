import dataclasses
import functools
import gc
import time
from typing import Dict, List, Tuple, Union

from flax.training import checkpoints
import jax
import jax.numpy as jnp
import jax.random as jran
import optax
from tqdm import tqdm
import tyro

from models.nerfs import make_nerf_ngp, make_skysphere_background_model_ngp
from models.renderers import render_image_inference, render_rays_train, update_ogrid
from utils import common, data
from utils.args import NeRFTrainingArgs
from utils.types import (
    NeRFBatchConfig,
    NeRFState,
    OccupancyDensityGrid,
    PinholeCamera,
    RenderedImage,
    RigidTransformation,
    SceneMetadata,
)


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


def train_epoch(
    KEY: jran.KeyArray,
    state: NeRFState,
    scene_metadata: SceneMetadata,
    permutation: jax.Array,
    n_batches: int,
    total_samples: int,
    ep_log: int,
    total_epochs: int,
    logger: common.Logger,
):
    n_processed_rays = 0
    total_loss = 0
    running_mean_effective_samp_per_ray = state.batch_config.mean_effective_samples_per_ray
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
            total_samples=total_samples,
            camera=scene_metadata.camera,
            all_xys=scene_metadata.all_xys,
            all_rgbas=scene_metadata.all_rgbas,
            all_transforms=scene_metadata.all_transforms,
            perm=perm,
        )
        n_processed_rays += state.batch_config.n_rays
        total_loss += metrics["loss"]
        loss_log = metrics["loss"] / state.batch_config.n_rays
        running_mean_samp_per_ray, running_mean_effective_samp_per_ray = (
            running_mean_samp_per_ray * .95 + .05 * metrics["measured_batch_size_before_compaction"] / state.batch_config.n_rays,
            running_mean_effective_samp_per_ray * .95 + .05 * metrics["measured_batch_size"] / state.batch_config.n_rays,
        )

        loss_db = data.linear_to_db(loss_log, maxval=1)
        pbar.set_description_str(
            desc="Training epoch#{:03d}/{:d} batch_size={}/{} samp./ray={}/{} n_rays={} loss={:.3e}({:.2f}dB)".format(
                ep_log,
                total_epochs,
                metrics["measured_batch_size"],
                metrics["measured_batch_size_before_compaction"],
                state.batch_config.mean_effective_samples_per_ray,
                state.batch_config.mean_samples_per_ray,
                state.batch_config.n_rays,
                loss_log,
                loss_db,
            )
        )

        if state.should_call_update_ogrid:
            # update occupancy grid
            KEY, key = jran.split(KEY, 2)
            state = update_ogrid(
                KEY=key,
                update_all=bool(state.should_update_all_ogrid_cells),
                state=state,
            )

        if state.should_update_batch_config:
            new_mean_effective_samples_per_ray = int(running_mean_effective_samp_per_ray + 1.5)
            new_mean_samples_per_ray = int(running_mean_samp_per_ray + 1.5)
            new_n_rays = total_samples // new_mean_samples_per_ray
            state = state.replace(
                batch_config=NeRFBatchConfig(
                    mean_effective_samples_per_ray=new_mean_effective_samples_per_ray,
                    mean_samples_per_ray=new_mean_samples_per_ray,
                    n_rays=new_n_rays,
                ),
            )

        if state.should_write_batch_metrics:
            logger.write_scalar("batch/↓loss", loss_log, state.step)
            logger.write_scalar("batch/↑loss (db)", loss_db, state.step)
            logger.write_scalar("batch/effective batch size (not compacted)", metrics["measured_batch_size_before_compaction"], state.step)
            logger.write_scalar("batch/↑effective batch size (compacted)", metrics["measured_batch_size"], state.step)
            logger.write_scalar("rendering/↓effective samples per ray", state.batch_config.mean_effective_samples_per_ray, state.step)
            logger.write_scalar("rendering/↓marched samples per ray", state.batch_config.mean_samples_per_ray, state.step)
            logger.write_scalar("rendering/↑number of rays", state.batch_config.n_rays, state.step)

    return total_loss / n_processed_rays, state


def train(KEY: jran.KeyArray, args: NeRFTrainingArgs, logger: common.Logger):
    if args.exp_dir.exists():
        logger.error("specified experiment directory '{}' already exists".format(args.exp_dir))
        exit(1)
    logs_dir = args.exp_dir.joinpath("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = common.setup_logging(
        "nerf.train",
        file=logs_dir.joinpath("train.log"),
        with_tensorboard=True,
        level=args.common.logging.upper(),
        file_level="DEBUG",
    )
    args.exp_dir.joinpath("config.yaml").write_text(tyro.to_yaml(args))
    logger.write_hparams(dataclasses.asdict(args))
    logger.info("configurations saved to '{}'".format(args.exp_dir.joinpath("config.yaml")))

    # model parameters
    nerf_model, init_input = (
        make_nerf_ngp(bound=args.scene.bound),
        (jnp.zeros((1, 3), dtype=jnp.float32), jnp.zeros((1, 3), dtype=jnp.float32))
    )
    KEY, key = jran.split(KEY, 2)
    nerf_variables = nerf_model.init(key, *init_input)
    if args.common.summary:
        print(nerf_model.tabulate(key, *init_input))

    if args.scene.with_bg:
        bg_model, init_input = (
            make_skysphere_background_model_ngp(bound=args.scene.bound),
            (jnp.zeros((1, 3), dtype=jnp.float32), jnp.zeros((1, 3), dtype=jnp.float32))
        )
        KEY, key = jran.split(KEY, 2)
        bg_variables = bg_model.init(key, *init_input)

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
            "nerf": {
                "density_mlp": True,
                "rgb_mlp": True,
                "position_encoder": False,
            },
            "bg": args.scene.with_bg,
        },
    )

    # training state
    state = NeRFState.create(
        ogrid=OccupancyDensityGrid.create(
            cascades=data.cascades_from_bound(args.scene.bound),
            grid_resolution=args.raymarch.density_grid_res,
        ),
        batch_config=NeRFBatchConfig(
            mean_effective_samples_per_ray=args.raymarch.diagonal_n_steps,
            mean_samples_per_ray=args.raymarch.diagonal_n_steps,
            n_rays=args.train.bs // args.raymarch.diagonal_n_steps,
        ),
        raymarch=args.raymarch,
        render=args.render,
        scene=args.scene,
        # unfreeze the frozen dict so that the weight_decay mask can apply, see:
        #   <https://github.com/deepmind/optax/issues/160>
        #   <https://github.com/google/flax/issues/1223>
        nerf_fn=nerf_model.apply,
        bg_fn=bg_model.apply if args.scene.with_bg else None,
        params={
            "nerf": nerf_variables["params"].unfreeze(),
            "bg": bg_variables["params"].unfreeze() if args.scene.with_bg else None,
        },
        tx=optimizer,
    )

    # data
    scene_metadata_train, _ = data.make_scene_metadata(
        rootdir=args.data_root,
        split="train",
        world_scale=args.scene.world_scale,
        image_scale=args.scene.image_scale,
    )

    scene_metadata_val, val_views = data.make_scene_metadata(
        rootdir=args.data_root,
        split="val",
        world_scale=args.scene.world_scale,
        image_scale=args.scene.image_scale,
    )

    logger.info("starting training")
    # training loop
    for ep in range(args.train.n_epochs):
        gc.collect()

        ep_log = ep + 1
        KEY, key = jran.split(KEY, 2)
        permutation = data.make_permutation(
            key,
            size=scene_metadata_train.all_xys.shape[0],
            loop=args.train.data_loop,
            shuffle=True,
        )

        try:
            KEY, key = jran.split(KEY, 2)
            loss_log, state = train_epoch(
                KEY=key,
                state=state,
                scene_metadata=scene_metadata_train,
                permutation=permutation,
                n_batches=args.train.n_batches,
                total_samples=args.train.bs,
                ep_log=ep_log,
                total_epochs=args.train.n_epochs,
                logger=logger,
            )
        except KeyboardInterrupt:
            logger.warn("aborted at epoch {}".format(ep_log))
            logger.info("saving training state ... ")
            ckpt_name = checkpoints.save_checkpoint(args.exp_dir, state, step="ep{}aborted".format(ep_log), overwrite=True, keep=2**30)
            logger.info("training state of epoch {} saved to: {}".format(ep_log, ckpt_name))
            logger.info("exiting cleanly ...")
            exit()

        loss_db = data.linear_to_db(loss_log, maxval=1)
        logger.info("epoch#{:03d}: loss={:.2e}({:.2f}dB)".format(ep_log, loss_log, loss_db))
        logger.write_scalar("epoch/↓loss", loss_log, step=ep_log)
        logger.write_scalar("epoch/↑loss (db)", loss_db, step=ep_log)

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

        if ep_log % args.train.validate_interval_epochs == 0:
            val_start_time = time.time()
            rendered_images: List[RenderedImage] = []
            state_eval = state\
                .replace(raymarch=args.raymarch_eval)\
                .replace(render=args.render_eval)
            for val_i, val_view in enumerate(tqdm(val_views, desc="validating", bar_format=common.tqdm_format)):
                logger.debug("validating on {}".format(val_view.file))
                val_transform = RigidTransformation(
                    rotation=scene_metadata_val.all_transforms[val_i, :9].reshape(3, 3),
                    translation=scene_metadata_val.all_transforms[val_i, -3:].reshape(3),
                )
                KEY, key = jran.split(KEY, 2)
                bg, rgb, depth = render_image_inference(
                    KEY=key,
                    camera=scene_metadata_val.camera,
                    transform_cw=val_transform,
                    state=state_eval,
                )
                rendered_images.append(RenderedImage(
                    bg=data.to_cpu(bg),
                    rgb=data.to_cpu(rgb),
                    depth=data.to_cpu(depth),
                ))
            val_end_time = time.time()
            logger.write_scalar(
                tag="validation/↓rendering time (ms) per image",
                value=(val_end_time - val_start_time) / len(rendered_images) * 1000,
                step=ep_log,
            )

            gt_rgbs_f32 = list(map(
                lambda val_view, rendered_image: data.blend_rgba_image_array(
                    val_view.image_rgba,
                    rendered_image.bg,
                ),
                val_views,
                rendered_images,
            ))

            logger.debug("calculating psnr")
            mean_psnr = sum(map(
                data.psnr,
                map(data.f32_to_u8, gt_rgbs_f32),
                map(lambda ri: ri.rgb, rendered_images),
            )) / len(rendered_images)
            logger.info("validated {} images, mean psnr={}".format(len(rendered_images), mean_psnr))
            logger.write_scalar("validation/↑mean psnr", mean_psnr, step=ep_log)

            logger.debug("writing images to tensorboard")
            concatenate_fn = lambda gt, rendered_image: data.add_border(functools.reduce(
                functools.partial(
                    data.side_by_side,
                    H=scene_metadata_val.camera.H,
                    W=scene_metadata_val.camera.W,
                ),
                [gt, rendered_image.rgb, rendered_image.depth],
            ))
            logger.write_image(
                tag="validation/[gt|rendered|depth]",
                image=list(map(
                    concatenate_fn,
                    map(data.f32_to_u8, gt_rgbs_f32),
                    rendered_images,
                )),
                step=ep_log,
                max_outputs=len(rendered_images),
            )

            del state_eval
            del gt_rgbs_f32
            del rendered_images
        del permutation
