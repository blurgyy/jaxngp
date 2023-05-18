from typing import List
from typing_extensions import assert_never

from PIL import Image
from flax.training import checkpoints
import jax
import jax.random as jran
import jax.numpy as jnp
import numpy as np

from models.nerfs import make_nerf_ngp, make_skysphere_background_model_ngp
from models.renderers import render_image_inference
from utils import common, data
from utils.args import NeRFTestingArgs
from utils.types import NeRFState, RenderedImage, RigidTransformation


def test(KEY: jran.KeyArray, args: NeRFTestingArgs, logger: common.Logger):
    logs_dir = args.exp_dir.joinpath("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = common.setup_logging(
        "nerf.test",
        file=logs_dir.joinpath("test.log"),
        level=args.common.logging.upper(),
        file_level="DEBUG",
    )
    if not args.ckpt.exists():
        logger.error("specified checkpoint '{}' does not exist".format(args.ckpt))
        exit(1)

    if args.trajectory == "orbit":
        logger.debug("generating {} testing frames".format(args.orbit.n_frames))
        scene_meta = data.load_scene(
            srcs=args.frames,
            scene_options=args.scene,
            orbit_options=args.orbit,
        )
        logger.info("generated {} camera transforms for testing".format(len(scene_meta.frames)))
    elif args.trajectory == "loaded":
        logger.debug("loading testing frames from {}".format(args.frames))
        scene_data, test_views = data.load_scene(
            srcs=args.frames,
            scene_options=args.scene,
            sort_frames=args.sort_frames,
        )
        scene_meta = scene_data.meta
        logger.info("loaded {} camera transforms for testing".format(len(scene_meta.frames)))

    if args.camera_override.enabled:
        scene_meta = scene_meta.replace(camera=args.camera_override.camera)

    # load parameters
    logger.debug("loading checkpoint from '{}'".format(args.ckpt))
    state: NeRFState = checkpoints.restore_checkpoint(
        args.ckpt,
        target=NeRFState.empty(
            raymarch=args.raymarch,
            render=args.render,
            scene_options=args.scene,
            scene_meta=scene_meta,
            nerf_fn=make_nerf_ngp(bound=scene_meta.bound).apply,
            bg_fn=make_skysphere_background_model_ngp(bound=scene_meta.bound).apply if scene_meta.bg else None,
        ),
    )
    # WARN:
    #   flax.checkpoints.restore_checkpoint() returns a pytree with all arrays of numpy's array type,
    #   which slows down inference.  use jax.device_put() to move them to jax's default device.
    # REF: <https://github.com/google/flax/discussions/1199#discussioncomment-635132>
    state = jax.device_put(state)
    logger.info("checkpoint loaded from '{}'".format(args.ckpt))

    rendered_images: List[RenderedImage] = []
    try:
        n_frames = len(scene_meta.frames)
        logger.info("starting testing (totally {} transform(s) to test)".format(n_frames))
        for test_i in common.tqdm(range(n_frames), desc="testing (resolultion: {}x{})".format(scene_meta.camera.W, scene_meta.camera.H)):
            logger.debug("testing on frame {}".format(scene_meta.frames[test_i]))
            transform = RigidTransformation(
                rotation=scene_meta.frames[test_i].transform_matrix_jax_array[:3, :3],
                translation=scene_meta.frames[test_i].transform_matrix_jax_array[:3, 3],
            )
            KEY, key = jran.split(KEY, 2)
            bg, rgb, depth = render_image_inference(
                KEY=key,
                transform_cw=transform,
                state=state,
            )
            rendered_images.append(RenderedImage(
                bg=data.to_cpu(bg),
                rgb=data.to_cpu(rgb),
                depth=data.to_cpu(depth),
            ))
    except KeyboardInterrupt:
        logger.warn("keyboard interrupt, tested {} images".format(len(rendered_images)))

    if args.trajectory == "loaded":
        if args.camera_override.enabled:
            logger.info("camera is overridden, not calculating psnr")
        else:
            gt_rgbs_f32 = map(
                lambda test_view, rendered_image: data.blend_rgba_image_array(
                    test_view.image_rgba_u8.astype(jnp.float32) / 255,
                    rendered_image.bg,
                ),
                test_views,
                rendered_images,
            )
            logger.debug("calculating psnr")
            mean_psnr = sum(map(
                data.psnr,
                map(data.f32_to_u8, gt_rgbs_f32),
                map(lambda ri: ri.rgb, rendered_images),
            )) / len(rendered_images)
            logger.info("tested {} images, mean psnr={}".format(len(rendered_images), mean_psnr))

    elif args.trajectory == "orbit":
        logger.debug("using generated orbiting trajectory, not calculating psnr")

    else:
        assert_never("")

    save_dest = args.exp_dir.joinpath("test")
    save_dest.mkdir(parents=True, exist_ok=True)

    if "video" in args.save_as:
        dest_rgb_video = save_dest.joinpath("rgb.mp4")
        dest_depth_video = save_dest.joinpath("depth.mp4")

        logger.debug("saving predicted color images as a video at '{}'".format(dest_rgb_video))
        data.write_video(
            save_dest.joinpath("rgb.mp4"),
            map(lambda img: img.rgb, rendered_images),
        )

        logger.debug("saving predicted depths as a video at '{}'".format(dest_depth_video))
        data.write_video(
            save_dest.joinpath("depth.mp4"),
            map(lambda img: img.depth, rendered_images),
        )

    if "image" in args.save_as:
        dest_rgb = save_dest.joinpath("rgb")
        dest_depth = save_dest.joinpath("depth")

        dest_rgb.mkdir(parents=True, exist_ok=True)
        dest_depth.mkdir(parents=True, exist_ok=True)

        logger.debug("saving as images")
        for save_i, img in enumerate(common.tqdm(rendered_images, desc="saving images")):
            Image.fromarray(np.asarray(img.rgb)).save(dest_rgb.joinpath("{:03d}.png".format(save_i)))
            Image.fromarray(np.asarray(img.depth)).save(dest_depth.joinpath("{:03d}.png".format(save_i)))
