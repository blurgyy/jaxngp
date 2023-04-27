from typing import List
from typing_extensions import assert_never

from PIL import Image
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import jax.random as jran
import numpy as np
from tqdm import tqdm

from models.nerfs import make_nerf_ngp
from models.renderers import render_image
from utils import common, data
from utils.args import NeRFTestingArgs
from utils.data import make_nerf_synthetic_scene_metadata
from utils.types import (
    NeRFBatchConfig,
    OccupancyDensityGrid,
    RenderedImage,
    RigidTransformation,
)


def test(KEY: jran.KeyArray, args: NeRFTestingArgs, logger: common.Logger):
    logs_dir = args.exp_dir.joinpath("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = common.setup_logging(
        "nerf.test",
        file=logs_dir.joinpath("test.log"),
        level=args.common.logging,
        file_level="DEBUG",
    )
    if not args.test_ckpt.exists():
        logger.warn("specified checkpoint '{}' does not exist".format(args.test_ckpt))
        exit(1)

    if len(args.test_indices) == 0:
        logger.warn("got empty test indices, you might want to specify some image indices via --test-indices")
        logger.warn("proceeding anyway ...")

    dtype = getattr(jnp, "float{}".format(args.common.prec))

    # model parameters
    KEY, key = jran.split(KEY, 2)
    model, init_input = (
        make_nerf_ngp(bound=args.scene.bound),
        (jnp.zeros((1, 3), dtype=dtype), jnp.zeros((1, 3), dtype=dtype))
    )
    # initialize model structure but discard parameters, as parameters are loaded later
    model.init(key, *init_input)
    if args.common.summary:
        print(model.tabulate(key, *init_input))

    # load parameters
    ckpt = checkpoints.restore_checkpoint(args.test_ckpt, target=None)
    batch_config = NeRFBatchConfig(**ckpt["batch_config"])
    batch_config = batch_config
    ogrid, params = OccupancyDensityGrid(**ckpt["ogrid"]), ckpt["params"]
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), params)
    params = jax.lax.stop_gradient(params)

    scene_metadata_test, test_views = make_nerf_synthetic_scene_metadata(
        rootdir=args.data_root,
        split=args.test_split,
        scale=args.scene.scale,
    )

    rendered_images: List[RenderedImage] = []
    logger.info("starting testing (totally {} image(s) to test)".format(len(args.test_indices)))
    for test_i in tqdm(args.test_indices, desc="Testing", bar_format=common.tqdm_format):
        if test_i < 0 or test_i >= len(test_views):
            logger.warn("skipping out-of-bounds index {} (index should be in range [0, {}])".format(test_i, len(args.test_indices) - 1))
        logger.debug("testing on image index {}".format(test_i))
        transform = RigidTransformation(
            rotation=scene_metadata_test.all_transforms[test_i, :9].reshape(3, 3),
            translation=scene_metadata_test.all_transforms[test_i, -3:].reshape(3),
        )
        if args.render.random_bg:
            KEY, key = jran.split(KEY, 2)
            bg = jran.uniform(key, (3,), dtype=jnp.float32, minval=0, maxval=1)
        else:
            bg = args.render.bg
        rgb, depth = render_image(
            bg=bg,
            bound=args.scene.bound,
            camera=scene_metadata_test.camera,
            transform_cw=transform,
            raymarch_options=args.raymarch,
            batch_config=batch_config,
            ogrid=ogrid,
            param_dict={"params": params},
            nerf_fn=model.apply,
        )
        rendered_images.append(RenderedImage(
            bg=bg,
            rgb=rgb,
            depth=depth,
        ))

    gt_rgbs = map(
        lambda i, test_i: data.blend_rgba_image_array(test_views[i].image_rgba, rendered_images[test_i].bg),
        args.test_indices,
        range(len(args.test_indices)),
    )
    logger.debug("calculating psnr")
    mean_psnr = sum(map(
        jax.jit(data.psnr),
        map(lambda gt_rgb: jnp.clip(gt_rgb * 255, 0, 255).astype(jnp.uint8), gt_rgbs),
        map(lambda ri: ri.rgb, rendered_images),
    )) / len(rendered_images)
    logger.info("tested {} images, mean psnr={}".format(len(rendered_images), mean_psnr))

    save_dest = args.exp_dir.joinpath(args.test_split)
    save_dest.mkdir(parents=True, exist_ok=True)
    if "image" in args.save_as:
        dest_rgb = save_dest.joinpath("rgb")
        dest_depth = save_dest.joinpath("depth")

        dest_rgb.mkdir(parents=True, exist_ok=True)
        dest_depth.mkdir(parents=True, exist_ok=True)

        logger.debug("saving as images")
        for save_i, img in enumerate(tqdm(rendered_images, desc="saving images", bar_format=common.tqdm_format)):
            Image.fromarray(np.asarray(img.rgb)).save(dest_rgb.joinpath("{:03d}.png".format(save_i)))
            Image.fromarray(np.asarray(img.depth)).save(dest_depth.joinpath("{:03d}.png".format(save_i)))

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

    else:
        assert_never()

    return mean_psnr
