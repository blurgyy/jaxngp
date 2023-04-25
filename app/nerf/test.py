import logging
from pathlib import Path
from typing import List

from PIL import Image
from flax.struct import dataclass
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
from utils.types import NeRFBatchConfig, OccupancyDensityGrid, RGBColor, RigidTransformation


@dataclass
class TestedImage:
    rgb: Image.Image
    depth: Image.Image
    gt_path: Path

    def gt_image(self, bg: RGBColor) -> Image.Image:
        return data.blend_rgba_image_array(Image.open(self.gt_path), bg=bg)

    def psnr(self, bg: RGBColor) -> float:
        return data.psnr(self.gt_image(bg), self.rgb)

    def save_rgb(self, dest: Path):
        Image.fromarray(np.asarray(self.rgb)).save(dest)
    def save_depth(self, dest: Path):
        Image.fromarray(np.asarray(self.depth)).save(dest)
    def save_comparison(self, dest: Path, bg: RGBColor):
        gt_image = self.gt_image(bg)
        comparison_image_data = data.side_by_side(
            gt_image,
            self.rgb,
            H=gt_image.shape[0],
            W=gt_image.shape[1]
        )
        Image.fromarray(np.asarray(comparison_image_data)).save(dest)


def test(KEY: jran.KeyArray, args: NeRFTestingArgs, logger: logging.Logger):
    logs_dir = args.exp_dir.joinpath("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = common.setup_logging("nerf.test", file=logs_dir.joinpath("test.log"))
    if not args.test_ckpt.exists():
        logger.warn("specified checkpoint '{}' does not exist".format(args.test_ckpt))
        exit(1)

    if len(args.test_indices) == 0:
        logger.warn("got empty test indices, you might want to specify some image indices via --test-indices")
        logger.warn("proceeding anyway ...")

    dtype = getattr(jnp, "float{}".format(args.common.prec))
    logger.setLevel(args.common.logging.upper())

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

    tested_images: List[TestedImage] = []
    logger.info("starting testing (totally {} image(s) to test)".format(len(args.test_indices)))
    for test_i in tqdm(args.test_indices, desc="Testing", bar_format=common.tqdm_format):
        if test_i < 0 or test_i >= len(test_views):
            logger.warn("skipping out-of-bounds index {} (index should be in range [0, {}])".format(test_i, len(args.test_indices) - 1))
        logger.debug("testing on image index {}".format(test_i))
        transform = RigidTransformation(
            rotation=scene_metadata_test.all_transforms[test_i, :9].reshape(3, 3),
            translation=scene_metadata_test.all_transforms[test_i, -3:].reshape(3),
        )
        KEY, key = jran.split(KEY, 2)
        rgb, depth = render_image(
            KEY=key,
            bound=args.scene.bound,
            camera=scene_metadata_test.camera,
            transform_cw=transform,
            options=args.render,
            raymarch_options=args.raymarch,
            batch_config=batch_config,
            ogrid=ogrid,
            param_dict={"params": params},
            nerf_fn=model.apply,
        )
        tested_images.append(TestedImage(
            rgb=rgb,
            depth=depth,
            gt_path=test_views[test_i].file,
        ))

    # calculate psnr
    logger.debug("calculating psnr")
    mean_psnr = sum(map(lambda timg: timg.psnr(args.render.bg), tqdm(tested_images, desc="calculating psnr", bar_format=common.tqdm_format))) / len(tested_images)
    logger.info("tested {} images, mean psnr={}".format(len(tested_images), mean_psnr))

    # save images
    for save_i, timg in enumerate(tqdm(tested_images, desc="saving images", bar_format=common.tqdm_format)):
        dest = args.exp_dir.joinpath(args.test_split)
        dest.mkdir(parents=True, exist_ok=True)

        dest_rgb = dest.joinpath("{:03d}-rgb.png".format(save_i))
        dest_depth = dest.joinpath("{:03d}-depth.png".format(save_i))
        dest_comparison = dest.joinpath("{:03d}-comparison.png".format(save_i))

        timg.save_rgb(dest_rgb)
        timg.save_depth(dest_depth)
        timg.save_comparison(dest_comparison, bg=args.render.bg)

    return mean_psnr
