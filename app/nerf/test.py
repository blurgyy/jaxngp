import logging

from PIL import Image
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import jax.random as jran
import numpy as np

from models.nerfs import make_nerf_ngp
from models.renderers import render_image
from utils import common, data
from utils.args import NeRFTestingArgs
from utils.data import make_nerf_synthetic_scene_metadata
from utils.types import NeRFBatchConfig, OccupancyDensityGrid, RigidTransformation


def test(args: NeRFTestingArgs, logger: logging.Logger):
    if not args.test_ckpt.exists():
        logger.warn("specified checkpoint '{}' does not exist".format(args.test_ckpt))
        exit(1)

    if len(args.test_indices) == 0:
        logger.warn("got empty test indices, you might want to specify some image indices via --test-indices")
        logger.warn("proceeding anyway ...")

    dtype = getattr(jnp, "float{}".format(args.common.prec))
    logger.setLevel(args.common.logging.upper())

    # deterministic
    KEY = common.set_deterministic(seed=args.common.seed)

    # model parameters
    KEY, key = jran.split(KEY, 2)
    model, init_input = (
        make_nerf_ngp(bound=args.bound),
        (jnp.zeros((1, 3), dtype=dtype), jnp.zeros((1, 3), dtype=dtype))
    )
    # initialize model structure but discard parameters, as parameters are loaded later
    model.init(key, *init_input)
    if args.common.display_model_summary:
        print(model.tabulate(key, *init_input))

    # load parameters
    ckpt = checkpoints.restore_checkpoint(args.test_ckpt, target=None)
    batch_config = NeRFBatchConfig(**ckpt["batch_config"])
    ogrid, params = OccupancyDensityGrid(**ckpt["ogrid"]), ckpt["params"]
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), params)

    scene_metadata_test, test_views = make_nerf_synthetic_scene_metadata(
        rootdir=args.data_root,
        split=args.test_split,
        scale=args.scale,
    )

    logger.info("starting testing (totally {} image(s) to test)".format(len(args.test_indices)))
    for test_i in args.test_indices:
        if test_i < 0 or test_i >= len(test_views):
            logger.warn("skipping out-of-bounds index {} (index should be in range [0, {}])".format(test_i, len(args.test_indices) - 1))
        logger.info("testing on image index {}".format(test_i))
        transform = RigidTransformation(
            rotation=scene_metadata_test.all_transforms[test_i, :9].reshape(3, 3),
            translation=scene_metadata_test.all_transforms[test_i, -3:].reshape(3),
        )
        KEY, key = jran.split(KEY, 2)
        rgb, depth = render_image(
            KEY=key,
            bound=args.bound,
            camera=scene_metadata_test.camera,
            transform_cw=transform,
            options=args.render,
            raymarch_options=args.raymarch,
            batch_config=batch_config,
            ogrid=ogrid,
            param_dict={"params": params},
            nerf_fn=model.apply,
        )
        gt_image = Image.open(test_views[test_i].file)
        gt_image = np.asarray(gt_image)
        gt_image = data.blend_alpha_channel(gt_image, bg=args.render.bg)
        logger.info("{}: psnr={}".format(test_views[test_i].file, data.psnr(gt_image, rgb)))
        dest = args.exp_dir\
            .joinpath(args.test_split)
        dest.mkdir(parents=True, exist_ok=True)

        # rgb
        dest_rgb = dest.joinpath("{:03d}-rgb.png".format(test_i))
        logger.debug("saving comparison image to {}".format(dest_rgb))
        Image.fromarray(np.asarray(rgb)).save(dest_rgb)

        # comparison image
        dest_comparison = dest.joinpath("{:03d}-comparison.png".format(test_i))
        logger.debug("saving comparison image to {}".format(dest_comparison))
        comparison_image_data = data.side_by_side(
            gt_image,
            rgb,
            H=scene_metadata_test.camera.H,
            W=scene_metadata_test.camera.W
        )
        comparison_image_data = data.add_border(comparison_image_data)
        Image.fromarray(np.asarray(comparison_image_data)).save(dest_comparison)

        # depth
        dest_depth = dest.joinpath("{:03d}-depth.png".format(test_i))
        logger.debug("saving predicted depth image to {}".format(dest_depth))
        Image.fromarray(np.asarray(depth)).save(dest_depth)

    return params
