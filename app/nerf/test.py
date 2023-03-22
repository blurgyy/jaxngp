import logging

from PIL import Image
from flax.training import checkpoints
import jax.numpy as jnp
import jax.random as jran
import numpy as np

from models.nerfs import make_nerf_ngp
from models.renderers import render_image
from utils import common, data
from utils.args import NeRFArgs
from utils.data import make_nerf_synthetic_scene_metadata
from utils.types import RigidTransformation


def test(args: NeRFArgs, logger: logging.Logger):
    if not args.test_ckpt.exists():
        logger.warn("specified checkpoint '{}' does not exist".format(args.test_ckpt))
        exit(1)

    if len(args.test_indices) == 0:
        logger.warn("got empty test indices, you might want to specify some image indices via --test-indices")
        logger.warn("proceeding anyway ...")

    dtype = getattr(jnp, "float{}".format(args.common.prec))
    logger.setLevel(args.common.logging.upper())

    # deterministic
    K = common.set_deterministic(seed=args.common.seed)

    # model parameters
    K, key = jran.split(K, 2)
    model, init_input = (
        make_nerf_ngp(aabb=args.aabb),
        (jnp.zeros((1, 3), dtype=dtype), jnp.zeros((1, 3), dtype=dtype))
    )
    # initialize model structure but discard parameters, as parameters are loaded later
    model.init(key, *init_input)
    if args.common.display_model_summary:
        print(model.tabulate(key, *init_input))

    # load parameters
    params = checkpoints.restore_checkpoint(args.test_ckpt, target=None)["params"]

    scene_metadata_test, test_views = make_nerf_synthetic_scene_metadata(
        rootdir=args.data_root,
        split=args.test_split,
        use_white_bg=args.render.use_white_bg,
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
        image = render_image(
            aabb=args.aabb,
            camera=scene_metadata_test.camera,
            transform_cw=transform,
            options=args.render,
            raymarch_options=args.raymarch,
            param_dict={"params": params},
            nerf_fn=model.apply,
        )
        gt_image = Image.open(test_views[test_i].file)
        gt_image = np.asarray(gt_image)
        gt_image = data.blend_alpha_channel(gt_image, use_white_bg=args.render.use_white_bg)
        logger.info("{}: psnr={}".format(test_views[test_i].file, data.psnr(gt_image, image)))
        dest = args.exp_dir\
            .joinpath("test")\
            .joinpath("{:03d}.png".format(test_i))
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("saving image to {}".format(dest))
        Image.fromarray(np.asarray(image)).save(dest)

    return params
