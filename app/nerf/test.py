from typing import List

from PIL import Image
from flax.training import checkpoints
import jax
import jax.random as jran
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from models.nerfs import make_nerf_ngp, make_skysphere_background_model_ngp
from models.renderers import render_image_inference
from utils import common, data
from utils.args import NeRFTestingArgs,GuiWindowArgs
from utils.types import NeRFState, RenderedImage, RigidTransformation,SceneMeta


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

    logger.info("loading testing frames")
    scene_data, test_views = data.load_scene(
        srcs=args.frames,
        world_scale=args.scene.world_scale,
        image_scale=args.scene.image_scale,
    )

    # load parameters
    state: NeRFState = checkpoints.restore_checkpoint(
        args.ckpt,
        target=NeRFState.empty(
            raymarch=args.raymarch,
            render=args.render,
            scene_options=args.scene,
            scene_meta=scene_data.meta,
            nerf_fn=make_nerf_ngp(bound=scene_data.meta.bound).apply,
            bg_fn=make_skysphere_background_model_ngp(bound=scene_data.meta.bound).apply if scene_data.meta.bg else None,
        ),
    )
    # WARN:
    #   flax.checkpoints.restore_checkpoint() returns a pytree with all arrays of numpy's array type,
    #   which slows down inference.  use jax.device_put() to convert them to jax's DeviceArray type.
    # REF: <https://github.com/google/flax/discussions/1199#discussioncomment-635132>
    state = jax.device_put(state)

    rendered_images: List[RenderedImage] = []
    try:
        logger.info("starting testing (totally {} image(s) to test)".format(len(test_views)))
        for test_i, test_view in enumerate(tqdm(test_views, desc="testing", bar_format=common.tqdm_format)):
            logger.debug("testing on {}".format(test_view))
            transform = RigidTransformation(
                rotation=scene_data.all_transforms[test_i, :9].reshape(3, 3),
                translation=scene_data.all_transforms[test_i, -3:].reshape(3),
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

    save_dest = args.exp_dir.joinpath(args.split)
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

    return mean_psnr


from utils.types import PinholeCamera
import jax.numpy as jnp
def test_render(KEY: jran.KeyArray, args: NeRFTestingArgs,gui_args:GuiWindowArgs, logger: common.Logger,camera_pose:jnp.array,state:NeRFState):
    
    dtype_np=np.float32
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
     #camera info
    Camera_W, Camera_H = 800,800
    fovx = 0.6911112070083618
    focal = float(.5 * Camera_W / np.tan(fovx / 2))
    image_scale=args.scene.image_scale
    camera = PinholeCamera(
            W=int(Camera_W * image_scale),
            H=int(Camera_H * image_scale),
            fx=focal * image_scale,
            fy=focal * image_scale,
            cx=Camera_W / 2 * image_scale,
            cy=Camera_H / 2 * image_scale,
        )
    scene_meta = SceneMeta(
        bound=gui_args.bound,
        camera=camera,
    )
    # load parameters
    # state: NeRFState = checkpoints.restore_checkpoint(
    #     args.ckpt,
    #     target=NeRFState.empty(
    #         raymarch=args.raymarch,
    #         render=args.render,
    #         scene_options=args.scene,
    #         scene_meta=scene_meta,
    #         nerf_fn=make_nerf_ngp(bound=scene_meta.bound).apply,
    #         bg_fn=make_skysphere_background_model_ngp(bound=gui_args.bound).apply if args.scene.with_bg else None,
    #     ),
    # )

    # # WARN:
    # #   flax.checkpoints.restore_checkpoint() returns a pytree with all arrays of numpy's array type,
    # #   which slows down inference.  use jax.device_put() to convert them to jax's DeviceArray type.
    # # REF: <https://github.com/google/flax/discussions/1199#discussioncomment-635132>
    # state = jax.device_put(state)
    
    
    #camera pose
    transform = RigidTransformation(rotation=camera_pose[:3, :3],
                                    translation=jnp.squeeze(camera_pose[:3, 3].reshape(-1,3),axis=0))
    #render
    KEY, key = jran.split(KEY, 2)
    bg, rgb, depth = render_image_inference(
        KEY=key,
        transform_cw=transform,
        state=state,
    )
    rendered_image=RenderedImage(
        bg=data.to_cpu(bg),
        rgb=data.to_cpu(rgb),
        depth=data.to_cpu(depth),
    )
    # #save
    # save_dest = args.exp_dir.joinpath("test")
    # save_dest.mkdir(parents=True, exist_ok=True)
    
    # dest_rgb = save_dest.joinpath("rgb")
    # dest_depth = save_dest.joinpath("depth")

    # dest_rgb.mkdir(parents=True, exist_ok=True)
    # dest_depth.mkdir(parents=True, exist_ok=True)

    # logger.info("saving as rgb image in {}".format(dest_rgb.joinpath("rgb.png")))
    # logger.info("saving as depth image in {}".format(dest_depth.joinpath("depth.png")))
    # Image.fromarray(np.asarray(rendered_image.rgb)).save(dest_rgb.joinpath("rgb.png"))
    # Image.fromarray(np.asarray(rendered_image.depth)).save(dest_depth.joinpath("depth.png"))
    # if "image" in args.save_as:
        # dest_rgb = save_dest.joinpath("rgb")
        # dest_depth = save_dest.joinpath("depth")

        # dest_rgb.mkdir(parents=True, exist_ok=True)
        # dest_depth.mkdir(parents=True, exist_ok=True)

        # logger.debug("saving as image")
        # Image.fromarray(np.asarray(rendered_image.rgb)).save(dest_rgb.joinpath("rgb.png"))
        # Image.fromarray(np.asarray(rendered_image.depth)).save(dest_depth.joinpath("depth.png"))
    rgb=np.array(rendered_image.rgb,dtype=dtype_np)/255.
    depth=np.array(rendered_image.depth,dtype=dtype_np)/255.
    return rgb,depth


# import logging
# import jax.numpy as jnp
# def test_render(KEY: jran.KeyArray, args: NeRFTestingArgs, logger: logging.Logger,camera_pose:jnp.array):
    
#     from utils.types import PinholeCamera
#     if not args.ckpt.exists():
#         logger.warn("specified checkpoint '{}' does not exist".format(args.test_ckpt))
#         exit(1)

#     dtype = getattr(jnp, "float{}".format(args.common.prec))
#     logger.setLevel(args.common.logging.upper())
#     # model parameters
#     KEY, key = jran.split(KEY, 2)
#     model, init_input = (
#         make_nerf_ngp(bound=args.scene.bound),
#         (jnp.zeros((1, 3), dtype=dtype), jnp.zeros((1, 3), dtype=dtype))
#     )
#     # initialize model structure but discard parameters, as parameters are loaded later
#     model.init(key, *init_input)
#     if args.common.summary:
#         print(model.tabulate(key, *init_input))

#     # load parameters
#     ckpt = checkpoints.restore_checkpoint(args.test_ckpt, target=None)
#     batch_config = NeRFBatchConfig(**ckpt["batch_config"])
#     batch_config = batch_config
#     ogrid, params = OccupancyDensityGrid(**ckpt["ogrid"]), ckpt["params"]
#     params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), params)
#     params = jax.lax.stop_gradient(params)

#     #camera info
#     Camera_W, Camera_H = 800,800
#     fovx = 0.6911112070083618
#     focal = float(.5 * Camera_W / np.tan(fovx / 2))
#     camera = PinholeCamera(
#         W=Camera_W,
#         H=Camera_H,
#         focal=focal,
#     )
#     #camera pose
#     transform = RigidTransformation(rotation=camera_pose[:3, :3],
#                                     translation=jnp.squeeze(camera_pose[:3, 3].reshape(-1,3),axis=0))
    
#     #render
#     KEY, key = jran.split(KEY, 2)
#     rgb, depth = render_image(
#         KEY=key,
#         bound=args.scene.bound,
#         #camera=scene_metadata_test.camera,
#         camera=camera,
#         transform_cw=transform,
#         options=args.render,
#         raymarch_options=args.raymarch,
#         batch_config=batch_config,
#         ogrid=ogrid,
#         param_dict={"params": params},
#         nerf_fn=model.apply,
#     )
    
#     #save images
#     tested_image=TestedImage(
#         rgb=rgb,
#         depth=depth,
#         gt_path=Path(""),
#     )
#     #dest = args.exp_dir.joinpath(args.test_split)
#     dest = args.exp_dir
#     dest.mkdir(parents=True, exist_ok=True)
    
#     dest_rgb = dest.joinpath("rgb.png")
#     dest_depth = dest.joinpath("depth.png")
#     logger.info("save rgb image to path:{}".format(dest_rgb))
#     tested_image.save_rgb(dest_rgb)
#     logger.info("save depth image to path:{}".format(dest_depth))
#     tested_image.save_depth(dest_depth)
#     dtype_np = getattr(np, "float{}".format(args.common.prec))
    
#     rgb=np.array(rgb,dtype=np.float128)/256.
#     depth=np.array(depth,dtype=np.float128)/256.
#     return rgb,depth
