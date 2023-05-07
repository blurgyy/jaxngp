#!/usr/bin/env python3

from dataclasses import dataclass
from functools import partial, reduce
import os
from pathlib import Path
from typing import Annotated, List
from typing_extensions import assert_never

from PIL import Image
import numpy as np
import tyro

from utils.common import setup_logging
from utils.data import (
    add_border,
    blend_rgba_image_array,
    create_dataset_from_single_camera_image_collection,
    create_dataset_from_video,
    psnr,
    side_by_side,
)
from utils.types import ColmapMatcherType, RGBColor


@dataclass(frozen=True, kw_only=True)
class Concatenate:
    image_paths: tyro.conf.Positional[List[Path]]
    # output image save path, the path will be overwritten with a warning
    out: Path
    # if specified, concatenate vertically instead of horizontally
    vertical: bool=False
    # gap between adjacent images, in pixels
    gap: int=0
    # border in pixels
    border: int=0
    bg: RGBColor=(1.0, 1.0, 1.0)


@dataclass(frozen=True, kw_only=True)
class Metrics:
    gt: Path
    image_paths: tyro.conf.Positional[List[Path]]
    psnr: bool=True
    bg: RGBColor=(1.0, 1.0, 1.0)


@dataclass(frozen=True, kw_only=True)
class CreateDataset:
    # video or directory of image collection
    src: tyro.conf.Positional[Path]

    # where to write the images and transforms_{train,val,test}.json
    root_dir: Path

    # `Sequntial` for continuous frames, `Exhaustive` for all possible pairs
    matcher: ColmapMatcherType

    # given that cameras' average distance to origin is 4.0, what would the scene's bound be?
    # (a scene's bound is the half width of its bounding box, e.g. a scene with bound 4.0 has a
    # bounding box with width 8.0, centered at the origin)
    # this parameter just specifies the "aabb_scale" in the generated
    # `transforms_{train,val,test}.json`.
    bound: float

    # how many frames to extract per second, only used when src is a video
    fps: int=3


CmdCat = Annotated[
    Concatenate,
    tyro.conf.subcommand(
        name="cat",
        prefix_name=False,
        description="concatenate images horizontally or vertically",
    ),
]
CmdMetrics = Annotated[
    Metrics,
    tyro.conf.subcommand(
        name="metrics",
        prefix_name=False,
        description="compute metrics between images",
    ),
]
CmdCreateDataseet = Annotated[
    CreateDataset,
    tyro.conf.subcommand(
        name="create-dataset",
        prefix_name=False,
        description="create a instant-ngp format dataset from a video or a directory of images",
    ),
]


Args = CmdCat | CmdCreateDataseet | CmdMetrics


def main(args: Args):
    logger = setup_logging("utils", level="DEBUG")
    if isinstance(args, Concatenate):
        if args.out.is_dir():
            logger.error("output path '{}' is a directory".format(args.out))
            exit(1)
        if args.out.exists():
            logger.warn("output path '{}' exists and will be overwritten".format(args.out))
            if not os.access(args.out, os.W_OK):
                logger.error("output path '{}' is readonly".format(args.out))
                exit(2)
        if args.out.suffix.lower() not in map(lambda x: "." + x, ["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"]):
            logger.warn("the file extension '{}' might not be supported".format(args.out.suffix))

        images = list(map(
            lambda img: blend_rgba_image_array(img, bg=args.bg) if img.shape[-1] == 4 else img,
            map(np.asarray, map(Image.open, args.image_paths)),
        ))
        H, W = images[0].shape[:2]
        oimg = reduce(
            partial(
                side_by_side,
                H=(None if args.vertical else H),
                W=(W if args.vertical else None),
                vertical=args.vertical,
                gap=args.gap,
            ),
            images,
        )
        oimg = add_border(oimg, width=args.border)
        logger.info("saving image ...")
        Image.fromarray(np.asarray(oimg)).save(args.out)
        logger.info("image ({}x{}) saved to '{}'".format(oimg.shape[1], oimg.shape[0], args.out))

    elif isinstance(args, Metrics):
        gt_image = np.asarray(Image.open(args.gt))
        if gt_image.shape[-1] == 4:
            gt_image = blend_rgba_image_array(gt_image, bg=args.bg)
        images = list(map(
            lambda img: blend_rgba_image_array(img, bg=args.bg) if img.shape[-1] == 4 else img,
            map(np.asarray, map(Image.open, args.image_paths)),
        ))
        for impath, img in zip(args.image_paths, images):
            if args.psnr:
                logger.info("psnr={} ({})".format(psnr(gt_image, img), impath))

    elif isinstance(args, CreateDataset):
        if args.src.is_dir():
            create_dataset_from_single_camera_image_collection(
                raw_images_dir=args.src,
                dataset_root_dir=args.root_dir,
                matcher=args.matcher,
                bound=args.bound,
            )
        else:
            create_dataset_from_video(
                video_path=args.src,
                dataset_root_dir=args.root_dir,
                bound=args.bound,
                fps=args.fps,
            )

    else:
        assert_never("tyro already ensures subcommand passed here are valid, this line should never be executed")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
