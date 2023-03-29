#!/usr/bin/env python3

from dataclasses import dataclass
from functools import partial, reduce
import os
from pathlib import Path
from typing import Annotated, List, Union
from typing_extensions import assert_never

from PIL import Image
import numpy as np
import tyro

from utils.common import setup_logging
from utils.data import add_border, blend_alpha_channel, psnr, side_by_side
from utils.types import RGBColor


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
    bg: RGBColor=[1.0, 1.0, 1.0]


@dataclass(frozen=True, kw_only=True)
class Metrics:
    gt: Path
    image_paths: tyro.conf.Positional[List[Path]]
    psnr: bool=True
    bg: RGBColor=[1.0, 1.0, 1.0]


Args = Union[
    Annotated[
        Concatenate,
        tyro.conf.subcommand(
            name="cat",
            prefix_name=False,
        ),
    ],
    Annotated[
        Metrics,
        tyro.conf.subcommand(
            name="metrics",
            prefix_name=False,
        ),
    ],
]


def main(args: Args):
    logger = setup_logging("utils", "INFO")
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
            lambda img: blend_alpha_channel(img, bg=args.bg) if img.shape[-1] == 4 else img,
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
            gt_image = blend_alpha_channel(gt_image, bg=args.bg)
        images = list(map(
            lambda img: blend_alpha_channel(img, bg=args.bg) if img.shape[-1] == 4 else img,
            map(np.asarray, map(Image.open, args.image_paths)),
        ))
        for impath, img in zip(args.image_paths, images):
            if args.psnr:
                logger.info("psnr={} ({})".format(psnr(gt_image, img), impath))

    else:
        assert_never("tyro already ensures subcommand passed here are valid, this line should never be executed")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
