#!/usr/bin/env python3

from dataclasses import dataclass
from functools import partial, reduce
import os
from pathlib import Path
from typing import Tuple, Union
from typing_extensions import assert_never

from PIL import Image
import numpy as np
import tyro

from utils.common import setup_logging
from utils.data import add_border, blend_alpha_channel, side_by_side
from utils.types import LogLevel


@dataclass(frozen=True, kw_only=True)
class Concatenate:
    # output image save path, the path will be overwritten with a warning
    out: Path
    # if specified, concatenate vertically instead of horizontally
    vertical: bool=False
    # gap between adjacent images, in pixels
    gap: int=0
    # border in pixels
    border: int=0
    use_white_bg: bool=True


@dataclass(frozen=True, kw_only=True)
class Metrics:
    psnr: bool=True


@dataclass(frozen=True, kw_only=True)
class Args:
    do: Union[Concatenate, Metrics]
    logging: LogLevel = "INFO"


def main(_args: Tuple[Args, list[str]]):
    args, extra_cli_args = _args
    logger = setup_logging("utils", args.logging)
    do = args.do
    if isinstance(do, Concatenate):
        if do.out.is_dir():
            logger.error("output path '{}' is a directory".format(do.out))
            exit(1)
        if do.out.exists():
            logger.warn("output path '{}' exists and will be overwritten".format(do.out))
            if not os.access(do.out, os.W_OK):
                logger.error("output path '{}' is readonly".format(do.out))
                exit(2)
        if do.out.suffix.lower() not in map(lambda x: "." + x, ["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"]):
            logger.warn("the file extension '{}' might not be supported".format(do.out.suffix))

        images = list(map(
            lambda img: blend_alpha_channel(img, use_white_bg=do.use_white_bg) if img.shape[-1] == 4 else img,
            map(np.asarray, map(Image.open, extra_cli_args)),
        ))
        H, W = images[0].shape[:2]
        oimg = reduce(
            partial(
                side_by_side,
                H=(None if do.vertical else H),
                W=(W if do.vertical else None),
                vertical=do.vertical,
                gap=do.gap,
            ),
            images,
        )
        oimg = add_border(oimg, width=do.border)
        logger.info("saving image ...")
        Image.fromarray(np.asarray(oimg)).save(do.out)
        logger.info("image ({}x{}) saved to '{}'".format(oimg.shape[1], oimg.shape[0], do.out))

    elif isinstance(do, Metrics):
        raise NotImplementedError("Metrics is not implemented")

    else:
        assert_never("tyro already ensures subcommand passed here are valid, this line should never be executed")


if __name__ == "__main__":
    do = tyro.cli(Args, return_unknown_args=True)
    main(do)
