#!/usr/bin/env python3

from typing import Literal

import tyro

from utils import common
from utils.args import NeRFArgs


def main(args: NeRFArgs):
    logger = common.setup_logging("nerf")

    if args.train_ckpt is not None and args.test_ckpt is not None:
        logger.error("--train-ckpt and --test-ckpt shouldn't be used together")
        exit(1)

    # set running mode
    if args.test_ckpt is not None:
        run_mode = "test"
    elif args.exp_dir.exists():
        logger.error("specified experiment directory '{}' already exists".format(args.exp_dir))
        exit(2)
    else:
        run_mode = "train"

    if args.use_white_bg:
        raise NotImplementedError("Blending image's alpha channel in NeRF is not implemented")
    if args.train_ckpt is not None:
        raise NotImplementedError("Resuming are not implemented")

    if run_mode == "train":
        from app.nerf.train import train
        train(args, logger)
    elif run_mode == "test":
        from app.nerf.test import test
        test(args, logger)


if __name__ == "__main__":
    cfg = tyro.cli(NeRFArgs)
    main(cfg)
