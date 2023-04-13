#!/usr/bin/env python3

from typing import Annotated, Union
from typing_extensions import assert_never

import tyro

from utils import common
from utils.args import NeRFTestingArgs, NeRFTrainingArgs


MainArgsType = Union[
    Annotated[
        NeRFTrainingArgs,
        tyro.conf.subcommand(
            name="train",
            prefix_name=False,
        ),
    ],
    Annotated[
        NeRFTestingArgs,
        tyro.conf.subcommand(
            name="test",
            prefix_name=False,
        ),
    ],
]


def main(args: MainArgsType):
    logger = common.setup_logging("nerf")
    KEY = common.set_deterministic(args.common.seed)

    if isinstance(args, NeRFTrainingArgs):
        from app.nerf.train import train
        train(KEY, args, logger)
    elif isinstance(args, NeRFTestingArgs):
        from app.nerf.test import test
        test(KEY, args, logger)
    else:
        assert_never()


if __name__ == "__main__":
    args = tyro.cli(MainArgsType)
    main(args)
