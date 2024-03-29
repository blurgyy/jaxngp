#!/usr/bin/env python3

from typing import Annotated
from typing_extensions import assert_never

import tyro

from utils.args import NeRFTrainingArgs,NeRFTestingArgs,NeRFGUIArgs
from utils import common



CmdTrain = Annotated[
    NeRFTrainingArgs,
    tyro.conf.subcommand(
        name="train",
        prefix_name=False,
    ),
]
CmdTest = Annotated[
    NeRFTestingArgs,
    tyro.conf.subcommand(
        name="test",
        prefix_name=False,
    ),
]
CmdGui = Annotated[
    NeRFGUIArgs,
    tyro.conf.subcommand(
        name="gui",
        prefix_name=False,
    ),
]


MainArgsType = CmdTrain | CmdTest | CmdGui


def main(args: MainArgsType):
    logger = common.setup_logging("nerf")
    KEY = common.set_deterministic(args.common.seed)

    if isinstance(args, NeRFTrainingArgs):
        from app.nerf.train import train
        return train(KEY, args, logger)
    elif isinstance(args, NeRFTestingArgs):
        from app.nerf.test import test
        return test(KEY, args, logger)
    elif isinstance(args, NeRFGUIArgs):
        from app.nerf.gui import GuiWindow
        return GuiWindow(KEY, args, logger)
    else:
        assert_never(args)


if __name__ == "__main__":
    args = tyro.cli(MainArgsType)
    exit(main(args))
