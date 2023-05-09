from typing import Annotated
from typing_extensions import assert_never

import tyro

from utils.args import NeRFTrainingArgs,NeRFTestingArgs,GuiWindowArgs


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
    GuiWindowArgs,
    tyro.conf.subcommand(
        name="gui",
        prefix_name=False,
    ),
]


MainArgsType = CmdTrain | CmdTest | CmdGui


def main(args: MainArgsType):
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES']='1'
    from utils import common
    
    logger = common.setup_logging("nerf")
    KEY = common.set_deterministic(args.common.seed)

    if isinstance(args, NeRFTrainingArgs):
        from app.nerf.train import train
        train(KEY, args, logger)
    elif isinstance(args, NeRFTestingArgs):
        from app.nerf.test import test
        test(KEY, args, logger)
    elif isinstance(args,GuiWindowArgs):        
        from app.nerf.gui import GuiWindow
        GuiWindow(KEY, args, logger)
    else:
        assert_never()


if __name__ == "__main__":
    args = tyro.cli(MainArgsType)
    main(args)