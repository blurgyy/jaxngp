from dataclasses import field
from pathlib import Path
from typing import List, Literal, Optional

from flax.struct import dataclass

from utils.types import RayMarchingOptions, RenderingOptions

@dataclass
class CommonArgs:
    # log level
    logging: Literal["DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    # float precision
    prec: int = 32
    # random seed
    seed: int = 1_000_000_007
    # display model information after model init
    display_model_summary: bool=False

@dataclass
class DataArgs:
    # number of workers used in dataloaders
    n_workers: int
    # loop within training data for this number of iterations, this helps reduce the effective
    # dataloader overhead.
    loop: int

@dataclass
class TrainingArgs:
    # learning rate
    lr: float

    momentum: Optional[float]

    # batch size
    bs: int

    # training epochs
    n_epochs: int

    # batches per epoch
    n_batches: int


@dataclass
class ImageFitArgs:
    common: CommonArgs=CommonArgs(
        prec=32,
    )
    data: DataArgs=DataArgs(
        n_workers=2,
        loop=1,
    )
    train: TrainingArgs=TrainingArgs(
        # paper:
        #   We observed fastest convergence with a learning rate of 10^{-4} for signed distance
        #   functions and 10^{-2} otherwise
        #
        # We use a smaller learning rate since our batch size is much smaller the paper (see below).
        lr=1e-3,
        momentum=None,  # using adam so momentum doesn't matter
        # paper:
        #   ...as well a a batch size of 2^{14} for neural radiance caching and 2^{18} otherwise.
        #
        # In our case, setting the batch size to a larger number hinders data loading performance,
        # and thus causes the GPU not being fully occupied.  On the other hand, setting the batch
        # size to a smaller one utilizes the GPU fully, but the iterations per second capped at
        # some rate which results in lower throughput. setting bs to 2^{10} achieves a satisfing
        # tradeoff here.
        bs=2**10,
        n_epochs=32,
        n_batches=2**30,
    )


@dataclass
class NeRFArgs:
    # a nerf-synthetic format directory
    data_root: Path

    # experiment artifacts are saved under this directory
    exp_dir: Path

    # number of images to validate
    val_num: int=3

    # whether to treat transparent pixels as white
    use_white_bg: bool=False

    # if specified, continue training from this checkpoint
    train_ckpt: Optional[Path]=None

    # if specified, switch to test mode and use this checkpoint
    test_ckpt: Optional[Path]=None

    # which test images should be tested on, indices are 0-based
    test_indices: List[int]=field(default_factory=list)

    # which split to test on
    test_split: Literal["train", "test", "val"]="test"

    common: CommonArgs=CommonArgs()

    data: DataArgs=DataArgs(
        n_workers=0,
        loop=1,
    )

    train: TrainingArgs=TrainingArgs(
        lr=1e-2,
        momentum=None,
        bs=2**10,
        n_epochs=32,
        n_batches=2**10,
    )

    # Width of axis-aligned bounding-box
    bound: float=1.5

    raymarch: RayMarchingOptions=RayMarchingOptions(
        steps=2**10,
    )

    rendering: RenderingOptions=RenderingOptions(
        ray_chunk_size=2**10,
    )
