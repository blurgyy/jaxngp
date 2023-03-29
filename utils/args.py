from dataclasses import field
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

from utils.types import LogLevel, RayMarchingOptions, RenderingOptions


@dataclass(frozen=True, kw_only=True)
class CommonArgs:
    # log level
    logging: LogLevel = "INFO"
    # float precision
    prec: int = 32
    # random seed
    seed: int = 1_000_000_007
    # display model information after model init
    display_model_summary: bool=False


@dataclass(frozen=True, kw_only=True)
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

    # loop within training data for this number of iterations, this helps reduce the effective
    # dataloader overhead.
    data_loop: int


@dataclass(frozen=True, kw_only=True)
class ImageFitArgs:
    common: CommonArgs=CommonArgs(
        prec=32,
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
        data_loop=1,
    )


@dataclass(frozen=True, kw_only=True)
class _NeRFArgs:
    # a nerf-synthetic format directory
    data_root: Path

    # experiment artifacts are saved under this directory
    exp_dir: Path

    raymarch: RayMarchingOptions
    render: RenderingOptions

    # half width of axis-aligned bounding-box, i.e. aabb's width is `bound*2`
    bound: float=1.5

    common: CommonArgs=CommonArgs()


@dataclass(frozen=True, kw_only=True)
class NeRFTrainingArgs(_NeRFArgs):
    # number of images to validate
    val_num: int=3

    # if specified, continue training from this checkpoint
    train_ckpt: Optional[Path]=None

    train: TrainingArgs=TrainingArgs(
        # After fixing ray integration weights (6cdaed1), the "mic", "ficus" models from the
        # nerf-synthetic dataset fail to converge.  Using a smaller initial lr like here (1e-3)
        # makes them converge again.
        lr=1e-3,
        momentum=None,
        bs=2**10,
        n_epochs=32,
        n_batches=2**12,
        data_loop=1,
    )

    # raymarching/rendering options during training
    raymarch: RayMarchingOptions=RayMarchingOptions(
        steps=2**7,  # TODO: add coarse network, or implement ray-marching with early stop
        stratified=True,
        n_importance=2**7,
    )
    render: RenderingOptions=RenderingOptions(
        ray_chunk_size=2**10,
        bg=(1.0, 1.0, 1.0),  # white, but ignored by default due to random_bg=True
        random_bg=True,
    )

    # raymarching/rendering options for validating during training
    raymarch_eval: RayMarchingOptions=RayMarchingOptions(
        steps=2**8,
        stratified=True,
        n_importance=2**8+2**9,
    )
    render_eval: RenderingOptions=RenderingOptions(
        ray_chunk_size=2**10,
        bg=(0.0, 0.0, 0.0),  # black
        random_bg=False,
    )


@dataclass(frozen=True, kw_only=True)
class NeRFTestingArgs(_NeRFArgs):
    # if specified, switch to test mode and use this checkpoint
    test_ckpt: Path

    # which test images should be tested on, indices are 0-based
    test_indices: List[int]=field(default_factory=list)

    # which split to test on
    test_split: Literal["train", "test", "val"]="test"

    # raymarching/rendering options during testing
    raymarch: RayMarchingOptions=RayMarchingOptions(
        steps=2**8,
        stratified=True,
        n_importance=2**8+2**9,
    )
    render: RenderingOptions=RenderingOptions(
        ray_chunk_size=2**10,
        bg=(0.0, 0.0, 0.0),  # black
        random_bg=False,
    )
