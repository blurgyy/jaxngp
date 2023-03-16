from dataclasses import dataclass
from typing import Literal, Optional

@dataclass(frozen=True)
class CommonArgs:
    # log level
    logging: Literal["DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    # float precision
    prec: int = 32
    # random seed
    seed: int = 1_000_000_007

@dataclass(frozen=True)
class DataArgs:
    # number of workers used in dataloaders
    n_workers: int
    # loop within training data for this number of iterations, this helps reduce the effective
    # dataloader overhead.
    loop: int

@dataclass(frozen=True)
class TrainingArgs:
    # learning rate
    lr: float

    momentum: Optional[float]

    # batch size
    bs: int
    # training epochs
    n_epochs: int


@dataclass(frozen=True)
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
        # ~532it/s. setting bs to 2^{10} with 2 extra data loader workers achieves a satisfing
        # tradeoff here.
        bs=2**10,
        n_epochs=32,
    )
