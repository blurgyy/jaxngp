from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import tyro

from utils.types import (
    CameraOverrideOptions,
    LogLevel,
    OrbitTrajectoryOptions,
    RayMarchingOptions,
    RenderingOptions,
    SceneOptions,
    TransformsProvider,
)


@dataclass(frozen=True, kw_only=True)
class CommonArgs:
    # log level
    logging: LogLevel = "INFO"

    # random seed
    seed: int = 1_000_000_007

    # display model information after model init
    summary: bool=False


@dataclass(frozen=True, kw_only=True)
class TrainingArgs:
    # learning rate
    lr: float

    momentum: float | None

    # batch size
    bs: int

    # training epochs
    n_epochs: int

    # batches per epoch
    n_batches: int

    # loop within training data for this number of iterations, this helps reduce the effective
    # dataloader overhead.
    data_loop: int

    # will validate every `validate_every` epochs, set this to a large value to disable validation
    validate_every: int

    # number of latest checkpoints to keep
    keep: int=1

    # how many epochs should a new checkpoint to be kept (in addition to keeping the last `keep`
    # checkpoints)
    keep_every: int | None=None

    @property
    def keep_every_n_steps(self) -> int | None:
        if self.keep_every is None:
            return None
        else:
            return self.keep_every * self.n_batches

@dataclass(frozen=True, kw_only=True)
class ImageFitArgs:
    common: CommonArgs=CommonArgs()
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
        validate_every=1,
    )


@dataclass(frozen=True, kw_only=True)
class NeRFArgsBase:
    # experiment artifacts are saved under this directory
    exp_dir: Path

    raymarch: RayMarchingOptions
    render: RenderingOptions
    scene: SceneOptions

    common: CommonArgs=CommonArgs()


@dataclass(frozen=True, kw_only=True)
class NeRFTrainingArgs(NeRFArgsBase):
    # directories or transform.json files containing data for training
    frames_train: tyro.conf.Positional[Tuple[Path, ...]]

    # directories or transform.json files containing data for validation
    frames_val: Tuple[Path, ...]=()

    # if specified, continue training from this checkpoint
    ckpt: Path | None=None

    # training hyper parameters
    train: TrainingArgs=TrainingArgs(
        # This is a relatively large learning rate, should be used jointly with
        # `threasholded_exponential` as density activation, and random color as supervision for
        # transparent pixels.
        lr=1e-2,
        momentum=None,
        bs=1024 * (1<<10),
        n_epochs=50,
        n_batches=2**10,
        data_loop=1,
        validate_every=10,
    )

    # raymarching/rendering options during training
    raymarch: RayMarchingOptions=RayMarchingOptions(
        diagonal_n_steps=1<<10,
        perturb=True,
        density_grid_res=128,
    )
    render: RenderingOptions=RenderingOptions(
        bg=(1.0, 1.0, 1.0),  # white, but ignored by default due to random_bg=True
        random_bg=True,
    )
    scene: SceneOptions=SceneOptions(
        sharpness_threshold=-1.,
        world_scale=1.0,
        resolution_scale=1.0,
    )

    # raymarching/rendering options for validating during training
    raymarch_eval: RayMarchingOptions=RayMarchingOptions(
        diagonal_n_steps=1<<10,
        perturb=False,
        density_grid_res=128,
    )
    render_eval: RenderingOptions=RenderingOptions(
        bg=(0.0, 0.0, 0.0),  # black
        random_bg=False,
    )


@dataclass(frozen=True, kw_only=True)
class NeRFTestingArgs(NeRFArgsBase):
    frames: tyro.conf.Positional[Tuple[Path, ...]]

    camera_override: CameraOverrideOptions=None

    # use checkpoint from this path (can be a directory) for testing
    ckpt: Path

    # if specified, render with a generated orbiting trajectory instead of the loaded frame
    # transformations
    trajectory: TransformsProvider="loaded"

    orbit: OrbitTrajectoryOptions

    # naturally sort frames according to their file names before testing
    sort_frames: bool=False

    # if specified value contains "video", a video will be saved; if specified value contains
    # "image", rendered images will be saved.  Value can contain both "video" and "image", e.g.,
    # `--save-as "video-image"` will save both video and images.
    save_as: str="image and video"

    # raymarching/rendering options during testing
    raymarch: RayMarchingOptions=RayMarchingOptions(
        diagonal_n_steps=1<<10,
        perturb=False,
        density_grid_res=128,
    )
    render: RenderingOptions=RenderingOptions(
        bg=(0.0, 0.0, 0.0),  # black
        random_bg=False,
    )
    scene: SceneOptions=SceneOptions(
        sharpness_threshold=-1.,
        world_scale=1.0,
        resolution_scale=1.0,
    )
