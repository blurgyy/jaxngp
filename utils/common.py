from concurrent.futures import Executor, Future, ThreadPoolExecutor
import functools
import logging
import logging
import os
from pathlib import Path
import random
import shutil
from typing import Any, Dict, Hashable, Iterable, Sequence, get_args

import colorama
from colorama import Back, Fore, Style
from flax.metrics import tensorboard
import git
import jax
from jax._src.lib import xla_client as xc
import jax.random as jran
import numpy as np
import tensorflow as tf
from tqdm import tqdm as tqdm_original

from ._constants import tqdm_format
from .types import LogLevel


class Logger(logging.Logger):
    _tb: tensorboard.SummaryWriter | None=None
    _executor: Executor | None=None

    _last_job: Future=None

    def __init__(self, name: str, level: int | LogLevel) -> None:
        super().__init__(name, level)

    def setup_tensorboard(self, tb: tensorboard.SummaryWriter, executor: Executor) -> None:
        self._tb = tb
        self._executor = executor

    def wait_last_job(self):
        if self._last_job is not None and not self._last_job.done():
            return self._last_job.result()

    def write_scalar(self, tag: str, value: Any, step: int) -> None:
        if self._tb is not None:
            self.wait_last_job()
            # NOTE: writing scalars is fast(ish) enough to not need a thread pool
            self._executor.submit(self._tb.scalar, tag, value, step)
    def write_image(self, tag: str, image: Any, step: int, max_outputs: int) -> None:
        if self._tb is not None:
            self.wait_last_job()
            self._last_job = self._executor.submit(self._tb.image, tag, image, step, max_outputs)
    def write_hparams(self, hparams: Dict[str, Any]) -> None:
        if self._tb is not None:
            self.wait_last_job()
            self._last_job = self._executor.submit(self._tb.hparams, hparams)

    def write_metrics_to_tensorboard(
        self,
        metrics: Dict[str, jax.Array | float],
        step: jax.Array | int,
    ) -> None:
        def linear_to_db(val: float, maxval: float):
            return 20 * np.log10(np.sqrt(maxval / val))
        self.write_scalar(
            "batch/↓loss (rgb)",
            metrics["loss"]["rgb"],
            step,
        )
        self.write_scalar(
            "batch/↑estimated PSNR (db)",
            linear_to_db(metrics["loss"]["rgb"], maxval=1.),
            step,
        )
        self.write_scalar(
            "batch/↓loss (total variation)",
            metrics["loss"]["total_variation"],
            step,
        )
        self.write_scalar(
            "batch/effective batch size (not compacted)",
            metrics["measured_batch_size_before_compaction"],
            step,
        )
        self.write_scalar(
            "batch/↑effective batch size (compacted)",
            metrics["measured_batch_size"],
            step,
        )
        self.write_scalar(
            "rendering/↓effective samples per ray",
            metrics["measured_batch_size"] / metrics["n_valid_rays"],
            step,
        )
        self.write_scalar(
            "rendering/↓marched samples per ray",
            metrics["measured_batch_size_before_compaction"] / metrics["n_valid_rays"],
            step,
        )
        self.write_scalar(
            "rendering/↑number of marched rays",
            metrics["n_valid_rays"],
            step,
        )


tqdm = functools.partial(tqdm_original, bar_format=tqdm_format)


def backup_current_codebase(
    exp_dir: Path | str,
    /,
    name_prefix: str,
) -> int:
    """Backup current codebase to a directory named 'src' under the specified `exp_dir` directory,
    creating it if it does not exist.
    """
    repo = git.Repo(".", search_parent_directories=True)
    try:
        __initial_commit = repo.commit("aa9f8c73c2a2d164e9e117b99a6543893eeed23f")
    except ValueError:
        raise ValueError("This is not the jaxngp repository")

    os.chdir(repo.git.working_dir)

    exp_dir = Path(exp_dir)
    save_root_dir = exp_dir.joinpath("runs")
    save_root_dir.mkdir(parents=False, exist_ok=True)
    epoch = 0
    save_dir = save_root_dir.joinpath("{}{:04d}".format(name_prefix, epoch))
    while save_dir.exists():
        epoch += 1
        save_dir = save_root_dir.joinpath("{}{:04d}".format(name_prefix, epoch))

    latest_run_lnk = exp_dir.joinpath("{}latest-run".format(name_prefix))
    if latest_run_lnk.exists():
        if (
            latest_run_lnk.is_symlink()
            and latest_run_lnk.readlink().parent.absolute() == save_dir.parent.absolute()
        ):
            latest_run_lnk.unlink()
        else:
            raise RuntimeError(
                "the path '{}' exists but is not a symlink to a previous run".format(latest_run_lnk)
            )
    elif latest_run_lnk.is_symlink():  # the link does not exist, but it is a symlink, that makes it a broken symlink
        latest_run_lnk.unlink()

    save_dir.mkdir(parents=False, exist_ok=False)
    latest_run_lnk.symlink_to(save_dir.absolute())

    shutil.copyfile("flake.nix", save_dir.joinpath("flake.nix"))
    shutil.copyfile("flake.lock", save_dir.joinpath("flake.lock"))
    shutil.copyfile("pyproject.toml", save_dir.joinpath("pyproject.toml"))
    shutil.copyfile("README.md", save_dir.joinpath("README.md"))

    def ignored_files(dir, files):
        return [
            ".clangd",
            ".clang-format",
        ]

    shutil.copytree("app", save_dir.joinpath("app"), dirs_exist_ok=False, ignore=ignored_files)
    shutil.copytree("deps", save_dir.joinpath("deps"), dirs_exist_ok=False, ignore=ignored_files)
    shutil.copytree("models", save_dir.joinpath("models"), dirs_exist_ok=False, ignore=ignored_files)
    shutil.copytree("utils", save_dir.joinpath("utils"), dirs_exist_ok=False, ignore=ignored_files)

    with open(save_dir.joinpath("commit-sha"), "w") as f:
        f.write(repo.head.object.hexsha)
    with open(save_dir.joinpath("working-directory"), "w") as f:
        f.write(os.getcwd())

    return save_dir


def compose(*fns):
    def _inner(x):
        for fn in fns:
            x = fn(x)
        return x
    return _inner


def mkValueError(desc, value, type):
    variants = get_args(type)
    assert value not in variants
    return ValueError("Unexpected {}: '{}', expected one of [{}]".format(desc, value, "|".join(variants)))


# NOTE:
#   Jitting a vmapped function seems to give the desired performance boost, while vmapping a jitted
#   function might not work at all.  Except for the experiments I conducted myself, some related
#   issues:
# REF:
#   * <https://github.com/google/jax/issues/6312>
#   * <https://github.com/google/jax/issues/7449>
def vmap_jaxfn_with(
        # kwargs copied from `jax.vmap` source
        in_axes: int | Sequence[Any]=0,
        out_axes: Any = 0,
        axis_name: Hashable | None = None,
        axis_size: int | None = None,
        spmd_axis_name: Hashable | None = None,
    ):
    return functools.partial(
        jax.vmap,
        in_axes=in_axes,
        out_axes=out_axes,
        axis_name=axis_name,
        axis_size=axis_size,
        spmd_axis_name=spmd_axis_name,
    )


def jit_jaxfn_with(
        # kwargs copied from `jax.jit` source
        static_argnums: int | Iterable[int] | None = None,
        static_argnames: str | Iterable[str] | None = None,
        device: xc.Device | None = None,
        backend: str | None = None,
        donate_argnums: int | Iterable[int] = (),
        inline: bool = False,
        keep_unused: bool = False,
        abstracted_axes: Any | None = None,
    ):
    return functools.partial(
        jax.jit,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        device=device,
        backend=backend,
        donate_argnums=donate_argnums,
        inline=inline,
        keep_unused=keep_unused,
        abstracted_axes=abstracted_axes,
    )


def setup_logging(
    name: str,
    /,
    file: str | Path | None=None,
    with_tensorboard: bool=False,
    level: LogLevel="INFO",
    file_level: LogLevel="DEBUG",
) -> Logger:
    colorama.just_fix_windows_console()

    class _formatter(logging.Formatter):
        def __init__(self, datefmt, rich_color: bool):
            fore = {
                "blue": Fore.BLUE if rich_color else "[",
                "green": Fore.GREEN if rich_color else "[",
                "yellow": Fore.YELLOW if rich_color else "[",
                "red": Fore.RED if rich_color else "[",
                "black": Fore.BLACK if rich_color else "[",
            }
            back = {
                "red": Back.RED if rich_color else "[",
                "yellow": Back.YELLOW if rich_color else "[",
            }
            style = {
                "bright": Style.BRIGHT if rich_color else "[",
                "reset_all": Style.RESET_ALL if rich_color else "]",
            }

            pathfmt = "%(module)s::%(funcName)s"
            fmt = "| %(asctime)s.%(msecs)03dZ LVL {bold}{pathfmt}{reset}: %(message)s".format(
                bold=style["bright"],
                pathfmt=pathfmt,
                reset=style["reset_all"],
            )
            formats = {
                logging.DEBUG: fmt.replace("LVL", fore["blue"] + "DEBUG" + style["reset_all"]),
                logging.INFO: fmt.replace("LVL", " " + fore["green"] + "INFO" + style["reset_all"]),
                logging.WARN: fmt.replace("LVL", " " + back["yellow"] + fore["black"] + "WARN" + style["reset_all"]),
                logging.ERROR: fmt.replace("LVL", back["red"] + fore["black"] + "ERROR" + style["reset_all"]),
                logging.CRITICAL: fmt.replace("LVL", " " + back["red"] + fore["black"] + style["bright"] + "CRIT" + style["reset_all"]),
            }
            self.formatters = {
                level: logging.Formatter(fmt=format, datefmt=datefmt)
                for level, format in formats.items()
            }

        def format(self, record):
            return self.formatters.get(record.levelno).format(record)

    datefmt = "%Y-%m-%dT%T"

    logger = Logger(name=name, level=level)
    logger.propagate = False

    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(_formatter(datefmt=datefmt, rich_color=True))
    logger.addHandler(ch)
    logger.setLevel(level)

    # file handler
    if file is not None:
        fh = logging.FileHandler(filename=file)
        fh.setLevel(file_level)
        fh.setFormatter(_formatter(datefmt=datefmt, rich_color=False))
        logger.addHandler(fh)
        def loglevel2int(log_level: LogLevel) -> int:
            return getattr(logging, log_level)
        logger.setLevel(min(loglevel2int(level), loglevel2int(file_level)))

    if with_tensorboard:
        tb = tensorboard.SummaryWriter(log_dir=file.parent, auto_flush=True)
        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="logger({})-".format(name),
        )
        logger.setup_tensorboard(tb=tb, executor=executor)

    # logger complains about `warn` being deprecated with another warning
    logger.warn = logger.warning
    return logger


def set_deterministic(seed: int) -> jran.KeyArray:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return jran.PRNGKey(seed)
