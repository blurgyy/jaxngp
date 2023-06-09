from concurrent.futures import Executor, Future, ThreadPoolExecutor
import functools
import logging
import logging
from pathlib import Path
import random
from typing import Any, Dict, Hashable, Iterable, Sequence, get_args

import colorama
from colorama import Back, Fore, Style
from flax.metrics import tensorboard
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


tqdm = functools.partial(tqdm_original, bar_format=tqdm_format)


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
