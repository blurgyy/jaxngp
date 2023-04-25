#!/usr/bin/env python3

import logging
from pathlib import Path
import random
from typing import (
    Any,
    Hashable,
    Iterable,
    Optional,
    Optional,
    Sequence,
    Union,
    get_args,
)

import colorama
from colorama import Back, Fore, Style
from flax.metrics import tensorboard
import jax
from jax._src.lib import xla_client as xc
import jax.random as jran
import numpy as np
import tensorflow as tf

from .types import LogLevel, Logger


_tqdm_format = "SBRIGHT{desc}RESET: HI{percentage:3.0f}%RESET {n_fmt}/{total_fmt} [{elapsed}<HI{remaining}RESET, {rate_fmt}]"
tqdm_format = _tqdm_format \
    .replace("HI", Fore.CYAN) \
    .replace("SBRIGHT", Style.BRIGHT) \
    .replace("RESET", Style.RESET_ALL)


# NOTE:
#   Jitting a vmapped function seems to give the desired performance boost, while vmapping a jitted
#   function might not work at all.  Except for the experiments I conducted myself, some related
#   issues:
# REF:
#   * <https://github.com/google/jax/issues/6312>
#   * <https://github.com/google/jax/issues/7449>
def vmap_jaxfn_with(
        # kwargs copied from `jax.vmap` source
        in_axes: Union[int, Sequence[Any]] = 0,
        out_axes: Any = 0,
        axis_name: Optional[Hashable] = None,
        axis_size: Optional[int] = None,
        spmd_axis_name: Optional[Hashable] = None,
    ):
    return lambda fn: jax.vmap(
            fn,
            in_axes=in_axes,
            out_axes=out_axes,
            axis_name=axis_name,
            axis_size=axis_size,
            spmd_axis_name=spmd_axis_name,
        )


def mkValueError(desc, value, type):
    variants = get_args(type)
    assert value not in variants
    return ValueError("Unexpected {}: '{}', expected one of [{}]".format(desc, value, "|".join(variants)))


def jit_jaxfn_with(
        # kwargs copied from `jax.jit` source
        static_argnums: Union[int, Iterable[int], None] = None,
        static_argnames: Union[str, Iterable[str], None] = None,
        device: Optional[xc.Device] = None,
        backend: Optional[str] = None,
        donate_argnums: Union[int, Iterable[int]] = (),
        inline: bool = False,
        keep_unused: bool = False,
        abstracted_axes: Optional[Any] = None,
    ):
    return lambda fn: jax.jit(
            fn,
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
    file: Optional[Union[str, Path]]=None,
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
                logging.WARN: fmt.replace("LVL", " " + fore["yellow"] + "WARN" + style["reset_all"]),
                logging.ERROR: fmt.replace("LVL", fore["red"] + "ERROR" + style["reset_all"]),
                logging.CRITICAL: fmt.replace("LVL", " " + back["red"] + fore["black"] + style["bright"] + "CRIT" + style["reset_all"]),
            }
            self.formatters = {
                level: logging.Formatter(fmt=format, datefmt=datefmt)
                for level, format in formats.items()
            }

        def format(self, record):
            return self.formatters.get(record.levelno).format(record)

    datefmt = "%Y-%m-%dT%T"

    logger = logging.getLogger(name)
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
        logger.tb = tb

    # logger complains about `warn` being deprecated with another warning
    logger.warn = logger.warning
    return logger


def setup_tensorboard(
    logs_dir: Union[str, Path],
) -> tensorboard.SummaryWriter:
    return tensorboard.SummaryWriter(logs_dir, auto_flush=True)


def set_deterministic(seed: int) -> jran.KeyArray:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return jran.PRNGKey(seed)
