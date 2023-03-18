#!/usr/bin/env python3

import logging
import random
from typing import Any, Callable, Hashable, Iterable, Optional, Sequence, Tuple, Union

import chex
import colorama
from colorama import Back, Fore, Style
import jax
from jax._src.lib import xla_client as xc
import jax.random as jran
import numpy as np
import sympy
import tensorflow as tf


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
        name: str="main",
        level: str="INFO"
    ) -> Tuple[logging.Logger, Tuple[Callable, Callable, Callable, Callable, Callable]]:
    colorama.just_fix_windows_console()

    fn_names = ["debug", "info", "warning", "error", "critical"]

    class _formatter(logging.Formatter):
        def __init__(self, datefmt):
            pathfmt = "%(module)s::%(funcName)s"
            fmt = "| %(asctime)s.%(msecs)03dZ LVL {bold}{pathfmt}{reset}: %(message)s".format(
                bold=Style.BRIGHT,
                pathfmt=pathfmt,
                reset=Style.RESET_ALL,
            )
            formats = {
                logging.DEBUG: fmt.replace("LVL", Fore.BLUE + "DEBUG" + Style.RESET_ALL),
                logging.INFO: fmt.replace("LVL", " " + Fore.GREEN + "INFO" + Style.RESET_ALL),
                logging.WARN: fmt.replace("LVL", " " + Fore.YELLOW + "WARN" + Style.RESET_ALL),
                logging.ERROR: fmt.replace("LVL", Fore.RED + "ERROR" + Style.RESET_ALL),
                logging.CRITICAL: fmt.replace("LVL", " " + Back.RED + Fore.BLACK + Style.BRIGHT + "CRIT" + Style.RESET_ALL),
            }
            self.formatters = {
                level: logging.Formatter(fmt=format, datefmt=datefmt)
                for level, format in formats.items()
            }

        def format(self, record):
            return self.formatters.get(record.levelno).format(record)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(_formatter(datefmt="%Y-%m-%dT%T"))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(ch)
    logger.propagate = False
    return logger, map(lambda fn: getattr(logger, fn), fn_names)


def set_deterministic(seed: int) -> jran.KeyArray:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return jran.PRNGKey(seed)


def find_smallest_prime_larger_or_equal_than(x: int):
    chex.assert_type(x, int)

    if x <= 2:
        return 2

    if x & 1 == 0:
        x += 1
    while not sympy.isprime(x):
        x += 2

    return x
