from __future__ import annotations

import logging
import sys
import time
from functools import partial, wraps
from pathlib import Path
from typing import Callable


def init_null_logger():
    """Initialize package root logger with NullHandler

    Configuring package root null logger for a library
    https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
    """
    pkg_root_name = __name__.split(".")[0]
    logger = logging.getLogger(pkg_root_name)
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.addHandler(logging.NullHandler())


def init_logger(
    *,
    quiet: bool = False,
    verbose: bool = False,
    log_file: str | Path | None = None,
):
    """Initialize package root logger with StreamHandler(& FileHandler)

    Configuring package root default logger for a CLI tool

    Parameters
    ----------
    quiet : bool, optional
        If True, no print info log on screen
    verbose: bool, optional
        If True & quiet=False, print debug log on screen
    log_file : str | Path | None, optional
        Log file
    """
    pkg_root_name = __name__.split(".")[0]
    logger = logging.getLogger(pkg_root_name)

    # Remove existing handler to avoid duplicate logging
    for handler in logger.handlers:
        logger.removeHandler(handler)
        handler.close()

    logger.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter(
        fmt="$asctime | $levelname | $message",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="$",
    )
    # Add stream handler for terminal stderr
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(log_formatter)
    if quiet:
        log_level = logging.WARNING
    else:
        log_level = logging.DEBUG if verbose else logging.INFO
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)

    if log_file:
        # Add file handler for log file
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(log_formatter)
        log_level = logging.DEBUG if verbose else logging.INFO
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)


def logging_timeit(
    func: Callable | None = None,
    /,
    *,
    msg: str = "Done",
    show_func_name: bool = False,
    debug: bool = False,
):
    """Elapsed time logging decorator

    e.g. `Done (elapsed time: 82.3[s]) [module.function]`

    Parameters
    ----------
    func : Callable | None, optional
        Target function
    msg : str, optional
        Logging message
    show_func_name : bool, optional
        If True, show elapsed time message with `module.function` definition
    debug : bool, optional
        If True, use `logger.debug` (By default `logger.info`)
    """
    if func is None:
        return partial(
            logging_timeit,
            msg=msg,
            show_func_name=show_func_name,
            debug=debug,
        )

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger = logging.getLogger(__name__)
        log = f"{msg} (elapsed time: {elapsed_time:.2f}[s])"
        if show_func_name:
            log = f"{log} [{func.__module__}.{func.__name__}]"
        logger_func = logger.debug if debug else logger.info
        logger_func(log)
        return result

    return wrapper
