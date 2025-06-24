"""Logging configuration for the PTE-ECG package.

This module provides a configured logger instance and utility functions for setting log levels
and file handlers. The logger is configured to output to stdout by default with a standardized
format that includes the logger name, log level, and message.
"""

import logging
import pathlib
import sys

logger = logging.getLogger(__package__)
logger.setLevel(logging.INFO)
logger.propagate = False
_formatter = logging.Formatter("%(name)s | %(levelname)s | %(message)s")
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(_formatter)
logger.addHandler(_console_handler)


def set_log_level(level: int) -> None:
    """Set the logging level for the package logger.

    Args:
        level: The logging level to set (e.g., logging.INFO, logging.DEBUG).
    """
    old_level = logger.level
    if level != old_level:
        logger.setLevel(level)


def set_log_file(fname: str | pathlib.Path, overwrite: bool = False) -> None:
    """Add a file handler to the package logger.

    Args:
        fname: Path to the log file.
        overwrite: If True, overwrite the log file if it exists. If False, append to it.
    """
    if isinstance(fname, str):
        fname = pathlib.Path(fname)
    mode = "w"
    if fname.exists() and not overwrite:
        mode = "a"
    file_handler = logging.FileHandler(fname, mode=mode)
    file_handler.setFormatter(_formatter)
    logger.addHandler(file_handler)
