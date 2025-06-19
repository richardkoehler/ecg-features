import logging
import sys


def setup_logger(
    name: str = __name__, level: int = logging.DEBUG, log_to_file: str | None = None
) -> logging.Logger:
    """
    Sets up and returns a logger with console (and optional file) output.

    Args:
        name: The name of the logger (typically __name__).
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
        log_to_file: If provided, logs will also be written to this file.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Avoid adding multiple handlers if already configured
    if not logger.handlers:
        formatter = logging.Formatter("%(message)s")
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        if log_to_file:
            file_handler = logging.FileHandler(log_to_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        logger.propagate = False  # Prevent double logging if used in packages
    return logger


logger = setup_logger("ecg_features", level=logging.INFO, log_to_file=None)
