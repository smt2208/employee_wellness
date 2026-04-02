# utils/logger.py
# Provides a single shared logger for the entire pipeline.
# Import get_logger() in any module that needs logging.

import logging
import sys
from config.settings import LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance.

    Parameters:
        name: Usually __name__ from the calling module.

    Returns:
        A Logger object writing to stdout with timestamp + level prefix.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if this is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
