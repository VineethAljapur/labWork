#!/usr/bin/env python3
# coding: utf-8

import sys
import logging


def setup_logger(logFile, debug=False):
    """
    Setup for the main Cichlids logger.

    Parameters
    ----------
    logFile : str
        name of log file to save
    debug : bool
        True to enable debug logging (default: False) and saving,
    """
    # The format of the logging string.
    logger = logging.getLogger(__name__)

    logger.setLevel(logging.INFO)
    if debug:
        logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(module)s %(processName)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    file_handler = logging.FileHandler(logFile)
    file_handler.setLevel(logging.INFO)
    if debug:
        file_handler.setLevel(logging.DEBUG)

    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


if __name__ == "__main__":
    pass
