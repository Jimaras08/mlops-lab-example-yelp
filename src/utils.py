"""
Utils for processing input data
"""

import logging


def setup_logging():
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel("INFO")
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
