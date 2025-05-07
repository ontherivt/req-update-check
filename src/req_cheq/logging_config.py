import logging
import sys


def setup_logging(level=logging.INFO):
    logger = logging.getLogger("req_update_check")
    logger.setLevel(level)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    formatter = logging.Formatter("%(message)s")
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger
