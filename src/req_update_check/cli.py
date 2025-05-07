from __future__ import annotations

import argparse
import logging

from .core import Requirements
from .logging_config import setup_logging

logger = logging.getLogger("req_update_check")


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Check Python package requirements for updates.",
    )
    parser.add_argument("requirements_file", help="Path to the requirements.txt file")
    parser.add_argument("--no-cache", action="store_true", help="Disable file caching")
    parser.add_argument(
        "--cache-dir",
        help="Custom cache directory (default: ~/.req-check-cache)",
    )

    args = parser.parse_args()

    # Handle caching setup
    if not args.no_cache:
        logger.info("File caching enabled")

    req = Requirements(
        args.requirements_file,
        allow_cache=not args.no_cache,
        cache_dir=args.cache_dir,
    )
    req.check_packages()
    req.report()


if __name__ == "__main__":
    main()
