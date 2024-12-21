import argparse

from .core import Requirements


def main():
    parser = argparse.ArgumentParser(description='Check Python package requirements for updates.')
    parser.add_argument('requirements_file', help='Path to the requirements.txt file')
    parser.add_argument('--no-cache', action='store_true', help='Disable file caching')
    parser.add_argument('--cache-dir', help='Custom cache directory (default: ~/.req-check-cache)')

    args = parser.parse_args()

    # Handle caching setup
    if not args.no_cache:
        print("File caching enabled")

    req = Requirements(
        args.requirements_file,
        allow_cache=not args.no_cache,
        cache_dir=args.cache_dir
    )
    req.check_packages()
    req.report()


if __name__ == "__main__":
    main()
