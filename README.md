# req-update-check

[![Tests](https://github.com/ontherivt/req-update-check/actions/workflows/tests.yml/badge.svg)](https://github.com/ontherivt/req-update-check/actions/workflows/tests.yml)
[![Coverage Status](https://coveralls.io/repos/github/ontherivt/req-update-check/badge.svg?branch=main&t=unEUVF)](https://coveralls.io/github/ontherivt/req-update-check?branch=main)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)


A Python tool to check your requirements.txt file for package updates, with optional file caching for better performance.

## Features

- Check for available updates in your requirements.txt file
- Show update severity (major/minor/patch)
- Display package homepages and changelogs when available
- Optional file caching for faster repeated checks
- Support for comments and inline comments in requirements.txt
- Ignores pre-release versions (alpha, beta, release candidates)

## Installation

Install from PyPI:

```bash
pip install req-update-check
```

Or install from the repo directly:

```bash
pip install git+https://github.com/ontherivt/req-update-check.git
```

Or install from source:

```bash
git clone https://github.com/ontherivt/req-update-check.git
cd req-update-check
pip install -e .
```

## Usage

Basic usage:

```bash
req-update-check requirements.txt
```

### Command Line Options

```bash
req-update-check [-h] [--no-cache] [--cache-dir CACHE_DIR] requirements_file
```

Arguments:
- `requirements_file`: Path to your requirements.txt file

_Note: You can also provide a pyproject.toml file, but only if you're using python 3.11+._

Options:
- `--no-cache`: Disable file caching
- `--cache-dir`: Custom cache directory (default: `~/.req-update-check-cache`)

### Example Output

```
File caching enabled
The following packages need to be updated:

requests: 2.28.0 -> 2.31.0 [minor]
    Pypi page: https://pypi.python.org/project/requests/
    Homepage: https://requests.readthedocs.io
    Changelog: https://requests.readthedocs.io/en/latest/community/updates/#release-history

redis: 4.5.0 -> 5.0.1 [major]
    Pypi page: https://pypi.python.org/project/redis/
    Homepage: https://github.com/redis/redis-py
    Changelog: https://github.com/redis/redis-py/blob/master/CHANGES
```

### Using file Caching

The tool supports file caching to improve performance when checking multiple times. You can configure the cache storage:

```bash
req-update-check --cache-dir ~/.your-cache-dir requirements.txt
```

## Requirements.txt Format

The tool supports requirements.txt files with the following formats:
```
package==1.2.3
package == 1.2.3  # with spaces
package==1.2.3  # with inline comments
# Full line comments
```

Note: Currently only supports exact version specifiers (`==`). Support for other specifiers (like `>=`, `~=`) is planned for future releases.

## Python API

You can also use req-update-check as a Python library:

```python
from req_update_check import Requirements

# Without file cache
req = Requirements('requirements.txt', allow_cache=False)
req.check_packages()
req.report()

# With file cache defaults
req = Requirements('requirements.txt')
req.check_packages()
req.report()
```

## Development

To set up for development:

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment: `source venv/bin/activate` (Unix) or `venv\Scripts\activate` (Windows)
4. Install development dependencies: `pip install -e ".[dev]"`

To run tests:
1. `python -m unittest`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
