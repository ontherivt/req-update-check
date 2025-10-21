# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`req-update-check` is a Python CLI tool that checks requirements.txt and pyproject.toml files for outdated packages. It queries PyPI to find available updates and reports version differences (major/minor/patch), with optional file caching for performance.

## Development Commands

### Setup
```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests
python -m unittest

# Run tests with coverage
coverage run -m unittest discover
coverage report
coverage xml

# Run a single test file
python -m unittest tests.test_req_cheq

# Run a specific test class or method
python -m unittest tests.test_req_cheq.TestRequirements.test_get_packages
```

### Linting
```bash
# Check code style and formatting
ruff check .
ruff format --check .

# Auto-fix issues
ruff check --fix .
ruff format .
```

### Running the Tool
```bash
# Basic usage
req-update-check requirements.txt

# With pyproject.toml (Python 3.11+ only)
req-update-check pyproject.toml

# Without cache
req-update-check --no-cache requirements.txt

# Custom cache directory
req-update-check --cache-dir /custom/path requirements.txt
```

## Architecture

### Core Components

**`src/req_update_check/core.py`** - Main logic
- `Requirements` class: Orchestrates the entire check process
  - Parses requirements.txt or pyproject.toml files
  - Queries PyPI simple API for package versions
  - Uses PyPI JSON API for package metadata (homepage, changelog)
  - Supports dependency-groups in pyproject.toml
  - Filters out pre-release versions (alpha, beta, rc)
- `get_packages()`: Handles both requirements.txt (line-based) and pyproject.toml (TOML parsing with Python 3.11+ tomllib)
- `check_packages()`: Iterates through packages and compares versions
- `get_latest_version()`: Queries PyPI simple API, skips pre-releases, returns latest stable
- `get_package_info()`: Queries PyPI JSON API for metadata without requiring local installation
- `check_major_minor()`: Semantic version comparison logic
- `report()`: Outputs formatted update information

**`src/req_update_check/cache.py`** - File-based caching
- `FileCache` class: JSON file-based cache with TTL (1 hour default)
- Caches: PyPI package index, latest versions, package metadata
- Stored in `~/.req-check-cache/` (or custom directory)

**`src/req_update_check/cli.py`** - Command-line interface
- Argument parsing with argparse
- Wires together Requirements class and cache configuration

### Key Data Flow
1. CLI parses args → creates Requirements instance
2. `get_packages()` parses input file (txt or toml)
3. `get_index()` fetches/caches PyPI package list
4. For each package: `get_latest_version()` queries PyPI (with cache)
5. `check_major_minor()` determines update severity
6. `report()` formats and displays results with metadata from `pip show`

### File Format Support
- **requirements.txt**: Line-based, supports `==` version pins, comments (`#`), inline comments
- **pyproject.toml**: Reads `project.dependencies` and `dependency-groups` (Python 3.11+ only via tomllib)
- Currently only `==` exact version specifiers are supported

## Configuration

### Ruff Settings (pyproject.toml)
- Line length: 120 characters
- Force single-line imports (`isort` configuration)
- Extensive rule set enabled (F, E, W, I, N, UP, S, B, etc.)
- Notable ignores: S101 (assert), FBT001/002 (boolean traps), PTH123 (path operations)

### Test Configuration
- Uses Python's built-in `unittest` framework
- Coverage includes `src/**` files
- Tests mock file I/O and HTTP requests
- Python 3.11+ specific tests skipped on older versions

## Version Support
- Python 3.9+ required
- pyproject.toml parsing requires Python 3.11+ (tomllib)
