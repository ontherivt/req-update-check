# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`req-update-check` is a Python CLI tool that checks requirements.txt and pyproject.toml files for outdated packages. It queries PyPI to find available updates and reports version differences (major/minor/patch), with optional file caching for performance.

**NEW in v0.3.0**: AI-powered changelog analysis that:
- Fetches changelogs from GitHub releases, direct URLs, or package metadata
- Scans the codebase to find how packages are used
- Uses Claude, Gemini, OpenAI, or custom AI providers to analyze upgrade safety
- Provides actionable recommendations based on actual code usage
- Displays token usage for cost transparency

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

# AI-powered analysis (requires API key)
export ANTHROPIC_API_KEY="sk-ant-..."
req-update-check requirements.txt --ai-check requests

# Analyze all packages
req-update-check requirements.txt --ai-check

# Use different AI provider
export GEMINI_API_KEY="..."
req-update-check requirements.txt --ai-check --ai-provider gemini
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
  - **NEW**: Integrates AI analyzer for upgrade analysis
- `get_packages()`: Handles both requirements.txt (line-based) and pyproject.toml (TOML parsing with Python 3.11+ tomllib)
- `check_packages()`: Iterates through packages and compares versions
- `get_latest_version()`: Queries PyPI simple API, skips pre-releases, returns latest stable
- `get_package_info()`: Queries PyPI JSON API for metadata without requiring local installation
- `check_major_minor()`: Semantic version comparison logic
- `report()`: Outputs formatted update information with optional AI analysis

**`src/req_update_check/cache.py`** - File-based caching
- `FileCache` class: JSON file-based cache with TTL support
- **NEW**: Per-item TTL support (default 1 hour, AI cache 24 hours, changelog 7 days)
- Caches: PyPI package index, latest versions, package metadata, AI analysis, changelogs
- Stored in `~/.req-check-cache/` (or custom directory)

**`src/req_update_check/cli.py`** - Command-line interface
- Argument parsing with argparse
- **NEW**: AI analysis arguments (--ai-check, --ai-provider, --ai-model, --api-key)
- Wires together Requirements class and cache configuration

### AI Analysis Components (NEW in v0.3.0)

**`src/req_update_check/ai_providers/`** - AI provider implementations
- `base.py`: Abstract `AIProvider` class and `AnalysisResult` dataclass
  - Standardized interface for all AI providers
  - Token usage tracking
  - Retry logic with exponential backoff
  - Robust JSON parsing with fallback extraction
- `claude.py`: Anthropic Claude API provider (claude-3-5-sonnet-20241022)
- `gemini.py`: Google Gemini API provider (gemini-2.0-flash-exp)
- `openai.py`: OpenAI API provider (gpt-4o)
- `custom.py`: OpenAI-compatible custom/local providers (Ollama, etc.)
- `factory.py`: Provider factory for creating provider instances

**`src/req_update_check/ai_analyzer.py`** - Main analysis orchestrator
- `ChangelogAnalyzer` class: Coordinates the full analysis pipeline
  - Fetches changelog content
  - Scans codebase for package usage
  - Builds structured prompts
  - Sends to AI provider
  - Returns `AnalysisResult` with safety assessment
  - Manages caching with codebase state hash

**`src/req_update_check/changelog_fetcher.py`** - Changelog retrieval
- Fetches from multiple sources (priority order):
  1. Direct changelog URL
  2. GitHub releases API
  3. Fallback message
- Extracts relevant version range
- Truncates to ~15K chars for token management
- Caches for 7 days

**`src/req_update_check/code_scanner.py`** - Codebase analysis
- Scans Python files for package imports
- Handles both hyphenated and underscored package names
- Extracts usage examples (~10 lines context)
- Generates usage hash for cache invalidation
- Excludes common directories (.git, .venv, __pycache__, etc.)

**`src/req_update_check/prompts.py`** - Prompt engineering
- Builds structured analysis prompts
- Combines changelog, version info, and codebase usage
- System prompt defines expected JSON response format

**`src/req_update_check/formatting.py`** - Output formatting
- Formats `AnalysisResult` for terminal display
- Color-coded safety indicators (✅/⚠️/🚨)
- Shows breaking changes, deprecations, recommendations, new features
- **NEW**: Displays token usage (input/output/total)

**`src/req_update_check/auth.py`** - API key management
- Checks multiple sources in priority order:
  1. CLI argument (--api-key)
  2. Provider-specific env vars (ANTHROPIC_API_KEY, etc.)
  3. Config file (future enhancement)
- Validates key formats
- Provides helpful error messages with setup links

**`src/req_update_check/exceptions.py`** - Custom exceptions
- `AIAnalysisError`: Base exception for AI features
- `APIKeyNotFoundError`: Missing API key with helpful guidance
- `AIProviderError`: Provider-specific errors
- `ChangelogFetchError`: Non-fatal changelog issues

### Key Data Flow

**Basic Check:**
1. CLI parses args → creates Requirements instance
2. `get_packages()` parses input file (txt or toml)
3. `get_index()` fetches/caches PyPI package list
4. For each package: `get_latest_version()` queries PyPI (with cache)
5. `check_major_minor()` determines update severity
6. `report()` formats and displays results with metadata

**AI Analysis Flow (when --ai-check enabled):**
1. Requirements.report() identifies packages needing AI analysis
2. For each package:
   - `get_package_info()` fetches homepage/changelog URLs
   - `ChangelogAnalyzer.analyze_update()` orchestrates:
     - `ChangelogFetcher.fetch_changelog()` retrieves changelog from GitHub/URL
     - `CodebaseScanner.find_package_usage()` scans for imports and usage
     - `PromptBuilder.build_analysis_prompt()` creates structured prompt
     - AI provider analyzes and returns JSON response
     - Response parsed into `AnalysisResult` with token counts
   - Result cached for 24 hours (with codebase hash)
   - `format_ai_analysis()` displays formatted results

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
