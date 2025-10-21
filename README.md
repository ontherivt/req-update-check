# req-update-check

[![Tests](https://github.com/ontherivt/req-update-check/actions/workflows/tests.yml/badge.svg)](https://github.com/ontherivt/req-update-check/actions/workflows/tests.yml)
[![Coverage Status](https://coveralls.io/repos/github/ontherivt/req-update-check/badge.svg?branch=main&t=unEUVF)](https://coveralls.io/github/ontherivt/req-update-check?branch=main)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)


A Python tool to check your requirements.txt file for package updates, with optional AI-powered changelog analysis and file caching for better performance.

## Features

- Check for available updates in your requirements.txt and pyproject.toml files
- Show update severity (major/minor/patch)
- Display package homepages and changelogs when available
- **AI-powered changelog analysis** - Analyze upgrade safety with Claude, Gemini, OpenAI, or custom AI providers
- **Codebase-aware recommendations** - AI scans your code to provide specific, actionable advice
- Optional file caching for faster repeated checks
- Support for comments and inline comments in requirements.txt
- Ignores pre-release versions (alpha, beta, release candidates)

## Installation

### Basic Installation

Install from PyPI:

```bash
pip install req-update-check
```

### Installation with AI Features

To use AI-powered analysis, install with AI providers:

```bash
# Install with all AI providers (Claude, Gemini, OpenAI)
pip install req-update-check[ai]

# Or install from source
git clone https://github.com/ontherivt/req-update-check.git
cd req-update-check
pip install -e ".[ai]"
```

## Usage

### Basic Usage

Check for updates without AI analysis:

```bash
req-update-check requirements.txt
```

### AI-Powered Analysis

Analyze upgrade safety with AI (requires API key):

```bash
# Analyze a specific package with Claude (default)
export ANTHROPIC_API_KEY="sk-ant-..."
req-update-check requirements.txt --ai-check requests

# Analyze all outdated packages
req-update-check requirements.txt --ai-check

# Use a different AI provider
export GEMINI_API_KEY="..."
req-update-check requirements.txt --ai-check --ai-provider gemini

# Use OpenAI
export OPENAI_API_KEY="sk-..."
req-update-check requirements.txt --ai-check --ai-provider openai
```

### Command Line Options

```bash
req-update-check [-h] [--no-cache] [--cache-dir CACHE_DIR]
                 [--ai-check [PACKAGE]] [--ai-provider {claude,gemini,openai,custom}]
                 [--ai-model MODEL] [--api-key API_KEY]
                 requirements_file
```

**Arguments:**
- `requirements_file`: Path to your requirements.txt or pyproject.toml file

_Note: pyproject.toml support requires Python 3.11+_

**General Options:**
- `--no-cache`: Disable file caching
- `--cache-dir CACHE_DIR`: Custom cache directory (default: `~/.req-check-cache`)

**AI Analysis Options:**
- `--ai-check [PACKAGE]`: Analyze updates with AI (optionally specify package name, or analyze all if omitted)
- `--ai-provider {claude,gemini,openai,custom}`: Choose AI provider (default: claude)
- `--ai-model MODEL`: Override default model for the provider
- `--api-key API_KEY`: Provide API key directly (or use environment variables)

### Example Output

**Basic output:**
```
File caching enabled
The following packages need to be updated:

requests: 2.28.0 -> 2.31.0 [minor]
    Pypi page: https://pypi.python.org/project/requests/
    Homepage: https://requests.readthedocs.io
    Changelog: https://requests.readthedocs.io/en/latest/community/updates/#release-history
```

**With AI analysis:**
```
File caching enabled
The following packages need to be updated:

requests: 2.28.0 -> 2.32.5 [minor]
    Pypi page: https://pypi.python.org/project/requests/
    Homepage: https://requests.readthedocs.io

    🤖 Analyzing with AI...

    AI ANALYSIS:
    ────────────────────────────────────────────────────────────
    ✅ Safety: SAFE (Confidence: high)
    Model: claude-3-5-sonnet-20241022
    Tokens: 8,245 in / 1,823 out / 10,068 total

    Recommendations:
      1. Review the changelog for security fixes in versions 2.29.0-2.32.0
      2. Test SSL certificate verification in your application
      3. Update request timeout handling if using default timeouts

    New Features:
      • Improved connection pooling performance
      • Better support for modern TLS versions
      • Enhanced cookie handling

    Summary: This is a safe minor version upgrade with important security
    fixes and performance improvements. No breaking changes detected in your
    current usage patterns.
    ────────────────────────────────────────────────────────────
```

## AI-Powered Analysis Features

### What Gets Analyzed

When you use `--ai-check`, the tool:

1. **Fetches changelogs** from GitHub releases, direct changelog URLs, or package metadata
2. **Scans your codebase** to find how you're using the package
3. **Sends to AI** with context about your usage patterns
4. **Returns analysis** with:
   - Safety assessment (safe/caution/breaking)
   - Breaking changes that affect your code
   - Deprecations in your current usage
   - Actionable upgrade recommendations
   - Relevant new features
   - Token usage statistics

### Supported AI Providers

| Provider | Model | Cost/Analysis* | Setup |
|----------|-------|----------------|-------|
| **Claude** (Anthropic) | claude-3-5-sonnet-20241022 | ~$0.05 | `export ANTHROPIC_API_KEY="sk-ant-..."` |
| **Gemini** (Google) | gemini-2.0-flash-exp | ~$0.01 | `export GEMINI_API_KEY="..."` |
| **OpenAI** | gpt-4o | ~$0.05 | `export OPENAI_API_KEY="sk-..."` |
| **Custom** | Your choice | Varies | Configure via config file |

*Estimated cost based on typical changelog and codebase size

### API Key Setup

**Option 1: Environment Variables (Recommended)**
```bash
# For Claude
export ANTHROPIC_API_KEY="sk-ant-..."

# For Gemini
export GEMINI_API_KEY="..."

# For OpenAI
export OPENAI_API_KEY="sk-..."
```

**Option 2: Command Line**
```bash
req-update-check requirements.txt --ai-check --api-key "your-key-here"
```

**Option 3: Config File** (Coming in Phase 4)
```toml
# ~/.config/req-update-check/config.toml
[ai.api_keys]
claude = "sk-ant-..."
gemini = "..."
```

### Caching

AI analysis results are cached for 24 hours to save on API costs. The cache is automatically invalidated when:
- Your codebase changes (files using the package are modified)
- 24 hours have passed
- You use `--no-cache`

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
from req_update_check.ai_providers import AIProviderFactory

# Basic usage without AI
req = Requirements('requirements.txt', allow_cache=False)
req.check_packages()
req.report()

# With AI analysis
provider = AIProviderFactory.create(
    provider_name='claude',
    api_key='sk-ant-...',  # or set ANTHROPIC_API_KEY env var
)

req = Requirements(
    'requirements.txt',
    ai_provider=provider,
)
req.check_packages()

# Analyze specific package
req.report(ai_check_packages=['requests'])

# Or analyze all packages
req.report(ai_check_packages=['*'])
```

## Development

To set up for development:

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment: `source venv/bin/activate` (Unix) or `venv\Scripts\activate` (Windows)
4. Install development dependencies: `pip install -e ".[dev,ai]"`

### Running Tests

```bash
# Run all tests
python -m unittest

# Run specific test file
python -m unittest tests.test_req_cheq

# Run with coverage
coverage run -m unittest discover
coverage report
coverage xml
```

### Code Quality

```bash
# Check code style
ruff check .

# Format code
ruff format .

# Auto-fix issues
ruff check --fix .
```

### Project Structure

```
src/req_update_check/
├── ai_providers/          # AI provider implementations
│   ├── base.py           # Abstract base class and AnalysisResult
│   ├── claude.py         # Claude (Anthropic) provider
│   ├── gemini.py         # Gemini (Google) provider
│   ├── openai.py         # OpenAI provider
│   ├── custom.py         # Custom/local provider
│   └── factory.py        # Provider factory
├── ai_analyzer.py        # Main analysis orchestrator
├── changelog_fetcher.py  # Fetch changelogs from various sources
├── code_scanner.py       # Scan codebase for package usage
├── prompts.py            # AI prompt templates
├── formatting.py         # Output formatting
├── auth.py               # API key management
├── cache.py              # File caching
├── core.py               # Main Requirements class
├── cli.py                # Command-line interface
└── exceptions.py         # Custom exceptions
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
