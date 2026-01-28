from __future__ import annotations

import argparse
import json
import logging
import sys

from .ai_providers import AIProviderFactory
from .core import Requirements
from .exceptions import APIKeyNotFoundError
from .logging_config import setup_logging

logger = logging.getLogger("req_update_check")


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Check Python package requirements for updates.",
    )
    parser.add_argument("requirements_file", help="Path to the requirements.txt or pyproject.toml file")
    parser.add_argument("--no-cache", action="store_true", help="Disable file caching")
    parser.add_argument(
        "--cache-dir",
        help="Custom cache directory (default: ~/.req-check-cache)",
    )

    # AI Analysis arguments
    ai_group = parser.add_argument_group("AI Analysis")
    ai_group.add_argument(
        "--ai-check",
        nargs="?",
        const="*",
        metavar="PACKAGE",
        help="Analyze updates with AI. Provide package name or omit for all packages.",
    )
    ai_group.add_argument(
        "--ai-provider",
        choices=AIProviderFactory.list_providers(),
        default="claude",
        help="AI provider to use (default: claude).",
    )
    ai_group.add_argument(
        "--ai-model",
        help="Override default AI model for the provider",
    )
    ai_group.add_argument(
        "--api-key",
        help="API key for AI provider (or set ANTHROPIC_API_KEY env var)",
    )

    # Output format
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    args = parser.parse_args()

    # Suppress logging early when JSON output requested to avoid contaminating stdout
    json_output = args.output == "json"
    if json_output:
        logging.getLogger("req_update_check").setLevel(logging.CRITICAL)

    # Determine AI check mode
    ai_provider = None
    ai_check_packages = None

    if args.ai_check:
        # Initialize AI provider
        try:
            ai_provider = AIProviderFactory.create(
                provider_name=args.ai_provider,
                api_key=args.api_key,
                model=args.ai_model,
            )
            logger.info(f"AI analysis enabled using {args.ai_provider} ({ai_provider.get_model_name()})")

            # Determine which packages to check
            ai_check_packages = ["*"] if args.ai_check == "*" else [args.ai_check]

        except APIKeyNotFoundError:
            logger.exception("APIKeyNotFoundError")
            sys.exit(1)
        except Exception:
            logger.exception("Failed to initialize AI provider")
            sys.exit(1)

    # Handle caching setup
    if not args.no_cache and not json_output:
        logger.info("File caching enabled")

    req = Requirements(
        args.requirements_file,
        allow_cache=not args.no_cache,
        cache_dir=args.cache_dir,
        ai_provider=ai_provider,
    )
    req.check_packages()
    result = req.report(ai_check_packages=ai_check_packages, output_format=args.output)

    # result is always a valid dict when json_output=True (never empty/falsy)
    if json_output:
        print(json.dumps(result, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
