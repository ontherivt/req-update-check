from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .ai_providers.base import AnalysisResult
from .changelog_fetcher import ChangelogFetcher
from .code_scanner import CodebaseScanner
from .prompts import PromptBuilder

if TYPE_CHECKING:
    from .ai_providers.base import AIProvider
    from .cache import FileCache

logger = logging.getLogger("req_update_check")


class ChangelogAnalyzer:
    """Analyzes package updates using AI providers"""

    def __init__(
        self,
        provider: AIProvider,
        cache: FileCache | None,
        codebase_path: str = ".",
    ):
        """
        Initialize changelog analyzer

        Args:
            provider: AI provider instance (Claude, Gemini, etc.)
            cache: Optional cache for API responses
            codebase_path: Root path to scan for package usage
        """
        self.provider = provider
        self.cache = cache
        self.codebase_path = codebase_path

        # Initialize sub-components
        self.changelog_fetcher = ChangelogFetcher(cache)
        self.code_scanner = CodebaseScanner(codebase_path)

        logger.debug(f"Initialized ChangelogAnalyzer with {provider.__class__.__name__}")

    def analyze_update(  # noqa: PLR0913
        self,
        package_name: str,
        current_version: str,
        latest_version: str,
        update_level: str,
        changelog_url: str | None = None,
        homepage_url: str | None = None,
    ) -> AnalysisResult:
        """
        Orchestrates the full analysis pipeline:
        1. Fetch changelog content
        2. Search codebase for package usage
        3. Send to AI provider with structured prompt
        4. Parse and return structured result

        Args:
            package_name: Name of the package
            current_version: Current installed version
            latest_version: Latest available version
            update_level: Type of update (major/minor/patch)
            changelog_url: Optional direct changelog URL
            homepage_url: Optional homepage/repository URL

        Returns:
            AnalysisResult with safety assessment and recommendations

        Raises:
            AIProviderError: If AI analysis fails
        """
        logger.info(f"Analyzing {package_name} upgrade: {current_version} → {latest_version}")

        # Check cache first
        cache_key = self._get_cache_key(package_name, current_version, latest_version)
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                logger.debug(f"Using cached analysis for {package_name}")

                return AnalysisResult.from_dict(cached)

        # Step 1: Fetch changelog
        logger.debug("Fetching changelog...")
        changelog = self.changelog_fetcher.fetch_changelog(
            package_name=package_name,
            changelog_url=changelog_url,
            homepage_url=homepage_url,
            from_version=current_version,
            to_version=latest_version,
        )

        # Step 2: Scan codebase for usage
        logger.debug("Scanning codebase for package usage...")
        usage = self.code_scanner.find_package_usage(package_name)

        # Step 3: Build prompt and analyze
        logger.debug("Building analysis prompt...")
        prompt = PromptBuilder.build_analysis_prompt(
            package_name=package_name,
            current_version=current_version,
            latest_version=latest_version,
            update_level=update_level,
            changelog=changelog,
            usage_report=usage,
        )

        logger.debug("Sending to AI provider...")
        result = self.provider.analyze(prompt)

        logger.info(f"Analysis complete: {result.safety} (confidence: {result.confidence})")

        # Step 4: Cache result
        if self.cache:
            # Cache for 24 hours (86400 seconds)
            self.cache.set(cache_key, result.to_dict(), ttl=86400)

        return result

    def _get_cache_key(self, package: str, current: str, latest: str) -> str:
        """
        Generate cache key including codebase state

        Args:
            package: Package name
            current: Current version
            latest: Latest version

        Returns:
            Cache key string
        """
        # Include usage hash so cache invalidates when codebase changes
        usage_hash = self.code_scanner.get_usage_hash(package)
        return f"ai-analysis:{package}:{current}:{latest}:{usage_hash}"
