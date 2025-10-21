from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from .cache import FileCache

logger = logging.getLogger("req_update_check")


class ChangelogFetcher:
    """Fetches and parses changelog content from various sources"""

    def __init__(self, cache: FileCache | None = None):
        """
        Initialize changelog fetcher

        Args:
            cache: Optional cache for storing fetched changelogs
        """
        self.cache = cache
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "req-update-check/0.3.0 (Python changelog fetcher)"})

    def fetch_changelog(
        self,
        package_name: str,
        changelog_url: str | None,
        homepage_url: str | None,
        from_version: str,
        to_version: str,
    ) -> str:
        """
        Fetch changelog content, trying multiple strategies:
        1. Direct changelog URL (if provided)
        2. GitHub releases API (if GitHub repo)
        3. Fallback to homepage scraping

        Args:
            package_name: Name of the package
            changelog_url: Direct changelog URL if available
            homepage_url: Homepage/repository URL
            from_version: Current version
            to_version: Target version

        Returns:
            Changelog content or "Changelog unavailable" message
        """
        # Try cache first
        cache_key = f"changelog:{package_name}:{to_version}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                logger.debug(f"Using cached changelog for {package_name} {to_version}")
                return cached

        content = None

        # Strategy 1: Direct changelog URL
        if changelog_url:
            logger.debug(f"Fetching changelog from direct URL: {changelog_url}")
            content = self._fetch_url(changelog_url)

        # Strategy 2: GitHub releases
        if not content and homepage_url and "github.com" in homepage_url:
            logger.debug(f"Fetching GitHub releases for {package_name}")
            content = self._fetch_github_releases(homepage_url, from_version, to_version)

        # Extract relevant version range
        if content:
            content = self._extract_version_range(content, from_version, to_version)

        # Fallback message
        result = content or self._create_fallback_message(package_name, changelog_url, homepage_url)

        # Cache result
        if self.cache and content:
            # Cache for 7 days (604800 seconds)
            self.cache.set(cache_key, result, ttl=604800)

        return result

    def _fetch_url(self, url: str) -> str | None:
        """
        Fetch content from a URL

        Args:
            url: URL to fetch

        Returns:
            Content as string or None if fetch fails
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.text  # noqa: TRY300
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to fetch URL {url}: {e}")
            return None

    def _fetch_github_releases(self, repo_url: str, from_ver: str, to_ver: str) -> str | None:
        """
        Fetch releases from GitHub API

        Args:
            repo_url: GitHub repository URL
            from_ver: Starting version
            to_ver: Ending version

        Returns:
            Formatted release notes or None if fetch fails
        """
        # Parse owner/repo from URL
        match = re.search(r"github\.com/([^/]+)/([^/]+)", repo_url)
        if not match:
            logger.debug(f"Could not parse GitHub URL: {repo_url}")
            return None

        owner, repo = match.groups()
        repo = repo.rstrip("/").replace(".git", "")

        # Fetch releases from GitHub API
        api_url = f"https://api.github.com/repos/{owner}/{repo}/releases"
        try:
            response = self.session.get(api_url, timeout=10)
            response.raise_for_status()
            releases = response.json()

            if not releases:
                logger.debug(f"No releases found for {owner}/{repo}")
                return None

            # Filter to relevant versions
            relevant = []
            for release in releases:
                tag = release.get("tag_name", "").lstrip("v")
                if self._version_in_range(tag, from_ver, to_ver):
                    name = release.get("name") or tag
                    body = release.get("body") or "No release notes provided."
                    relevant.append(f"## {name}\n\n{body}")

            if relevant:
                logger.debug(f"Found {len(relevant)} relevant releases")
                return "\n\n---\n\n".join(relevant)

            logger.debug("No releases in version range")
            return None  # noqa: TRY300

        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to fetch GitHub releases: {e}")
            return None

    def _version_in_range(self, version: str, from_ver: str, to_ver: str) -> bool:
        """
        Check if version is in the range (from_ver, to_ver]

        Args:
            version: Version to check
            from_ver: Lower bound (exclusive)
            to_ver: Upper bound (inclusive)

        Returns:
            True if version is in range
        """
        try:
            # Simple version comparison - just normalize and compare strings
            # This is basic but works for most cases
            v_parts = self._normalize_version(version)
            from_parts = self._normalize_version(from_ver)
            to_parts = self._normalize_version(to_ver)

            # Version should be > from_ver and <= to_ver
            return v_parts > from_parts and v_parts <= to_parts  # noqa: TRY300
        except Exception:  # noqa: BLE001
            # If comparison fails, include it to be safe
            return True

    def _normalize_version(self, version: str) -> tuple:
        """
        Normalize version string to comparable tuple

        Args:
            version: Version string (e.g., "1.2.3")

        Returns:
            Tuple of integers for comparison
        """
        # Remove 'v' prefix if present
        version = version.lstrip("v")

        # Split on dots and convert to integers
        parts = []
        for part in version.split("."):
            # Extract numeric part (handle versions like "1.2.3rc1")
            match = re.match(r"(\d+)", part)
            if match:
                parts.append(int(match.group(1)))
            else:
                parts.append(0)

        # Pad to at least 3 parts
        while len(parts) < 3:
            parts.append(0)

        return tuple(parts)

    def _extract_version_range(self, content: str, from_ver: str, to_ver: str) -> str:
        """
        Extract only the relevant version range from full changelog
        Limits content to ~15000 chars to keep prompt size manageable

        Args:
            content: Full changelog content
            from_ver: Starting version
            to_ver: Ending version

        Returns:
            Extracted relevant portion
        """
        # For now, just truncate if too long
        # TODO: Implement smarter extraction based on version headers
        max_chars = 15000

        if len(content) <= max_chars:
            return content

        # Truncate and add note
        truncated = content[:max_chars]
        truncated += "\n\n... (changelog truncated for length)"

        return truncated

    def _create_fallback_message(
        self,
        package_name: str,
        changelog_url: str | None,
        homepage_url: str | None,
    ) -> str:
        """
        Create fallback message when changelog unavailable

        Args:
            package_name: Package name
            changelog_url: Changelog URL if available
            homepage_url: Homepage URL if available

        Returns:
            Fallback message
        """
        msg = f"Changelog unavailable for {package_name}."

        if changelog_url:
            msg += f"\nCheck: {changelog_url}"
        elif homepage_url:
            msg += f"\nCheck: {homepage_url}"

        msg += "\n\nNote: Analysis will be based on codebase usage only."

        return msg
