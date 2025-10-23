from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

from req_update_check.changelog_fetcher import ChangelogFetcher


class TestChangelogFetcher(unittest.TestCase):
    """Tests for changelog fetching functionality"""

    def setUp(self):
        """Create fetcher instance"""
        self.fetcher = ChangelogFetcher(cache=None)

    def test_normalize_version(self):
        """Test version normalization"""
        self.assertEqual(self.fetcher._normalize_version("1.2.3"), (1, 2, 3))
        self.assertEqual(self.fetcher._normalize_version("v1.2.3"), (1, 2, 3))
        self.assertEqual(self.fetcher._normalize_version("2.0"), (2, 0, 0))
        self.assertEqual(self.fetcher._normalize_version("1.2.3rc1"), (1, 2, 3))

    def test_version_in_range(self):
        """Test version range checking"""
        self.assertTrue(self.fetcher._version_in_range("1.5.0", "1.0.0", "2.0.0"))
        self.assertTrue(self.fetcher._version_in_range("2.0.0", "1.0.0", "2.0.0"))  # Upper bound inclusive
        self.assertFalse(self.fetcher._version_in_range("1.0.0", "1.0.0", "2.0.0"))  # Lower bound exclusive
        self.assertFalse(self.fetcher._version_in_range("2.5.0", "1.0.0", "2.0.0"))

    def test_extract_version_range_short_content(self):
        """Test that short content is returned as-is"""
        content = "Short changelog content"
        result = self.fetcher._extract_version_range(content, "1.0.0", "2.0.0")
        self.assertEqual(result, content)

    def test_extract_version_range_long_content(self):
        """Test that long content is truncated"""
        content = "x" * 20000  # 20k characters
        result = self.fetcher._extract_version_range(content, "1.0.0", "2.0.0")
        self.assertLess(len(result), len(content))
        self.assertIn("truncated", result)

    def test_create_fallback_message(self):
        """Test fallback message creation"""
        msg = self.fetcher._create_fallback_message(
            "requests",
            changelog_url="https://example.com/changelog",
            homepage_url=None,
        )

        self.assertIn("requests", msg)
        self.assertIn("unavailable", msg)
        self.assertIn("https://example.com/changelog", msg)

    def test_create_fallback_message_with_homepage(self):
        """Test fallback with homepage only"""
        msg = self.fetcher._create_fallback_message(
            "flask",
            changelog_url=None,
            homepage_url="https://github.com/pallets/flask",
        )

        self.assertIn("flask", msg)
        self.assertIn("https://github.com/pallets/flask", msg)

    @patch("req_update_check.changelog_fetcher.requests.Session.get")
    def test_fetch_url_success(self, mock_get):
        """Test successful URL fetch"""
        mock_response = Mock()
        mock_response.text = "Changelog content"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = self.fetcher._fetch_url("https://example.com/changelog")

        self.assertEqual(result, "Changelog content")
        mock_get.assert_called_once()

    @patch("req_update_check.changelog_fetcher.requests.Session.get")
    def test_fetch_url_failure(self, mock_get):
        """Test URL fetch failure"""
        mock_get.side_effect = Exception("Network error")

        result = self.fetcher._fetch_url("https://example.com/changelog")

        self.assertIsNone(result)

    @patch("req_update_check.changelog_fetcher.requests.Session.get")
    def test_fetch_github_releases_success(self, mock_get):
        """Test fetching GitHub releases"""
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "tag_name": "v1.5.0",
                "name": "Release 1.5.0",
                "body": "Bug fixes and improvements",
            },
            {
                "tag_name": "v1.0.0",
                "name": "Release 1.0.0",
                "body": "Initial release",
            },
        ]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = self.fetcher._fetch_github_releases(
            "https://github.com/owner/repo",
            "1.0.0",
            "2.0.0",
        )

        self.assertIsNotNone(result)
        self.assertIn("1.5.0", result)
        self.assertIn("Bug fixes", result)

    @patch("req_update_check.changelog_fetcher.requests.Session.get")
    def test_fetch_github_releases_invalid_url(self, mock_get):
        """Test GitHub releases with invalid URL"""
        result = self.fetcher._fetch_github_releases(
            "https://example.com/not-github",
            "1.0.0",
            "2.0.0",
        )

        self.assertIsNone(result)
        mock_get.assert_not_called()

    def test_fetch_changelog_with_cache(self):
        """Test changelog fetch uses cache"""
        mock_cache = MagicMock()
        mock_cache.get.return_value = "Cached changelog"

        fetcher = ChangelogFetcher(cache=mock_cache)
        result = fetcher.fetch_changelog(
            package_name="requests",
            changelog_url=None,
            homepage_url=None,
            from_version="1.0.0",
            to_version="2.0.0",
        )

        self.assertEqual(result, "Cached changelog")
        mock_cache.get.assert_called_once()

    @patch("req_update_check.changelog_fetcher.requests.Session.get")
    def test_fetch_changelog_direct_url(self, mock_get):
        """Test fetching from direct changelog URL"""
        mock_response = Mock()
        mock_response.text = "Direct changelog"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = self.fetcher.fetch_changelog(
            package_name="requests",
            changelog_url="https://example.com/changelog",
            homepage_url=None,
            from_version="1.0.0",
            to_version="2.0.0",
        )

        self.assertIn("Direct changelog", result)

    def test_is_github_releases_page(self):
        """Test detection of GitHub releases pages"""
        # Should detect GitHub releases pages
        self.assertTrue(self.fetcher._is_github_releases_page("https://github.com/owner/repo/releases"))
        self.assertTrue(self.fetcher._is_github_releases_page("https://github.com/DataDog/dd-trace-py/releases"))

        # Should not detect non-releases GitHub pages
        self.assertFalse(self.fetcher._is_github_releases_page("https://github.com/owner/repo"))
        self.assertFalse(self.fetcher._is_github_releases_page("https://example.com/changelog"))
        self.assertFalse(self.fetcher._is_github_releases_page("https://readthedocs.io/changelog"))

    @patch("req_update_check.changelog_fetcher.requests.Session.get")
    def test_fetch_changelog_skips_github_releases_url(self, mock_get):
        """Test that GitHub releases URLs use API instead of direct fetch"""
        # Mock GitHub API response
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "tag_name": "v1.5.0",
                "name": "Release 1.5.0",
                "body": "Bug fixes from GitHub API",
            },
        ]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = self.fetcher.fetch_changelog(
            package_name="test-package",
            changelog_url="https://github.com/owner/repo/releases",  # This should be skipped
            homepage_url="https://github.com/owner/repo",
            from_version="1.0.0",
            to_version="2.0.0",
        )

        # Should use GitHub API, not direct URL fetch
        self.assertIn("Bug fixes from GitHub API", result)
        # Verify API endpoint was called (contains api.github.com), not the HTML releases page
        mock_get.assert_called_once()
        call_args = mock_get.call_args[0][0]
        self.assertIn("api.github.com", call_args)
        self.assertIn("/repos/", call_args)  # API path structure


if __name__ == "__main__":
    unittest.main()
