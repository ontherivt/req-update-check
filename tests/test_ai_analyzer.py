from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

from req_update_check.ai_analyzer import ChangelogAnalyzer
from req_update_check.ai_providers.base import AnalysisResult


class TestChangelogAnalyzer(unittest.TestCase):
    """Tests for AI-powered changelog analysis"""

    def setUp(self):
        """Create mock provider and analyzer"""
        self.mock_provider = Mock()
        self.mock_provider.get_model_name.return_value = "test-model"
        self.mock_provider.__class__.__name__ = "MockProvider"

        self.mock_cache = MagicMock()
        self.analyzer = ChangelogAnalyzer(
            provider=self.mock_provider,
            cache=self.mock_cache,
            codebase_path=".",
        )

    def test_get_cache_key(self):
        """Test cache key generation"""
        with patch.object(self.analyzer.code_scanner, "get_usage_hash", return_value="abc123"):
            key = self.analyzer._get_cache_key("requests", "1.0.0", "2.0.0")

            self.assertIn("requests", key)
            self.assertIn("1.0.0", key)
            self.assertIn("2.0.0", key)
            self.assertIn("abc123", key)

    def test_analyze_update_with_cache_hit(self):
        """Test analysis uses cached result"""
        cached_result = {
            "safety": "safe",
            "confidence": "high",
            "breaking_changes": [],
            "deprecations": [],
            "recommendations": [],
            "new_features": [],
            "summary": "Cached result",
            "provider": "test",
            "model": "test-model",
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }

        self.mock_cache.get.return_value = cached_result

        result = self.analyzer.analyze_update(
            package_name="requests",
            current_version="1.0.0",
            latest_version="2.0.0",
            update_level="major",
        )

        self.assertEqual(result.summary, "Cached result")
        self.mock_cache.get.assert_called_once()
        self.mock_provider.analyze.assert_not_called()

    def test_analyze_update_no_cache(self):
        """Test analysis without cache"""
        self.mock_cache.get.return_value = None

        mock_result = AnalysisResult(
            safety="safe",
            confidence="high",
            breaking_changes=[],
            deprecations=[],
            recommendations=["Update recommended"],
            new_features=["New feature X"],
            summary="Analysis complete",
            provider="test",
            model="test-model",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )

        self.mock_provider.analyze.return_value = mock_result

        with (
            patch.object(self.analyzer.changelog_fetcher, "fetch_changelog", return_value="Changelog"),
            patch.object(self.analyzer.code_scanner, "find_package_usage") as mock_usage,
        ):
            mock_usage.return_value = Mock(
                package_name="requests",
                files_count=5,
                import_locations=[],
                usage_examples=[],
            )
            mock_usage.return_value.to_prompt_text = Mock(return_value="Usage info")

            result = self.analyzer.analyze_update(
                package_name="requests",
                current_version="1.0.0",
                latest_version="2.0.0",
                update_level="major",
            )

        self.assertEqual(result.summary, "Analysis complete")
        self.mock_provider.analyze.assert_called_once()
        self.mock_cache.set.assert_called_once()

    def test_analyze_update_builds_prompt(self):
        """Test that analysis builds proper prompt"""
        self.mock_cache.get.return_value = None

        mock_result = AnalysisResult(
            safety="safe",
            confidence="high",
            breaking_changes=[],
            deprecations=[],
            recommendations=[],
            new_features=[],
            summary="Test",
            provider="test",
            model="test-model",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
        )

        self.mock_provider.analyze.return_value = mock_result

        with (
            patch.object(self.analyzer.changelog_fetcher, "fetch_changelog", return_value="Test changelog"),
            patch.object(self.analyzer.code_scanner, "find_package_usage") as mock_usage,
        ):
            mock_report = Mock()
            mock_report.to_prompt_text.return_value = "Usage report"
            mock_usage.return_value = mock_report

            self.analyzer.analyze_update(
                package_name="flask",
                current_version="1.0.0",
                latest_version="2.0.0",
                update_level="major",
                changelog_url="https://example.com/changelog",
                homepage_url="https://example.com",
            )

        # Verify provider.analyze was called with a string prompt
        call_args = self.mock_provider.analyze.call_args
        self.assertEqual(len(call_args[0]), 1)  # One positional arg
        prompt = call_args[0][0]
        self.assertIsInstance(prompt, str)
        self.assertIn("flask", prompt)


if __name__ == "__main__":
    unittest.main()
