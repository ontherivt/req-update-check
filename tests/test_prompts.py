from __future__ import annotations

import unittest
from unittest.mock import Mock

from req_update_check.prompts import PromptBuilder


class TestPromptBuilder(unittest.TestCase):
    """Tests for AI prompt building"""

    def test_build_analysis_prompt_basic(self):
        """Test building a basic analysis prompt"""
        mock_usage = Mock()
        mock_usage.to_prompt_text.return_value = "Found in 5 files"

        prompt = PromptBuilder.build_analysis_prompt(
            package_name="requests",
            current_version="2.0.0",
            latest_version="2.31.0",
            update_level="minor",
            changelog="## 2.31.0\n- Bug fixes\n- New features",
            usage_report=mock_usage,
        )

        self.assertIn("requests", prompt)
        self.assertIn("2.0.0", prompt)
        self.assertIn("2.31.0", prompt)
        self.assertIn("minor", prompt)
        self.assertIn("Bug fixes", prompt)
        self.assertIn("Found in 5 files", prompt)
        self.assertIn("CHANGELOG", prompt)
        self.assertIn("USAGE", prompt)

    def test_build_analysis_prompt_truncates_long_changelog(self):
        """Test that long changelogs are truncated"""
        mock_usage = Mock()
        mock_usage.to_prompt_text.return_value = "Usage info"

        long_changelog = "x" * 20000  # 20k characters

        prompt = PromptBuilder.build_analysis_prompt(
            package_name="flask",
            current_version="1.0.0",
            latest_version="2.0.0",
            update_level="major",
            changelog=long_changelog,
            usage_report=mock_usage,
        )

        self.assertLess(len(prompt), len(long_changelog) + 1000)
        self.assertIn("truncated", prompt)

    def test_build_analysis_prompt_short_changelog_not_truncated(self):
        """Test that short changelogs are not truncated"""
        mock_usage = Mock()
        mock_usage.to_prompt_text.return_value = "Usage info"

        short_changelog = "Short changelog"

        prompt = PromptBuilder.build_analysis_prompt(
            package_name="numpy",
            current_version="1.0.0",
            latest_version="1.1.0",
            update_level="patch",
            changelog=short_changelog,
            usage_report=mock_usage,
        )

        self.assertIn(short_changelog, prompt)
        self.assertNotIn("truncated", prompt)

    def test_get_system_prompt(self):
        """Test retrieving system prompt"""
        system_prompt = PromptBuilder.get_system_prompt()

        self.assertIn("expert software engineer", system_prompt)
        self.assertIn("JSON format", system_prompt)
        self.assertIn("breaking_changes", system_prompt)
        self.assertIn("deprecations", system_prompt)
        self.assertIn("recommendations", system_prompt)
        self.assertIn("new_features", system_prompt)
        self.assertIn("summary", system_prompt)
        self.assertIn("safe", system_prompt)
        self.assertIn("caution", system_prompt)
        self.assertIn("breaking", system_prompt)

    def test_prompt_includes_all_required_sections(self):
        """Test that prompt includes all required sections"""
        mock_usage = Mock()
        mock_usage.to_prompt_text.return_value = "Usage"

        prompt = PromptBuilder.build_analysis_prompt(
            package_name="test",
            current_version="1.0",
            latest_version="2.0",
            update_level="major",
            changelog="Changes",
            usage_report=mock_usage,
        )

        # Check for required sections
        self.assertIn("Package:", prompt)
        self.assertIn("Upgrade:", prompt)
        self.assertIn("Update Type:", prompt)
        self.assertIn("CHANGELOG", prompt)
        self.assertIn("USAGE", prompt)
        self.assertIn("safety", prompt)
        self.assertIn("recommendations", prompt)


if __name__ == "__main__":
    unittest.main()
