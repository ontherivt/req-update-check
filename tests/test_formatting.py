from __future__ import annotations

import unittest

from req_update_check.ai_providers.base import AnalysisResult
from req_update_check.formatting import format_ai_analysis


class TestFormatting(unittest.TestCase):
    """Tests for AI analysis formatting"""

    def test_format_basic_analysis(self):
        """Test formatting a basic analysis result"""
        result = AnalysisResult(
            safety="safe",
            confidence="high",
            breaking_changes=[],
            deprecations=[],
            recommendations=["Update to latest version"],
            new_features=["New feature X"],
            summary="Safe to update",
            provider="claude",
            model="claude-3-5-sonnet",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )

        output = format_ai_analysis(result)

        self.assertIn("ANALYSIS", output)
        self.assertIn("✅", output)  # Safe icon
        self.assertIn("SAFE", output)
        self.assertIn("high", output)
        self.assertIn("claude-3-5-sonnet", output)
        self.assertIn("Update to latest version", output)
        self.assertIn("Safe to update", output)

    def test_format_breaking_changes(self):
        """Test formatting with breaking changes"""
        result = AnalysisResult(
            safety="breaking",
            confidence="high",
            breaking_changes=["API change in method X", "Removed deprecated function Y"],
            deprecations=["Old function Z"],
            recommendations=["Review breaking changes"],
            new_features=[],
            summary="Breaking changes detected",
            provider="openai",
            model="gpt-4o",
            input_tokens=200,
            output_tokens=100,
            total_tokens=300,
        )

        output = format_ai_analysis(result)

        self.assertIn("🚨", output)  # Breaking icon
        self.assertIn("BREAKING", output)
        self.assertIn("API change in method X", output)
        self.assertIn("Removed deprecated function Y", output)
        self.assertIn("Old function Z", output)

    def test_format_with_many_features(self):
        """Test formatting truncates long feature lists"""
        result = AnalysisResult(
            safety="safe",
            confidence="medium",
            breaking_changes=[],
            deprecations=[],
            recommendations=[],
            new_features=[f"Feature {i}" for i in range(10)],  # 10 features
            summary="Many new features",
            provider="gemini",
            model="gemini-2.0-flash",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
        )

        output = format_ai_analysis(result)

        # Should show first 3 features
        self.assertIn("Feature 0", output)
        self.assertIn("Feature 1", output)
        self.assertIn("Feature 2", output)
        # Should show "and X more"
        self.assertIn("and 7 more", output)

    def test_format_caution_level(self):
        """Test formatting caution safety level"""
        result = AnalysisResult(
            safety="caution",
            confidence="medium",
            breaking_changes=["Minor API change"],
            deprecations=[],
            recommendations=["Test thoroughly"],
            new_features=[],
            summary="Proceed with caution",
            provider="claude",
            model="claude-3-5-sonnet",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )

        output = format_ai_analysis(result)

        self.assertIn("⚠️", output)  # Caution icon
        self.assertIn("CAUTION", output)


if __name__ == "__main__":
    unittest.main()
