from __future__ import annotations

import json
import unittest

from req_update_check.ai_providers.base import AIProvider
from req_update_check.ai_providers.base import AnalysisResult
from req_update_check.exceptions import AIProviderError


class MockProvider(AIProvider):
    """Mock provider for testing"""

    def __init__(self, response_data: dict | None = None):
        self.response_data = response_data or self._get_default_response()
        self.called = False

    def analyze(self, prompt: str) -> AnalysisResult:
        self.called = True
        return self._parse_response(json.dumps(self.response_data))

    def get_model_name(self) -> str:
        return "mock-model-1.0"

    def estimate_cost(self, prompt_tokens: int) -> float:
        return 0.01

    @staticmethod
    def _get_default_response() -> dict:
        return {
            "safety": "safe",
            "confidence": "high",
            "breaking_changes": [],
            "deprecations": [],
            "recommendations": ["Test before production"],
            "new_features": ["Performance improvements"],
            "summary": "Safe upgrade with performance benefits.",
        }


class TestAnalysisResult(unittest.TestCase):
    """Test AnalysisResult dataclass"""

    def test_to_dict(self):
        result = AnalysisResult(
            safety="safe",
            confidence="high",
            breaking_changes=[],
            deprecations=[],
            recommendations=["Update"],
            new_features=["Feature 1"],
            summary="Test summary",
            provider="claude",
            model="claude-3-5-sonnet-20241022",
        )

        data = result.to_dict()

        self.assertEqual(data["safety"], "safe")
        self.assertEqual(data["provider"], "claude")
        self.assertEqual(data["new_features"], ["Feature 1"])

    def test_from_dict(self):
        data = {
            "safety": "caution",
            "confidence": "medium",
            "breaking_changes": ["Breaking change 1"],
            "deprecations": [],
            "recommendations": ["Review code"],
            "new_features": [],
            "summary": "Caution needed",
            "provider": "gemini",
            "model": "gemini-2.0-flash",
        }

        result = AnalysisResult.from_dict(data)

        self.assertEqual(result.safety, "caution")
        self.assertEqual(result.confidence, "medium")
        self.assertEqual(result.breaking_changes, ["Breaking change 1"])


class TestAIProvider(unittest.TestCase):
    """Test AIProvider base class"""

    def test_parse_response_success(self):
        provider = MockProvider()
        response_json = json.dumps(provider.response_data)

        result = provider._parse_response(response_json)

        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual(result.safety, "safe")
        self.assertEqual(result.confidence, "high")
        self.assertEqual(result.provider, "mock")
        self.assertEqual(result.model, "mock-model-1.0")

    def test_parse_response_invalid_json(self):
        provider = MockProvider()

        with self.assertRaises(AIProviderError) as context:
            provider._parse_response("not valid json")

        self.assertIn("Failed to parse", str(context.exception))

    def test_parse_response_missing_field(self):
        provider = MockProvider()
        incomplete_data = {"safety": "safe", "confidence": "high"}  # Missing required fields

        with self.assertRaises(AIProviderError) as context:
            provider._parse_response(json.dumps(incomplete_data))

        self.assertIn("Missing required field", str(context.exception))

    def test_get_system_prompt(self):
        provider = MockProvider()
        prompt = provider._get_system_prompt()

        self.assertIn("expert software engineer", prompt)
        self.assertIn("JSON format", prompt)
        self.assertIn("breaking changes", prompt)

    def test_retry_with_backoff_success(self):
        provider = MockProvider()
        call_count = 0

        def func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = provider._retry_with_backoff(func, max_retries=3)

        self.assertEqual(result, "success")
        self.assertEqual(call_count, 1)

    def test_retry_with_backoff_eventual_success(self):
        provider = MockProvider()
        call_count = 0

        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary error")
            return "success"

        result = provider._retry_with_backoff(func, max_retries=5)

        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)

    def test_retry_with_backoff_all_fail(self):
        provider = MockProvider()

        def func():
            raise ConnectionError("Persistent error")

        with self.assertRaises(AIProviderError) as context:
            provider._retry_with_backoff(func, max_retries=3)

        self.assertIn("Failed after 3 retries", str(context.exception))


if __name__ == "__main__":
    unittest.main()
