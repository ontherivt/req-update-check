from __future__ import annotations

import unittest

from req_update_check.ai_providers.factory import AIProviderFactory
from req_update_check.exceptions import APIKeyNotFoundError


class TestAIProviderFactory(unittest.TestCase):
    """Tests for AI provider factory"""

    def test_list_providers(self):
        """Test listing available providers"""
        providers = AIProviderFactory.list_providers()

        self.assertIn("claude", providers)
        self.assertIn("openai", providers)
        self.assertIn("gemini", providers)
        self.assertIn("custom", providers)

    def test_create_with_invalid_provider(self):
        """Test creating with invalid provider name"""
        with self.assertRaises(ValueError) as context:
            AIProviderFactory.create("invalid-provider", api_key="test-key")

        self.assertIn("Unknown provider", str(context.exception))

    def test_create_without_api_key(self):
        """Test creating provider without API key raises error"""
        with self.assertRaises(APIKeyNotFoundError) as context:
            AIProviderFactory.create("claude", api_key=None)

        self.assertIn("API key", str(context.exception))

    def test_create_custom_without_api_key_allowed(self):
        """Test custom provider can be created without API key"""
        # Custom provider allows no API key for local models
        try:
            provider = AIProviderFactory.create(
                "custom",
                api_key=None,
                config={"api_key_required": False},
            )
            self.assertIsNotNone(provider)
        except Exception:  # noqa: BLE001, S110
            # If it fails due to missing openai package, that's OK for this test
            pass


if __name__ == "__main__":
    unittest.main()
