from __future__ import annotations

import os
import unittest

from src.req_update_check.auth import AIProviderAuth
from src.req_update_check.exceptions import APIKeyNotFoundError


class TestAIProviderAuth(unittest.TestCase):
    """Test API key management"""

    def setUp(self):
        self.auth = AIProviderAuth()
        # Clear any existing env vars
        for var in ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"]:
            if var in os.environ:
                del os.environ[var]

    def test_get_api_key_from_cli_argument(self):
        """CLI argument should take precedence"""
        key = self.auth.get_api_key("claude", cli_key="sk-ant-test123")

        self.assertEqual(key, "sk-ant-test123")

    def test_get_api_key_from_env_var(self):
        """Should get key from environment variable"""
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-env123"

        key = self.auth.get_api_key("claude")

        self.assertEqual(key, "sk-ant-env123")

    def test_get_api_key_alternative_env_var(self):
        """Should check alternative env var names"""
        os.environ["CLAUDE_API_KEY"] = "sk-ant-alt123"

        key = self.auth.get_api_key("claude")

        self.assertEqual(key, "sk-ant-alt123")

    def test_get_api_key_cli_overrides_env(self):
        """CLI argument should override environment variable"""
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-env123"

        key = self.auth.get_api_key("claude", cli_key="sk-ant-cli123")

        self.assertEqual(key, "sk-ant-cli123")

    def test_get_api_key_not_found(self):
        """Should raise error when no key found"""
        with self.assertRaises(APIKeyNotFoundError) as context:
            self.auth.get_api_key("claude")

        error = context.exception
        self.assertEqual(error.provider, "claude")
        self.assertIn("ANTHROPIC_API_KEY", str(error))
        self.assertIn("console.anthropic.com", str(error))

    def test_validate_key_format_claude_valid(self):
        """Should validate Claude key format"""
        self.assertTrue(self.auth._validate_key_format("claude", "sk-ant-abc123"))

    def test_validate_key_format_claude_invalid(self):
        """Should reject invalid Claude key format"""
        self.assertFalse(self.auth._validate_key_format("claude", "invalid-key"))

    def test_validate_key_format_empty(self):
        """Should reject empty keys"""
        self.assertFalse(self.auth._validate_key_format("claude", ""))
        self.assertFalse(self.auth._validate_key_format("claude", "   "))

    def test_validate_key_format_gemini(self):
        """Gemini has no strict prefix requirement"""
        self.assertTrue(self.auth._validate_key_format("gemini", "any-key-format"))

    def test_get_api_key_gemini(self):
        """Test getting Gemini API key"""
        os.environ["GEMINI_API_KEY"] = "gemini-test-key"

        key = self.auth.get_api_key("gemini")

        self.assertEqual(key, "gemini-test-key")


if __name__ == "__main__":
    unittest.main()
