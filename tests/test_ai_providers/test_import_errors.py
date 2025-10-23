from __future__ import annotations

import importlib
import unittest
from unittest.mock import patch

from req_update_check.exceptions import AIProviderError

try:
    import anthropic  # noqa: F401
    import google.generativeai  # noqa: F401
    import openai  # noqa: F401

    optional_imports_installed = True
except ImportError:
    optional_imports_installed = False

from req_update_check.ai_providers import ClaudeProvider
from req_update_check.ai_providers import GeminiProvider
from req_update_check.ai_providers import OpenAIProvider


@unittest.skipIf(optional_imports_installed, "tests skipped for environments with optional dependencies")
class TestAIProviderImportErrors(unittest.TestCase):
    """Tests for AI provider import error handling when optional dependencies are not installed"""

    def setUp(self):
        """Clean up any cached imports before each test"""
        # Remove AI provider modules from cache to force re-import
        self.import_module_patcher = patch.object(importlib, "import_module")
        self.mocked_import_module = self.import_module_patcher.start()

    def test_claude_provider_missing_anthropic_package(self):
        """Test ClaudeProvider raises proper error when anthropic package is missing"""
        # Mock the import to raise ImportError with "No module named" message
        self.mocked_import_module.side_effect = Exception("My random exception that is not a failure")

        with self.assertRaises(AIProviderError) as context:
            ClaudeProvider(api_key="test-key")

        error_msg = str(context.exception)
        self.assertIn("anthropic package not installed", error_msg)
        self.assertIn("pip install 'req-update-check[ai]'", error_msg)
        self.assertIn("pip install anthropic", error_msg)

    def test_openai_provider_missing_openai_package(self):
        """Test OpenAIProvider raises proper error when openai package is missing"""
        with self.assertRaises(AIProviderError) as context:
            OpenAIProvider(api_key="test-key")

        error_msg = str(context.exception)
        self.assertIn("openai package not installed", error_msg)
        self.assertIn("pip install 'req-update-check[ai]'", error_msg)
        self.assertIn("pip install openai", error_msg)

    def test_gemini_provider_missing_google_generativeai_package(self):
        """Test GeminiProvider raises proper error when google-generativeai package is missing"""
        with self.assertRaises(AIProviderError) as context:
            GeminiProvider(api_key="test-key")

        error_msg = str(context.exception)
        self.assertIn("google-generativeai package not installed", error_msg)
        self.assertIn("pip install 'req-update-check[ai]'", error_msg)
        self.assertIn("pip install google-generativeai", error_msg)
