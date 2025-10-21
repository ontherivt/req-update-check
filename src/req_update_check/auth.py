from __future__ import annotations

import os
from pathlib import Path

from .exceptions import APIKeyNotFoundError


class AIProviderAuth:
    """Handles API authentication for various AI providers"""

    # Map provider names to environment variable names (in order of preference)
    ENV_VAR_MAP = {
        "claude": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
        "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
        "custom": ["AI_API_KEY"],
    }

    # Key format validators (optional, for better error messages)
    KEY_PREFIXES = {
        "claude": "sk-ant-",
        "openai": "sk-",
        # Gemini and custom don't have strict prefixes
    }

    def get_api_key(self, provider: str, cli_key: str | None = None) -> str:
        """
        Get API key for provider, checking multiple sources

        Args:
            provider: Provider name (claude, gemini, openai, custom)
            cli_key: Optional API key from CLI argument

        Returns:
            API key string

        Raises:
            APIKeyNotFoundError: If no valid key found
        """
        # 1. Check CLI argument first
        if cli_key:
            if self._validate_key_format(provider, cli_key):
                return cli_key
            # If format invalid, continue to env vars in case user made mistake

        # 2. Check environment variables
        env_vars = self.ENV_VAR_MAP.get(provider, [])
        for env_var in env_vars:
            key = os.getenv(env_var)
            if key:
                if self._validate_key_format(provider, key):
                    return key

        # 3. Check config file (if implemented in future)
        config_key = self._get_key_from_config(provider)
        if config_key:
            if self._validate_key_format(provider, config_key):
                return config_key

        # No key found - raise helpful error
        raise APIKeyNotFoundError(provider, env_vars)

    def _validate_key_format(self, provider: str, key: str) -> bool:
        """
        Validate API key format for provider

        Args:
            provider: Provider name
            key: API key to validate

        Returns:
            True if valid (or no validation rule), False if invalid
        """
        if not key or not key.strip():
            return False

        # Check provider-specific prefix if defined
        if provider in self.KEY_PREFIXES:
            expected_prefix = self.KEY_PREFIXES[provider]
            if not key.startswith(expected_prefix):
                return False

        return True

    def _get_key_from_config(self, provider: str) -> str | None:
        """
        Get API key from config file (future enhancement)

        Args:
            provider: Provider name

        Returns:
            API key if found in config, None otherwise
        """
        config_path = Path.home() / ".config" / "req-update-check" / "config.toml"

        if not config_path.exists():
            return None

        # TODO: Implement TOML parsing when config support is added
        # For now, return None
        return None
