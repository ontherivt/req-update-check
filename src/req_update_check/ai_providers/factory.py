from __future__ import annotations

import logging
import typing

from req_update_check.auth import AIProviderAuth

from .claude import ClaudeProvider
from .custom import CustomProvider
from .gemini import GeminiProvider
from .openai import OpenAIProvider

logger = logging.getLogger("req_update_check")

if typing.TYPE_CHECKING:
    from .base import AIProvider


class AIProviderFactory:
    """Factory to create AI provider instances"""

    # Registry of available providers
    PROVIDERS = {
        "claude": ClaudeProvider,
        "gemini": GeminiProvider,
        "openai": OpenAIProvider,
        "custom": CustomProvider,
    }

    @classmethod
    def create(
        cls,
        provider_name: str,
        api_key: str | None = None,
        model: str | None = None,
        config: dict | None = None,
    ) -> AIProvider:
        """
        Create provider instance with proper authentication

        Args:
            provider_name: Name of provider (claude, gemini, etc.)
            api_key: Optional API key override
            model: Optional model override
            config: Optional custom provider config

        Returns:
            AIProvider instance

        Raises:
            ValueError: If provider not found
            APIKeyNotFoundError: If API key not configured
        """
        # Validate provider exists
        if provider_name not in cls.PROVIDERS:
            available = ", ".join(cls.PROVIDERS.keys())
            msg = f"Unknown provider: {provider_name}. Available: {available}"
            raise ValueError(msg)

        provider_class = cls.PROVIDERS[provider_name]

        logger.debug(f"Creating {provider_name} provider")

        # Handle custom provider separately (may not need API key)
        if provider_name == "custom":
            # For custom provider, API key may not be required
            # Config can be passed in or loaded from config file
            return provider_class(api_key=api_key, model=model, config=config)

        # For standard providers, get API key
        auth = AIProviderAuth()
        resolved_api_key = auth.get_api_key(provider_name, api_key)

        # Create instance with API key and model
        return provider_class(resolved_api_key, model)

    @classmethod
    def list_providers(cls) -> list[str]:
        """
        Get list of available provider names

        Returns:
            List of provider names
        """
        return list(cls.PROVIDERS.keys())
