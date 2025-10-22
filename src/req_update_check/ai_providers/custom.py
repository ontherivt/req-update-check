from __future__ import annotations

import logging

from req_update_check.exceptions import AIProviderError

from .base import AIProvider
from .base import AnalysisResult

OpenAI = None
openai_import_error = None

try:
    from openai import OpenAI
except ImportError as e:
    openai_import_error = str(e)
except Exception as e:  # noqa: BLE001
    openai_import_error = f"Import failed: {type(e).__name__}: {e}"

logger = logging.getLogger("req_update_check")


class CustomProvider(AIProvider):
    """
    Custom provider for OpenAI-compatible APIs or user plugins

    Configuration via ~/.config/req-update-check/providers.toml:

    [providers.custom]
    name = "My Local LLM"
    base_url = "http://localhost:11434/v1"  # Ollama, etc.
    model = "codellama"
    api_key_required = false
    """

    def __init__(self, api_key: str | None = None, model: str | None = None, config: dict | None = None):
        """
        Initialize custom provider

        Args:
            api_key: Optional API key (may not be required for local models)
            model: Optional model override
            config: Custom configuration dict with base_url, etc.
        """
        if OpenAI is None:
            if openai_import_error and "No module named" in openai_import_error:
                msg = (
                    "openai package not installed (required for custom providers). "
                    "Install with: pip install 'req-update-check[ai]' or pip install openai"
                )
            else:
                msg = (
                    f"openai package import failed: {openai_import_error}. "
                    "This may be due to a broken dependency. Try reinstalling: "
                    "pip install --force-reinstall openai"
                )
            raise AIProviderError(msg)

        # Get configuration
        self.config = config or {}
        base_url = self.config.get("base_url", "http://localhost:11434/v1")
        self.model = model or self.config.get("model", "llama2")
        api_key_required = self.config.get("api_key_required", False)

        # Initialize client
        # For OpenAI-compatible APIs, we use the OpenAI client with a custom base_url
        if api_key_required and not api_key:
            msg = "API key required for custom provider but not provided"
            raise AIProviderError(msg)

        # OpenAI client requires an api_key, even for local models
        # Use a dummy key if not required
        effective_api_key = api_key or "not-needed"

        self.client = OpenAI(api_key=effective_api_key, base_url=base_url)

        logger.debug(f"Initialized custom provider with model: {self.model} at {base_url}")

    def analyze(self, prompt: str) -> AnalysisResult:
        """
        Send prompt to custom API and parse JSON response

        Args:
            prompt: The analysis prompt

        Returns:
            AnalysisResult with analysis

        Raises:
            AIProviderError: If API call fails
        """
        try:
            response = self._retry_with_backoff(
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    # Note: Not all custom APIs support response_format
                    # Some may need this removed
                ),
            )

            # Extract text from response
            response_text = response.choices[0].message.content
            logger.debug(f"Received response from custom provider ({len(response_text)} chars)")

            # Extract token usage from response (if available)
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, "usage") and response.usage:
                input_tokens = getattr(response.usage, "prompt_tokens", 0)
                output_tokens = getattr(response.usage, "completion_tokens", 0)

            logger.debug(f"Token usage - Input: {input_tokens}, Output: {output_tokens}")

            # Parse into AnalysisResult
            return self._parse_response(response_text, input_tokens, output_tokens)

        except Exception as e:
            # Re-raise if already AIProviderError
            if isinstance(e, AIProviderError):
                raise

            # Wrap other exceptions
            msg = f"Custom API error: {e}"
            raise AIProviderError(msg) from e

    def get_model_name(self) -> str:
        """Return the model being used"""
        return self.model

    def estimate_cost(self, prompt_tokens: int) -> float:
        """
        Estimate cost for custom provider

        For local models, cost is typically $0
        Override this if using a paid custom API

        Args:
            prompt_tokens: Estimated input token count

        Returns:
            Estimated cost in USD
        """
        # Local models are typically free
        return 0.0
