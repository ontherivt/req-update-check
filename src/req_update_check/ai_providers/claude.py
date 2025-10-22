from __future__ import annotations

import logging

from req_update_check.exceptions import AIProviderError

from .base import AIProvider
from .base import AnalysisResult

Anthropic = None
anthropic_import_error = None

try:
    from anthropic import Anthropic
except ImportError as e:
    anthropic_import_error = str(e)
except Exception as e:  # noqa: BLE001
    anthropic_import_error = f"Import failed: {type(e).__name__}: {e}"

logger = logging.getLogger("req_update_check")


class ClaudeProvider(AIProvider):
    """Anthropic Claude API provider"""

    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"

    def __init__(self, api_key: str, model: str | None = None):
        """
        Initialize Claude provider

        Args:
            api_key: Anthropic API key
            model: Optional model override (defaults to DEFAULT_MODEL)
        """
        if Anthropic is None:
            if anthropic_import_error and "No module named" in anthropic_import_error:
                msg = (
                    "anthropic package not installed. "
                    "Install with: pip install 'req-update-check[ai]' or pip install anthropic"
                )
            else:
                msg = (
                    f"anthropic package import failed: {anthropic_import_error}. "
                    "This may be due to a broken dependency. Try reinstalling: "
                    "pip install --force-reinstall anthropic"
                )
            raise AIProviderError(msg)

        self.client = Anthropic(api_key=api_key)
        self.model = model or self.DEFAULT_MODEL
        logger.debug(f"Initialized Claude provider with model: {self.model}")

    def analyze(self, prompt: str) -> AnalysisResult:
        """
        Send prompt to Claude and parse JSON response

        Args:
            prompt: The analysis prompt

        Returns:
            AnalysisResult with analysis

        Raises:
            AIProviderError: If API call fails
        """
        try:
            response = self._retry_with_backoff(
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    temperature=0,
                    system=self._get_system_prompt(),
                    messages=[{"role": "user", "content": prompt}],
                ),
            )

            # Extract text from response
            response_text = response.content[0].text
            logger.debug(f"Received response from Claude ({len(response_text)} chars)")

            # Extract token usage from response
            input_tokens = response.usage.input_tokens if hasattr(response, "usage") else 0
            output_tokens = response.usage.output_tokens if hasattr(response, "usage") else 0

            logger.debug(f"Token usage - Input: {input_tokens}, Output: {output_tokens}")

            # Parse into AnalysisResult
            return self._parse_response(response_text, input_tokens, output_tokens)

        except Exception as e:
            # Re-raise if already AIProviderError
            if isinstance(e, AIProviderError):
                raise

            # Wrap other exceptions
            msg = f"Claude API error: {e}"
            raise AIProviderError(msg) from e

    def get_model_name(self) -> str:
        """Return the model being used"""
        return self.model

    def estimate_cost(self, prompt_tokens: int) -> float:
        """
        Estimate cost for Claude 3.5 Sonnet

        Pricing as of 2024:
        - Input: $3.00 per million tokens
        - Output: $15.00 per million tokens

        Args:
            prompt_tokens: Estimated input token count

        Returns:
            Estimated cost in USD
        """
        # Claude 3.5 Sonnet pricing
        input_cost_per_mtok = 3.00
        output_cost_per_mtok = 15.00

        # Assume ~2000 output tokens for analysis
        estimated_output_tokens = 2000

        input_cost = (prompt_tokens / 1_000_000) * input_cost_per_mtok
        output_cost = (estimated_output_tokens / 1_000_000) * output_cost_per_mtok

        return input_cost + output_cost
