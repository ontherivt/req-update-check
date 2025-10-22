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


class OpenAIProvider(AIProvider):
    """OpenAI API provider (GPT-4, etc.)"""

    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, api_key: str, model: str | None = None):
        """
        Initialize OpenAI provider

        Args:
            api_key: OpenAI API key
            model: Optional model override (defaults to DEFAULT_MODEL)
        """
        if OpenAI is None:
            if openai_import_error and "No module named" in openai_import_error:
                msg = (
                    "openai package not installed. "
                    "Install with: pip install 'req-update-check[ai]' or pip install openai"
                )
            else:
                msg = (
                    f"openai package import failed: {openai_import_error}. "
                    "This may be due to a broken dependency. Try reinstalling: "
                    "pip install --force-reinstall openai"
                )
            raise AIProviderError(msg)

        self.client = OpenAI(api_key=api_key)
        self.model = model or self.DEFAULT_MODEL
        logger.debug(f"Initialized OpenAI provider with model: {self.model}")

    def analyze(self, prompt: str) -> AnalysisResult:
        """
        Send prompt to OpenAI and parse JSON response

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
                    response_format={"type": "json_object"},
                    temperature=0,
                ),
            )

            # Extract text from response
            response_text = response.choices[0].message.content
            logger.debug(f"Received response from OpenAI ({len(response_text)} chars)")

            # Extract token usage from response
            input_tokens = response.usage.prompt_tokens if hasattr(response, "usage") else 0
            output_tokens = response.usage.completion_tokens if hasattr(response, "usage") else 0

            logger.debug(f"Token usage - Input: {input_tokens}, Output: {output_tokens}")

            # Parse into AnalysisResult
            return self._parse_response(response_text, input_tokens, output_tokens)

        except Exception as e:
            # Re-raise if already AIProviderError
            if isinstance(e, AIProviderError):
                raise

            # Wrap other exceptions
            msg = f"OpenAI API error: {e}"
            raise AIProviderError(msg) from e

    def get_model_name(self) -> str:
        """Return the model being used"""
        return self.model

    def estimate_cost(self, prompt_tokens: int) -> float:
        """
        Estimate cost for OpenAI models

        Pricing as of 2024 (GPT-4o):
        - Input: $2.50 per million tokens
        - Output: $10.00 per million tokens

        Args:
            prompt_tokens: Estimated input token count

        Returns:
            Estimated cost in USD
        """
        # GPT-4o pricing (adjust if using different model)
        input_cost_per_mtok = 2.50
        output_cost_per_mtok = 10.00

        # Assume ~2000 output tokens for analysis
        estimated_output_tokens = 2000

        input_cost = (prompt_tokens / 1_000_000) * input_cost_per_mtok
        output_cost = (estimated_output_tokens / 1_000_000) * output_cost_per_mtok

        return input_cost + output_cost
