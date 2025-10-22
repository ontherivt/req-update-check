from __future__ import annotations

import logging

from req_update_check.exceptions import AIProviderError

from .base import AIProvider
from .base import AnalysisResult

genai = None
genai_import_error = None

try:
    import google.generativeai as genai
except ImportError as e:
    genai_import_error = str(e)
except Exception as e:  # noqa: BLE001
    genai_import_error = f"Import failed: {type(e).__name__}: {e}"

logger = logging.getLogger("req_update_check")


class GeminiProvider(AIProvider):
    """Google Gemini API provider"""

    DEFAULT_MODEL = "gemini-2.0-flash-exp"

    def __init__(self, api_key: str, model: str | None = None):
        """
        Initialize Gemini provider

        Args:
            api_key: Google API key
            model: Optional model override (defaults to DEFAULT_MODEL)
        """
        if genai is None:
            if genai_import_error and "No module named" in genai_import_error:
                msg = (
                    "google-generativeai package not installed. "
                    "Install with: pip install 'req-update-check[ai]' or pip install google-generativeai"
                )
            else:
                msg = (
                    f"google-generativeai package import failed: {genai_import_error}. "
                    "This may be due to a broken dependency. Try reinstalling: "
                    "pip install --force-reinstall google-generativeai"
                )
            raise AIProviderError(msg)

        genai.configure(api_key=api_key)
        self.model_name = model or self.DEFAULT_MODEL

        # Create model with JSON response format
        self.client = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": 0,
                "response_mime_type": "application/json",
            },
        )

        logger.debug(f"Initialized Gemini provider with model: {self.model_name}")

    def analyze(self, prompt: str) -> AnalysisResult:
        """
        Send prompt to Gemini and parse JSON response

        Args:
            prompt: The analysis prompt

        Returns:
            AnalysisResult with analysis

        Raises:
            AIProviderError: If API call fails
        """
        try:
            # Combine system prompt with user prompt for Gemini
            # (Gemini doesn't have separate system/user messages in the same way)
            full_prompt = f"{self._get_system_prompt()}\n\n{prompt}"

            response = self._retry_with_backoff(lambda: self.client.generate_content(full_prompt))

            # Extract text from response
            response_text = response.text
            logger.debug(f"Received response from Gemini ({len(response_text)} chars)")

            # Extract token usage from response
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, "usage_metadata"):
                input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
                output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)

            logger.debug(f"Token usage - Input: {input_tokens}, Output: {output_tokens}")

            # Parse into AnalysisResult
            return self._parse_response(response_text, input_tokens, output_tokens)

        except Exception as e:
            # Re-raise if already AIProviderError
            if isinstance(e, AIProviderError):
                raise

            # Wrap other exceptions
            msg = f"Gemini API error: {e}"
            raise AIProviderError(msg) from e

    def get_model_name(self) -> str:
        """Return the model being used"""
        return self.model_name

    def estimate_cost(self, prompt_tokens: int) -> float:
        """
        Estimate cost for Gemini 2.0 Flash

        Pricing as of 2024:
        - Free tier: 15 RPM, 1M TPM, 1500 RPD
        - Paid: $0.10/1M input tokens (128k context)
        - Paid: $0.30/1M input tokens (>128k context)

        Args:
            prompt_tokens: Estimated input token count

        Returns:
            Estimated cost in USD
        """
        # Gemini 2.0 Flash pricing
        if prompt_tokens < 128_000:  # noqa: SIM108
            input_cost_per_mtok = 0.10
        else:
            input_cost_per_mtok = 0.30  # Long context pricing

        # Assume ~2000 output tokens for analysis
        estimated_output_tokens = 2000
        output_cost_per_mtok = 0.30  # Output cost

        input_cost = (prompt_tokens / 1_000_000) * input_cost_per_mtok
        output_cost = (estimated_output_tokens / 1_000_000) * output_cost_per_mtok

        return input_cost + output_cost
