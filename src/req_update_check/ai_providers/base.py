from __future__ import annotations

import json
import logging
import time
from abc import ABC
from abc import abstractmethod
from dataclasses import asdict
from dataclasses import dataclass

from req_update_check.exceptions import AIProviderError

logger = logging.getLogger("req_update_check")


@dataclass
class AnalysisResult:
    """Result of AI analysis on a package update"""

    safety: str  # "safe" | "caution" | "breaking"
    confidence: str  # "high" | "medium" | "low"
    breaking_changes: list[str]
    deprecations: list[str]
    recommendations: list[str]
    new_features: list[str]
    summary: str
    provider: str  # Which AI generated this
    model: str  # Which model version
    input_tokens: int = 0  # Number of input tokens used
    output_tokens: int = 0  # Number of output tokens used
    total_tokens: int = 0  # Total tokens used

    def to_dict(self) -> dict:
        """Convert to dictionary for caching"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> AnalysisResult:
        """Create from dictionary"""
        # Handle older cached results that don't have token fields
        if "input_tokens" not in data:
            data["input_tokens"] = 0
        if "output_tokens" not in data:
            data["output_tokens"] = 0
        if "total_tokens" not in data:
            data["total_tokens"] = 0
        return cls(**data)


class AIProvider(ABC):
    """Abstract base class for AI providers"""

    @abstractmethod
    def analyze(self, prompt: str) -> AnalysisResult:
        """
        Send prompt to AI and return structured result

        Args:
            prompt: The analysis prompt to send

        Returns:
            AnalysisResult with safety assessment and recommendations

        Raises:
            AIProviderError: If API call fails
        """

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model being used"""

    @abstractmethod
    def estimate_cost(self, prompt_tokens: int) -> float:
        """
        Estimate cost in USD for given token count

        Args:
            prompt_tokens: Estimated number of input tokens

        Returns:
            Estimated cost in USD
        """

    def _retry_with_backoff(self, func, max_retries: int = 3):
        """
        Common retry logic with exponential backoff

        Args:
            func: Function to retry
            max_retries: Maximum number of retry attempts

        Returns:
            Result of func()

        Raises:
            AIProviderError: If all retries fail
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:  # noqa: PERF203, BLE001
                last_error = e
                error_str = str(e)

                # Check for quota/rate limit errors (common across providers)
                is_quota_error = any(
                    keyword in error_str.lower() for keyword in ["quota", "rate limit", "429", "resource_exhausted"]
                )

                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.debug(f"Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                    time.sleep(wait_time)
                elif is_quota_error:
                    # Use warning instead of exception to avoid full stack trace for quota errors
                    logger.warning(f"API quota/rate limit exceeded after {max_retries} retries")
                else:
                    logger.warning(f"All {max_retries} retries failed: {type(e).__name__}: {error_str[:200]}")

        # Simplify error message for quota errors
        error_msg = str(last_error)
        if "quota" in error_msg.lower() or "429" in error_msg:
            error_msg = error_msg.split("\n")[0]  # Just first line for quota errors

        raise AIProviderError(f"Failed after {max_retries} retries: {error_msg}") from last_error

    def _get_system_prompt(self) -> str:
        """Get the standard system prompt for analysis"""
        return """You are an expert software engineer analyzing Python package upgrades.
Given a package upgrade, its changelog, and codebase usage, assess:
1. Breaking changes that affect this codebase
2. Deprecation warnings relevant to current usage
3. New features that might be beneficial
4. Overall safety recommendation

Respond in JSON format with this exact structure:
{
  "safety": "safe" | "caution" | "breaking",
  "confidence": "high" | "medium" | "low",
  "breaking_changes": ["list of breaking changes that affect this codebase"],
  "deprecations": ["list of deprecations found in current usage"],
  "recommendations": ["actionable items before upgrading"],
  "new_features": ["relevant new features worth adopting"],
  "summary": "2-3 sentence assessment"
}

Guidelines:
- Focus on changes that impact the actual code shown
- Be specific about file locations when citing issues
- "breaking" means code will break without changes
- "caution" means review needed but likely safe
- "safe" means upgrade with minimal risk
- Include version numbers when referencing changes"""

    def _parse_response(
        self,
        response_text: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> AnalysisResult:
        """
        Parse JSON response from AI into AnalysisResult

        Args:
            response_text: JSON string from AI (may contain extra text)
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used

        Returns:
            AnalysisResult object

        Raises:
            AIProviderError: If parsing fails
        """
        try:
            # Try to parse as-is first
            data = json.loads(response_text)

            # Validate required fields
            required_fields = [
                "safety",
                "confidence",
                "breaking_changes",
                "deprecations",
                "recommendations",
                "new_features",
                "summary",
            ]
            for field in required_fields:
                if field not in data:
                    msg = f"Missing required field in AI response: {field}"
                    raise AIProviderError(msg)

            return AnalysisResult(
                safety=data["safety"],
                confidence=data["confidence"],
                breaking_changes=data["breaking_changes"],
                deprecations=data["deprecations"],
                recommendations=data["recommendations"],
                new_features=data["new_features"],
                summary=data["summary"],
                provider=self.__class__.__name__.replace("Provider", "").lower(),
                model=self.get_model_name(),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )
        except json.JSONDecodeError as e:
            # If direct parsing fails, try to extract JSON from the response
            # Look for JSON object between curly braces
            logger.debug(f"Direct JSON parsing failed, attempting to extract JSON: {e}")

            try:
                # Find the first { and try to parse from there
                start_idx = response_text.find("{")
                if start_idx == -1:
                    msg = f"Failed to parse AI response as JSON - no JSON object found: {e}"
                    raise AIProviderError(msg) from e

                # Try to find matching closing brace
                # Use a simple decoder to find where valid JSON ends
                decoder = json.JSONDecoder()
                data, end_idx = decoder.raw_decode(response_text, start_idx)

                logger.debug(f"Successfully extracted JSON from position {start_idx} to {end_idx}")

                # Validate required fields
                required_fields = [
                    "safety",
                    "confidence",
                    "breaking_changes",
                    "deprecations",
                    "recommendations",
                    "new_features",
                    "summary",
                ]
                for field in required_fields:
                    if field not in data:
                        msg = f"Missing required field in AI response: {field}"
                        raise AIProviderError(msg)

                return AnalysisResult(
                    safety=data["safety"],
                    confidence=data["confidence"],
                    breaking_changes=data["breaking_changes"],
                    deprecations=data["deprecations"],
                    recommendations=data["recommendations"],
                    new_features=data["new_features"],
                    summary=data["summary"],
                    provider=self.__class__.__name__.replace("Provider", "").lower(),
                    model=self.get_model_name(),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                )

            except (json.JSONDecodeError, ValueError) as extract_error:
                msg = f"Failed to parse AI response as JSON: {e}. Extraction also failed: {extract_error}"
                raise AIProviderError(msg) from e
        except KeyError as e:
            msg = f"Invalid AI response structure: {e}"
            raise AIProviderError(msg) from e
