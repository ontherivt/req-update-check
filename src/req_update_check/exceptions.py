from __future__ import annotations


class AIAnalysisError(Exception):
    """Base class for AI analysis errors"""


class APIKeyNotFoundError(AIAnalysisError):
    """API key not configured"""

    def __init__(self, provider: str, env_vars: list[str]):
        self.provider = provider
        self.env_vars = env_vars
        msg = self._build_message()
        super().__init__(msg)

    def _build_message(self) -> str:
        """Build helpful error message"""
        msg = f"{self.provider.title()} API key not found. Please provide via:\n"
        msg += "  1. --api-key argument\n"
        msg += f"  2. Environment variable: {' or '.join(self.env_vars)}\n"
        msg += "  3. Config file: ~/.config/req-update-check/config.toml\n"

        # Provider-specific help
        urls = {
            "claude": "https://console.anthropic.com/",
            "gemini": "https://aistudio.google.com/apikey",
            "openai": "https://platform.openai.com/api-keys",
        }
        if self.provider in urls:
            msg += f"\nGet your API key at: {urls[self.provider]}"

        return msg


class ChangelogFetchError(AIAnalysisError):
    """Could not fetch changelog - non-fatal"""


class APIRateLimitError(AIAnalysisError):
    """Hit rate limit - suggest retry"""


class AIProviderError(AIAnalysisError):
    """Generic provider error"""
