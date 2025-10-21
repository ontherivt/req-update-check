from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .code_scanner import UsageReport


class PromptBuilder:
    """Builds prompts for AI analysis of package upgrades"""

    # System prompt is defined in base.py for consistency across providers
    # This class focuses on building the user prompt

    @staticmethod
    def build_analysis_prompt(  # noqa: PLR0913
        package_name: str,
        current_version: str,
        latest_version: str,
        update_level: str,
        changelog: str,
        usage_report: UsageReport,
    ) -> str:
        """
        Build the user prompt for AI analysis

        Args:
            package_name: Name of the package
            current_version: Current installed version
            latest_version: Latest available version
            update_level: Type of update (major/minor/patch)
            changelog: Changelog content
            usage_report: How the package is used in the codebase

        Returns:
            Formatted prompt string
        """
        # Truncate changelog if too long
        max_changelog_chars = 15000
        changelog_truncated = False

        if len(changelog) > max_changelog_chars:
            changelog = changelog[:max_changelog_chars]
            changelog_truncated = True

        # Build the prompt
        prompt = f"""Package: {package_name}
Upgrade: {current_version} → {latest_version}
Update Type: {update_level}

=== CHANGELOG ===
{changelog}"""

        if changelog_truncated:
            prompt += "\n... (truncated for length)"

        prompt += f"""

=== CURRENT USAGE IN CODEBASE ===
{usage_report.to_prompt_text()}

Analyze this upgrade for safety and provide specific, actionable recommendations."""

        return prompt

    @staticmethod
    def get_system_prompt() -> str:
        """
        Get the system prompt for AI analysis

        Note: This is also defined in base.py AIProvider._get_system_prompt()
        for consistency. This method is provided for reference.

        Returns:
            System prompt string
        """
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
