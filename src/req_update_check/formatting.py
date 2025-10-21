from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ai_providers.base import AnalysisResult


def format_ai_analysis(result: AnalysisResult) -> str:
    """
    Format AI analysis result for terminal output

    Args:
        result: AnalysisResult to format

    Returns:
        Formatted string suitable for console display
    """
    # Safety indicator with emoji
    safety_icons = {
        "safe": "✅",
        "caution": "⚠️ ",
        "breaking": "🚨",
    }

    output = []
    output.append("\n\tAI ANALYSIS:")
    output.append("\t" + "─" * 60)

    # Safety and confidence
    icon = safety_icons.get(result.safety, "❓")
    output.append(f"\t{icon} Safety: {result.safety.upper()} (Confidence: {result.confidence})")
    output.append(f"\tModel: {result.model}")

    # Token usage (if available)
    if result.total_tokens > 0:
        output.append(
            f"\tTokens: {result.input_tokens:,} in / {result.output_tokens:,} out / {result.total_tokens:,} total",
        )

    output.append("")

    # Breaking changes
    if result.breaking_changes:
        output.append("\tBreaking Changes:")
        output.extend([f"\t  • {change}" for change in result.breaking_changes])
        output.append("")

    # Deprecations
    if result.deprecations:
        output.append("\tDeprecations in Your Code:")
        output.extend([f"\t  • {dep}" for dep in result.deprecations])
        output.append("")

    # Recommendations
    if result.recommendations:
        output.append("\tRecommendations:")
        output.extend([f"\t  {i}. {rec}" for i, rec in enumerate(result.recommendations, 1)])
        output.append("")

    # New features (limit to 3 to avoid overwhelming)
    if result.new_features:
        output.append("\tNew Features:")
        output.extend([f"\t  • {feature}" for feature in result.new_features[:3]])
        if len(result.new_features) > 3:
            output.append(f"\t  ... and {len(result.new_features) - 3} more")
        output.append("")
    # Summary
    output.append(f"\tSummary: {result.summary}")
    output.append("\t" + "─" * 60)

    return "\n".join(output)
