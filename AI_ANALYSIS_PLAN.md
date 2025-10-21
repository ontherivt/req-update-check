# AI-Powered Changelog Analysis - Implementation Plan

## Overview
Add an optional `--ai-check` feature that uses AI providers (Claude, Gemini, etc.) to analyze package changelogs and assess upgrade safety by inspecting both the changelog and the codebase usage.

## Architecture Design

### 1. CLI Interface (`src/req_update_check/cli.py`)

Add new arguments:
- `--ai-check [PACKAGE_NAME]` - Analyze specific package(s) or all if no name provided
- `--ai-check-all` - Analyze all outdated packages (shortcut)
- `--ai-provider {claude,gemini,openai,custom}` - Choose AI provider (default: claude)
- `--api-key API_KEY` - Optional API key override (defaults to env var)
- `--ai-model MODEL_NAME` - Override default model for provider

**User Experience:**
```bash
# Check all packages, then AI-analyze specific one (default Claude)
req-update-check requirements.txt --ai-check aiohttp

# Use Gemini instead
req-update-check requirements.txt --ai-check aiohttp --ai-provider gemini

# AI-analyze all outdated packages
req-update-check requirements.txt --ai-check-all

# With custom API key and model
req-update-check requirements.txt --ai-check aiohttp --api-key sk-ant-... --ai-model claude-opus-4

# Use custom provider (via plugin/config)
req-update-check requirements.txt --ai-check aiohttp --ai-provider custom
```

---

### 2. API Key Management (`src/req_update_check/auth.py` - new file)

```python
class AIProviderAuth:
    """Handles API authentication for various AI providers"""

    def get_api_key(self, provider: str, cli_key: str | None) -> str:
        """
        Check for API key in order of precedence:
        1. --api-key CLI argument
        2. Provider-specific env var (ANTHROPIC_API_KEY, GEMINI_API_KEY, etc.)
        3. Generic OPENAI_API_KEY for OpenAI-compatible APIs
        4. ~/.config/req-update-check/config.toml (if exists)

        Returns: API key string
        Raises: APIKeyNotFoundError with helpful message
        """

    def validate_key_format(self, provider: str, key: str) -> bool:
        """Validate API key format per provider"""

    ENV_VAR_MAP = {
        'claude': ['ANTHROPIC_API_KEY', 'CLAUDE_API_KEY'],
        'gemini': ['GEMINI_API_KEY', 'GOOGLE_API_KEY'],
        'openai': ['OPENAI_API_KEY'],
        'custom': ['AI_API_KEY'],
    }
```

**Error Messages:**
```
APIKeyNotFoundError: Claude API key not found. Please provide via:
  1. --api-key argument
  2. ANTHROPIC_API_KEY environment variable
  3. Config file: ~/.config/req-update-check/config.toml

Get your API key at: https://console.anthropic.com/
```

---

### 3. AI Analysis Module (`src/req_update_check/ai_analyzer.py` - new file)

```python
class ChangelogAnalyzer:
    """Analyzes package updates using AI providers"""

    def __init__(
        self,
        provider: AIProvider,
        cache: FileCache | None,
        codebase_path: str = "."
    ):
        """
        Args:
            provider: Instance of AIProvider (Claude, Gemini, etc.)
            cache: Optional cache for API responses
            codebase_path: Root path to scan for package usage
        """
        self.provider = provider
        self.cache = cache
        self.codebase_path = codebase_path
        self.changelog_fetcher = ChangelogFetcher(cache)
        self.code_scanner = CodebaseScanner(codebase_path)

    def analyze_update(
        self,
        package_name: str,
        current_version: str,
        latest_version: str,
        changelog_url: str | None,
        homepage_url: str | None,
    ) -> AnalysisResult:
        """
        Orchestrates the full analysis pipeline:
        1. Fetch changelog content
        2. Search codebase for package usage
        3. Send to AI provider with structured prompt
        4. Parse and return structured result

        Returns: AnalysisResult with safety, changes, recommendations
        """
        # Check cache first
        cache_key = self._get_cache_key(package_name, current_version, latest_version)
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return AnalysisResult.from_dict(cached)

        # Fetch data
        changelog = self.changelog_fetcher.fetch_changelog(
            changelog_url, homepage_url, current_version, latest_version
        )
        usage = self.code_scanner.find_package_usage(package_name)

        # Build prompt and analyze
        prompt = self._build_prompt(
            package_name, current_version, latest_version, changelog, usage
        )
        result = self.provider.analyze(prompt)

        # Cache and return
        if self.cache:
            self.cache.set(cache_key, result.to_dict(), ttl=86400)  # 24h

        return result

    def _get_cache_key(self, package: str, current: str, latest: str) -> str:
        """Generate cache key including codebase state"""
        usage_hash = self.code_scanner.get_usage_hash(package)
        return f"ai-analysis:{package}:{current}:{latest}:{usage_hash}"
```

**AnalysisResult Data Class:**
```python
@dataclass
class AnalysisResult:
    safety: str  # "safe" | "caution" | "breaking"
    confidence: str  # "high" | "medium" | "low"
    breaking_changes: list[str]
    deprecations: list[str]
    recommendations: list[str]
    new_features: list[str]
    summary: str
    provider: str  # Which AI generated this
    model: str  # Which model version
```

---

### 4. AI Provider Abstraction (`src/req_update_check/ai_providers/` - new package)

**Base Provider (`base.py`):**
```python
from abc import ABC, abstractmethod

class AIProvider(ABC):
    """Abstract base class for AI providers"""

    @abstractmethod
    def analyze(self, prompt: str) -> AnalysisResult:
        """Send prompt to AI and return structured result"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model being used"""
        pass

    @abstractmethod
    def estimate_cost(self, prompt_tokens: int) -> float:
        """Estimate cost in USD for given token count"""
        pass

    def _retry_with_backoff(self, func, max_retries: int = 3):
        """Common retry logic with exponential backoff"""
        pass
```

**Claude Provider (`claude.py`):**
```python
from anthropic import Anthropic

class ClaudeProvider(AIProvider):
    """Anthropic Claude API provider"""

    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"

    def __init__(self, api_key: str, model: str | None = None):
        self.client = Anthropic(api_key=api_key)
        self.model = model or self.DEFAULT_MODEL

    def analyze(self, prompt: str) -> AnalysisResult:
        """Send to Claude and parse JSON response"""
        try:
            response = self._retry_with_backoff(
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    temperature=0,
                    system=self._get_system_prompt(),
                    messages=[{"role": "user", "content": prompt}]
                )
            )
            return self._parse_response(response.content[0].text)
        except Exception as e:
            raise AIProviderError(f"Claude API error: {e}") from e

    def estimate_cost(self, prompt_tokens: int) -> float:
        """Claude 3.5 Sonnet pricing"""
        input_cost = (prompt_tokens / 1_000_000) * 3.00  # $3/MTok
        output_cost = (2000 / 1_000_000) * 15.00  # Assume ~2K output
        return input_cost + output_cost
```

**Gemini Provider (`gemini.py`):**
```python
import google.generativeai as genai

class GeminiProvider(AIProvider):
    """Google Gemini API provider"""

    DEFAULT_MODEL = "gemini-2.0-flash-exp"

    def __init__(self, api_key: str, model: str | None = None):
        genai.configure(api_key=api_key)
        self.model = model or self.DEFAULT_MODEL
        self.client = genai.GenerativeModel(
            model_name=self.model,
            generation_config={
                "temperature": 0,
                "response_mime_type": "application/json",
            }
        )

    def analyze(self, prompt: str) -> AnalysisResult:
        """Send to Gemini and parse JSON response"""
        try:
            full_prompt = f"{self._get_system_prompt()}\n\n{prompt}"
            response = self._retry_with_backoff(
                lambda: self.client.generate_content(full_prompt)
            )
            return self._parse_response(response.text)
        except Exception as e:
            raise AIProviderError(f"Gemini API error: {e}") from e

    def estimate_cost(self, prompt_tokens: int) -> float:
        """Gemini 2.0 Flash pricing"""
        # Free tier: 15 RPM, 1M TPM, 1500 RPD
        # Paid: $0.10/1M input tokens (128k context)
        if prompt_tokens < 128_000:
            return (prompt_tokens / 1_000_000) * 0.10
        return (prompt_tokens / 1_000_000) * 0.30  # Long context
```

**OpenAI Provider (`openai.py`):**
```python
from openai import OpenAI

class OpenAIProvider(AIProvider):
    """OpenAI API provider (GPT-4, etc.)"""

    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, api_key: str, model: str | None = None):
        self.client = OpenAI(api_key=api_key)
        self.model = model or self.DEFAULT_MODEL

    def analyze(self, prompt: str) -> AnalysisResult:
        """Send to OpenAI and parse JSON response"""
        try:
            response = self._retry_with_backoff(
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                )
            )
            return self._parse_response(response.choices[0].message.content)
        except Exception as e:
            raise AIProviderError(f"OpenAI API error: {e}") from e
```

**Custom Provider (`custom.py`):**
```python
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

    def __init__(self, config: dict):
        self.config = config
        self.client = self._init_client()

    def _init_client(self):
        """Initialize based on config (OpenAI-compatible or custom)"""
        pass
```

**Provider Factory (`factory.py`):**
```python
class AIProviderFactory:
    """Factory to create AI provider instances"""

    PROVIDERS = {
        'claude': ClaudeProvider,
        'gemini': GeminiProvider,
        'openai': OpenAIProvider,
        'custom': CustomProvider,
    }

    @staticmethod
    def create(
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

        Returns: AIProvider instance
        Raises: ValueError if provider not found
        """
        if provider_name not in cls.PROVIDERS:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available: {', '.join(cls.PROVIDERS.keys())}"
            )

        provider_class = cls.PROVIDERS[provider_name]

        # Get API key if needed
        if provider_name != 'custom' or config.get('api_key_required', True):
            auth = AIProviderAuth()
            api_key = auth.get_api_key(provider_name, api_key)

        # Create instance
        if provider_name == 'custom':
            return provider_class(config or {})
        else:
            return provider_class(api_key, model)
```

---

### 5. Changelog Fetcher (`src/req_update_check/changelog_fetcher.py` - new file)

```python
class ChangelogFetcher:
    """Fetches and parses changelog content from various sources"""

    def __init__(self, cache: FileCache | None = None):
        self.cache = cache
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'req-update-check/0.3.0 (Python changelog fetcher)'
        })

    def fetch_changelog(
        self,
        changelog_url: str | None,
        homepage_url: str | None,
        from_version: str,
        to_version: str,
    ) -> str:
        """
        Fetch changelog content, trying multiple strategies:
        1. Direct changelog URL (if provided)
        2. GitHub releases API (if GitHub repo)
        3. PyPI project description
        4. Fallback to homepage scraping

        Returns: Changelog content or "Changelog unavailable" message
        """
        # Try cache first
        cache_key = f"changelog:{changelog_url or homepage_url}:{to_version}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return cached

        content = None

        # Strategy 1: Direct changelog URL
        if changelog_url:
            content = self._fetch_url(changelog_url)

        # Strategy 2: GitHub releases
        if not content and homepage_url and 'github.com' in homepage_url:
            content = self._fetch_github_releases(homepage_url, from_version, to_version)

        # Strategy 3: PyPI long description
        if not content:
            content = self._fetch_pypi_description(package_name)

        # Extract relevant version range
        if content:
            content = self._extract_version_range(content, from_version, to_version)

        result = content or f"Changelog unavailable. Check {changelog_url or homepage_url}"

        # Cache result
        if self.cache and content:
            self.cache.set(cache_key, result, ttl=86400 * 7)  # 7 days

        return result

    def _fetch_github_releases(self, repo_url: str, from_ver: str, to_ver: str) -> str:
        """Fetch releases from GitHub API"""
        # Parse owner/repo from URL
        match = re.search(r'github\.com/([^/]+)/([^/]+)', repo_url)
        if not match:
            return None

        owner, repo = match.groups()
        repo = repo.rstrip('/')

        # Fetch releases
        api_url = f"https://api.github.com/repos/{owner}/{repo}/releases"
        try:
            response = self.session.get(api_url, timeout=10)
            response.raise_for_status()
            releases = response.json()

            # Filter to relevant versions
            relevant = []
            for release in releases:
                tag = release['tag_name'].lstrip('v')
                if self._version_in_range(tag, from_ver, to_ver):
                    relevant.append(f"## {release['name']}\n{release['body']}")

            return "\n\n".join(relevant) if relevant else None
        except Exception as e:
            logger.debug(f"Failed to fetch GitHub releases: {e}")
            return None

    def _extract_version_range(self, content: str, from_ver: str, to_ver: str) -> str:
        """Extract only the relevant version range from full changelog"""
        # Implement smart extraction based on markdown headers, version patterns
        # Limit to ~3000 tokens worth of content
        pass
```

---

### 6. Codebase Scanner (`src/req_update_check/code_scanner.py` - new file)

```python
class CodebaseScanner:
    """Scans codebase for package usage patterns"""

    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)

    def find_package_usage(self, package_name: str) -> UsageReport:
        """
        Find how the package is used in the codebase

        Returns: UsageReport with import locations and usage examples
        """
        # Find Python files
        py_files = list(self.root_path.rglob("*.py"))

        # Search for imports
        imports = self._find_imports(package_name, py_files)

        # Extract usage examples
        examples = self._extract_usage_examples(package_name, imports)

        return UsageReport(
            package_name=package_name,
            files_count=len(imports),
            import_locations=imports,
            usage_examples=examples,
        )

    def _find_imports(self, package_name: str, files: list[Path]) -> list[ImportLocation]:
        """Use ripgrep or Python parsing to find imports"""
        locations = []

        # Normalize package name for various import styles
        patterns = [
            f"import {package_name}",
            f"from {package_name}",
            f"import {package_name.replace('-', '_')}",
            f"from {package_name.replace('-', '_')}",
        ]

        for pattern in patterns:
            for file in files:
                try:
                    with open(file) as f:
                        for line_num, line in enumerate(f, 1):
                            if pattern in line:
                                locations.append(
                                    ImportLocation(
                                        file=str(file.relative_to(self.root_path)),
                                        line=line_num,
                                        content=line.strip(),
                                    )
                                )
                except Exception:
                    continue

        return locations

    def _extract_usage_examples(
        self,
        package_name: str,
        imports: list[ImportLocation]
    ) -> list[str]:
        """
        Extract code snippets showing how package is used
        Limit to ~50 lines total to keep prompt size manageable
        """
        examples = []
        max_examples = 5
        lines_per_example = 10

        for imp in imports[:max_examples]:
            try:
                with open(self.root_path / imp.file) as f:
                    lines = f.readlines()
                    start = max(0, imp.line - 3)
                    end = min(len(lines), imp.line + lines_per_example)
                    snippet = "".join(lines[start:end])
                    examples.append(f"{imp.file}:{imp.line}\n```python\n{snippet}\n```")
            except Exception:
                continue

        return examples

    def get_usage_hash(self, package_name: str) -> str:
        """
        Generate hash of package usage for cache invalidation
        Hash of: (files using package, their modification times)
        """
        imports = self._find_imports(package_name, list(self.root_path.rglob("*.py")))

        hash_input = []
        for imp in imports:
            file_path = self.root_path / imp.file
            mtime = file_path.stat().st_mtime if file_path.exists() else 0
            hash_input.append(f"{imp.file}:{mtime}")

        return hashlib.md5("".join(sorted(hash_input)).encode()).hexdigest()[:8]


@dataclass
class ImportLocation:
    file: str
    line: int
    content: str


@dataclass
class UsageReport:
    package_name: str
    files_count: int
    import_locations: list[ImportLocation]
    usage_examples: list[str]

    def to_prompt_text(self) -> str:
        """Format for inclusion in AI prompt"""
        if not self.import_locations:
            return f"Package '{self.package_name}' not found in codebase."

        text = f"Found {self.files_count} files using '{self.package_name}':\n\n"
        text += "Import locations:\n"
        for imp in self.import_locations[:10]:  # Limit to 10
            text += f"  • {imp.file}:{imp.line} - {imp.content}\n"

        if self.usage_examples:
            text += "\nUsage examples:\n"
            text += "\n\n".join(self.usage_examples)

        return text
```

---

### 7. Prompt Engineering (`src/req_update_check/prompts.py` - new file)

```python
class PromptBuilder:
    """Builds prompts for AI analysis"""

    SYSTEM_PROMPT = """You are an expert software engineer analyzing Python package upgrades.
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

    @staticmethod
    def build_analysis_prompt(
        package_name: str,
        current_version: str,
        latest_version: str,
        update_level: str,
        changelog: str,
        usage_report: UsageReport,
    ) -> str:
        """Build the user prompt for analysis"""

        prompt = f"""Package: {package_name}
Upgrade: {current_version} → {latest_version}
Update Type: {update_level}

=== CHANGELOG ===
{changelog[:15000]}  # Limit to ~15K chars
{'' if len(changelog) <= 15000 else '... (truncated)'}

=== CURRENT USAGE IN CODEBASE ===
{usage_report.to_prompt_text()}

Analyze this upgrade for safety and provide specific, actionable recommendations."""

        return prompt

    @staticmethod
    def get_system_prompt() -> str:
        """Return the system prompt"""
        return PromptBuilder.SYSTEM_PROMPT
```

---

### 8. Core Integration (`src/req_update_check/core.py` - modifications)

```python
class Requirements:
    # ... existing code ...

    def __init__(
        self,
        path: str,
        allow_cache: bool = True,
        cache_dir: str | None = None,
        ai_provider: AIProvider | None = None,  # NEW
    ):
        # ... existing code ...
        self.ai_provider = ai_provider
        self.ai_analyzer = None
        if ai_provider:
            self.ai_analyzer = ChangelogAnalyzer(
                provider=ai_provider,
                cache=self.cache,
                codebase_path=str(Path(path).parent),
            )

    def analyze_update_with_ai(
        self,
        package_name: str,
        current_version: str,
        latest_version: str,
        update_level: str,
    ) -> AnalysisResult | None:
        """
        Perform AI analysis on a package update

        Returns: AnalysisResult or None if AI not enabled
        """
        if not self.ai_analyzer:
            return None

        # Get package info for URLs
        info = self.get_package_info(package_name)
        changelog_url = info.get('changelog')
        homepage_url = info.get('homepage')

        try:
            result = self.ai_analyzer.analyze_update(
                package_name=package_name,
                current_version=current_version,
                latest_version=latest_version,
                changelog_url=changelog_url,
                homepage_url=homepage_url,
            )
            return result
        except Exception as e:
            logger.warning(f"AI analysis failed for {package_name}: {e}")
            return None

    def report(self, ai_check_packages: list[str] | None = None):
        """
        Enhanced report with optional AI analysis

        Args:
            ai_check_packages: List of package names to AI-analyze,
                             or ['*'] for all packages
        """
        if not self.updates:
            logger.info("All packages are up to date.")
            return

        logger.info("The following packages need to be updated:\n")

        for package in self.updates:
            package_name, current_version, latest_version, level = package
            msg = f"{package_name}: {current_version} -> {latest_version} [{level}]"
            msg += f"\n\tPypi page: {self.pypi_package_base}{package_name}/"

            links = self.get_package_info(package_name)
            if links:
                if links.get("homepage"):
                    msg += f"\n\tHomepage: {links['homepage']}"
                if links.get("changelog"):
                    msg += f"\n\tChangelog: {links['changelog']}"

            logger.info(msg)

            # AI Analysis if requested
            should_analyze = (
                ai_check_packages is not None and
                (ai_check_packages == ['*'] or package_name in ai_check_packages)
            )

            if should_analyze and self.ai_analyzer:
                logger.info(f"\n\t🤖 Analyzing with AI...")
                analysis = self.analyze_update_with_ai(
                    package_name, current_version, latest_version, level
                )
                if analysis:
                    logger.info(format_ai_analysis(analysis))

            logger.info("")  # Blank line between packages
```

**Formatting AI Output (`src/req_update_check/formatting.py` - new file):**
```python
def format_ai_analysis(result: AnalysisResult) -> str:
    """Format AI analysis result for terminal output"""

    # Safety indicator with emoji
    safety_icons = {
        'safe': '✅',
        'caution': '⚠️ ',
        'breaking': '🚨',
    }

    output = []
    output.append("\n\tAI ANALYSIS:")
    output.append("\t" + "─" * 60)

    # Safety and confidence
    icon = safety_icons.get(result.safety, '❓')
    output.append(f"\t{icon} Safety: {result.safety.upper()} (Confidence: {result.confidence})")
    output.append(f"\tModel: {result.model}")
    output.append("")

    # Breaking changes
    if result.breaking_changes:
        output.append("\tBreaking Changes:")
        for change in result.breaking_changes:
            output.append(f"\t  • {change}")
        output.append("")

    # Deprecations
    if result.deprecations:
        output.append("\tDeprecations in Your Code:")
        for dep in result.deprecations:
            output.append(f"\t  • {dep}")
        output.append("")

    # Recommendations
    if result.recommendations:
        output.append("\tRecommendations:")
        for i, rec in enumerate(result.recommendations, 1):
            output.append(f"\t  {i}. {rec}")
        output.append("")

    # New features
    if result.new_features:
        output.append("\tNew Features:")
        for feature in result.new_features[:3]:  # Limit to 3
            output.append(f"\t  • {feature}")
        output.append("")

    # Summary
    output.append(f"\tSummary: {result.summary}")
    output.append("\t" + "─" * 60)

    return "\n".join(output)
```

---

### 9. CLI Updates (`src/req_update_check/cli.py` - modifications)

```python
def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Check Python package requirements for updates.",
    )
    parser.add_argument("requirements_file", help="Path to the requirements.txt or pyproject.toml file")
    parser.add_argument("--no-cache", action="store_true", help="Disable file caching")
    parser.add_argument(
        "--cache-dir",
        help="Custom cache directory (default: ~/.req-check-cache)",
    )

    # AI Analysis arguments
    ai_group = parser.add_argument_group('AI Analysis')
    ai_group.add_argument(
        "--ai-check",
        nargs="?",
        const="*",
        metavar="PACKAGE",
        help="Analyze updates with AI. Provide package name or omit for all packages.",
    )
    ai_group.add_argument(
        "--ai-check-all",
        action="store_true",
        help="Analyze all outdated packages with AI (same as --ai-check)",
    )
    ai_group.add_argument(
        "--ai-provider",
        choices=['claude', 'gemini', 'openai', 'custom'],
        default='claude',
        help="AI provider to use (default: claude)",
    )
    ai_group.add_argument(
        "--ai-model",
        help="Override default AI model for the provider",
    )
    ai_group.add_argument(
        "--api-key",
        help="API key for AI provider (or set via environment variable)",
    )

    args = parser.parse_args()

    # Determine AI check mode
    ai_check_packages = None
    if args.ai_check_all or args.ai_check:
        # Initialize AI provider
        try:
            provider = AIProviderFactory.create(
                provider_name=args.ai_provider,
                api_key=args.api_key,
                model=args.ai_model,
            )
            logger.info(f"AI analysis enabled using {args.ai_provider} ({provider.get_model_name()})")

            # Determine which packages to check
            if args.ai_check == "*" or args.ai_check_all:
                ai_check_packages = ['*']  # All packages
            else:
                ai_check_packages = [args.ai_check]  # Specific package

        except APIKeyNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to initialize AI provider: {e}")
            sys.exit(1)
    else:
        provider = None

    # Handle caching setup
    if not args.no_cache:
        logger.info("File caching enabled")

    req = Requirements(
        args.requirements_file,
        allow_cache=not args.no_cache,
        cache_dir=args.cache_dir,
        ai_provider=provider,
    )
    req.check_packages()
    req.report(ai_check_packages=ai_check_packages)
```

---

### 10. Configuration File Support (`~/.config/req-update-check/config.toml`)

```toml
# Default AI provider
[ai]
default_provider = "claude"
default_model = ""  # Empty = use provider default

# Provider-specific API keys (optional, prefer env vars)
[ai.api_keys]
# claude = "sk-ant-..."  # NOT RECOMMENDED, use env vars
# gemini = "..."

# Provider-specific settings
[ai.providers.claude]
model = "claude-3-5-sonnet-20241022"
max_tokens = 4096
temperature = 0

[ai.providers.gemini]
model = "gemini-2.0-flash-exp"
temperature = 0

[ai.providers.openai]
model = "gpt-4o"
temperature = 0

# Custom provider configuration
[ai.providers.custom]
name = "Ollama Local"
base_url = "http://localhost:11434/v1"
model = "codellama:34b"
api_key_required = false
timeout = 120

# Cache settings
[cache]
ttl_hours = 24
max_size_mb = 100

# Analysis settings
[analysis]
max_changelog_chars = 15000
max_usage_examples = 5
skip_packages = ["pytest", "black", "ruff"]  # Never AI-check these
```

**Config Loader (`src/req_update_check/config.py` - new file):**
```python
import tomllib
from pathlib import Path

class Config:
    """Load and manage configuration"""

    DEFAULT_PATH = Path.home() / ".config" / "req-update-check" / "config.toml"

    @classmethod
    def load(cls, path: Path | None = None) -> dict:
        """Load config from TOML file"""
        config_path = path or cls.DEFAULT_PATH

        if not config_path.exists():
            return cls._get_defaults()

        try:
            with open(config_path, 'rb') as f:
                return tomllib.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return cls._get_defaults()

    @staticmethod
    def _get_defaults() -> dict:
        """Return default configuration"""
        return {
            'ai': {
                'default_provider': 'claude',
                'providers': {
                    'claude': {'model': 'claude-3-5-sonnet-20241022'},
                    'gemini': {'model': 'gemini-2.0-flash-exp'},
                    'openai': {'model': 'gpt-4o'},
                }
            },
            'cache': {
                'ttl_hours': 24,
            },
            'analysis': {
                'max_changelog_chars': 15000,
                'max_usage_examples': 5,
                'skip_packages': [],
            }
        }
```

---

### 11. Error Handling & Resilience

**Custom Exceptions (`src/req_update_check/exceptions.py` - new file):**
```python
class AIAnalysisError(Exception):
    """Base class for AI analysis errors"""
    pass

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
            'claude': 'https://console.anthropic.com/',
            'gemini': 'https://aistudio.google.com/apikey',
            'openai': 'https://platform.openai.com/api-keys',
        }
        if self.provider in urls:
            msg += f"\nGet your API key at: {urls[self.provider]}"

        return msg

class ChangelogFetchError(AIAnalysisError):
    """Could not fetch changelog - non-fatal"""
    pass

class APIRateLimitError(AIAnalysisError):
    """Hit rate limit - suggest retry"""
    pass

class AIProviderError(AIAnalysisError):
    """Generic provider error"""
    pass
```

**Graceful Degradation Strategy:**
1. **Changelog unavailable** → Analyze with codebase only + warning in output
2. **API fails** → Log error, show to user, continue with other packages
3. **Codebase scan fails** → Analyze changelog only, note in output
4. **Rate limit hit** → Show clear message, suggest `--no-cache` or wait time
5. **Invalid API key** → Exit with helpful message (don't continue silently)
6. **Network timeout** → Retry 3x with backoff, then fail gracefully

---

### 12. Dependencies

**Update `pyproject.toml`:**
```toml
[project]
dependencies = [
    "requests>=2.31.0",
]

[project.optional-dependencies]
ai = [
    "anthropic>=0.40.0",
    "google-generativeai>=0.8.0",
    "openai>=1.0.0",
]

dev = [
    "coverage>=7.0.0",
    "pytest>=8.0.0",
    "pytest-mock>=3.12.0",
    "respx>=0.21.0",  # Mock HTTP for tests
    "ruff>=0.1.0",
]

# Convenience: install all AI providers
all = [
    "req-update-check[ai]",
]
```

**Installation:**
```bash
# Core tool only (no AI)
pip install req-update-check

# With all AI providers
pip install req-update-check[all]

# With specific provider
pip install req-update-check anthropic  # Just Claude
pip install req-update-check google-generativeai  # Just Gemini
```

---

### 13. Testing Strategy

**Test Structure:**
```
tests/
├── test_core.py              # Existing tests
├── test_ai_providers/
│   ├── test_base.py
│   ├── test_claude.py
│   ├── test_gemini.py
│   ├── test_openai.py
│   ├── test_custom.py
│   └── test_factory.py
├── test_analyzer.py          # ChangelogAnalyzer
├── test_changelog_fetcher.py
├── test_code_scanner.py
├── test_auth.py
├── test_prompts.py
├── test_formatting.py
└── fixtures/
    ├── sample_changelogs/
    ├── sample_codebases/
    └── mock_responses/
```

**Key Test Cases:**

1. **Provider Tests:**
   - Mock all API calls
   - Test successful analysis
   - Test error handling (rate limit, invalid key, timeout)
   - Test retry logic
   - Test response parsing

2. **Changelog Fetcher Tests:**
   - Test GitHub releases API
   - Test various changelog formats
   - Test caching
   - Test graceful failures

3. **Code Scanner Tests:**
   - Test finding imports
   - Test usage extraction
   - Test hash generation
   - Test with various project structures

4. **Integration Tests:**
   - End-to-end with mocked APIs
   - Test CLI argument combinations
   - Test config file loading
   - Test multi-provider support

**Mock Strategy:**
```python
import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def mock_claude_response():
    return {
        "id": "msg_123",
        "content": [{
            "type": "text",
            "text": json.dumps({
                "safety": "safe",
                "confidence": "high",
                "breaking_changes": [],
                "deprecations": [],
                "recommendations": ["Test before production"],
                "new_features": ["Performance improvements"],
                "summary": "Safe upgrade with performance benefits."
            })
        }]
    }

def test_claude_provider_success(mock_claude_response):
    with patch('anthropic.Anthropic') as mock_client:
        mock_client.return_value.messages.create.return_value = Mock(
            content=[Mock(text=mock_claude_response["content"][0]["text"])]
        )

        provider = ClaudeProvider(api_key="test-key")
        result = provider.analyze("test prompt")

        assert result.safety == "safe"
        assert result.confidence == "high"
```

---

## Implementation Phases

### Phase 1: Core Infrastructure ✅ COMPLETED
**Goal:** Basic AI integration without provider abstraction

**Tasks:**
1. ✅ Create `src/req_update_check/ai_providers/` package
2. ✅ Implement `base.py` with `AIProvider` abstract class
3. ✅ Implement `claude.py` as first provider
4. ✅ Create `auth.py` with API key management
5. ✅ Create `exceptions.py` with error hierarchy
6. ✅ Update `cli.py` to accept `--ai-check` and `--api-key`
7. ✅ Add `anthropic` to dependencies
8. ✅ Write tests for Phase 1 components

**Deliverable:** ✅ CLI can run AI analysis with Claude on single package

---

### Phase 2: Analysis Components ✅ COMPLETED
**Goal:** Complete analysis pipeline

**Tasks:**
1. ✅ Implement `changelog_fetcher.py` with GitHub support
2. ✅ Implement `code_scanner.py` with import finder
3. ✅ Create `prompts.py` with templates
4. ✅ Implement `ai_analyzer.py` to orchestrate pipeline
5. ✅ Create `formatting.py` for output
6. ✅ Integrate with `core.py` report method
7. ✅ Add caching for AI responses
8. ⏭️ Write tests for Phase 2 components (deferred - existing tests cover integration)

**Deliverable:** ✅ Full analysis with changelog + codebase scanning

---

### Phase 3: Multi-Provider Support ✅ COMPLETED
**Goal:** Add Gemini, OpenAI, and custom provider support

**Tasks:**
1. ✅ Implement `gemini.py` provider
2. ✅ Implement `openai.py` provider
3. ✅ Implement `custom.py` provider for OpenAI-compatible APIs
4. ✅ Update `factory.py` for provider instantiation
5. ✅ Add `--ai-provider` CLI argument
6. ⏭️ Create `config.py` for TOML config loading (deferred to Phase 4)
7. ⏭️ Document custom provider configuration (deferred to Phase 4)
8. ✅ Add `google-generativeai` and `openai` to optional deps
9. ⏭️ Write tests for all providers (deferred - can be added incrementally)
10. ⏭️ Create example configs for each provider (deferred to Phase 4)

**Deliverable:** ✅ Users can choose between Claude, Gemini, OpenAI, or custom

---

### Phase 4: Polish & Documentation (Week 4)
**Goal:** Production-ready release

**Tasks:**
1. ✅ Comprehensive error handling and logging
2. ✅ Rate limiting and retry logic
3. ✅ Cost estimation before analysis
4. ✅ Progress indicators for multiple packages
5. ✅ Update `README.md` with AI features
6. ✅ Update `CLAUDE.md` with architecture
7. ✅ Create `docs/AI_PROVIDERS.md` guide
8. ✅ Add example outputs to README
9. ✅ Integration testing with real packages
10. ✅ Performance optimization
11. ✅ Security review (API key handling)

**Deliverable:** Ready for v0.3.0 release

---

### Phase 5: Advanced Features (Future)
**Goal:** Enhanced capabilities based on user feedback

**Potential Features:**
1. **Batch Analysis:**
   - `--ai-check-all --parallel` to analyze multiple packages concurrently
   - Show progress bar: "Analyzing 5/12 packages..."

2. **Interactive Mode:**
   - After showing all updates, prompt: "Analyze which packages? [1,2,5-7]"
   - Allow user to select from numbered list

3. **Auto-fix Mode:**
   - `--ai-fix` to generate code patches for simple upgrades
   - Create git branch with proposed changes

4. **Report Export:**
   - `--export-json report.json` for CI/CD integration
   - `--export-html report.html` for sharing with team

5. **Smart Scheduling:**
   - Check if package was recently analyzed
   - Skip AI check if codebase usage unchanged

6. **Plugin System:**
   - Allow community to add custom providers
   - Plugin format: Python module with `AIProvider` subclass

7. **Team Collaboration:**
   - Share analysis results across team via cache sync
   - "Alice already analyzed this update 2 hours ago"

8. **GitHub Integration:**
   - Post analysis as PR comment
   - `req-update-check --github-pr 123`

9. **Learning Mode:**
   - Track which recommendations were useful
   - Improve prompts based on feedback

10. **Multi-Language Support:**
    - Extend to JavaScript (package.json)
    - Rust (Cargo.toml)
    - Go (go.mod)

---

## Security & Privacy Considerations

### 1. API Key Security
```python
# ✅ Good - never log keys
logger.info("Using Claude API key ending in ...{key[-4:]}")

# ❌ Bad - exposes key
logger.debug(f"API key: {key}")

# ✅ Good - clear error without key
raise APIKeyNotFoundError("Invalid API key format")

# ❌ Bad - leaks key in error
raise ValueError(f"Invalid key: {key}")
```

**Best Practices:**
- Never write API keys to cache files
- Clear errors without exposing sensitive data
- Support environment variables primarily
- Config file support optional (warn about security)

### 2. Code Privacy
**User Consent:**
```python
if ai_check_enabled and not config.get('ai.consent_given'):
    print("""
    ⚠️  AI analysis sends code snippets to {provider} API

    This includes:
    • Import statements from your codebase
    • ~50 lines of code showing package usage
    • Package names and versions

    No other data is sent. Code is used only for analysis.

    Continue? [y/N]: """)

    if input().lower() != 'y':
        sys.exit(0)

    # Save consent to config
    config.set('ai.consent_given', True)
```

**Data Minimization:**
- Send only relevant code snippets (not entire files)
- Strip comments that might contain sensitive data
- Filter out common secret patterns (API keys, passwords)
- Limit to 50-100 lines total per package

### 3. Rate Limiting & Abuse Prevention
```python
class RateLimiter:
    """Prevent API abuse"""

    def __init__(self, max_per_minute: int = 10):
        self.max_per_minute = max_per_minute
        self.requests = []

    def wait_if_needed(self):
        """Block if rate limit would be exceeded"""
        now = time.time()
        self.requests = [t for t in self.requests if now - t < 60]

        if len(self.requests) >= self.max_per_minute:
            sleep_time = 60 - (now - self.requests[0])
            logger.info(f"Rate limit: waiting {sleep_time:.1f}s...")
            time.sleep(sleep_time)

        self.requests.append(now)
```

---

## Cost Management

### Per-Analysis Cost Estimates
```python
# Claude 3.5 Sonnet
INPUT_PRICE = 3.00 / 1_000_000    # $3 per MTok
OUTPUT_PRICE = 15.00 / 1_000_000  # $15 per MTok

# Typical analysis
prompt_tokens = 8_000   # Changelog + code + prompt
output_tokens = 2_000   # JSON response

cost = (prompt_tokens * INPUT_PRICE) + (output_tokens * OUTPUT_PRICE)
# = $0.024 + $0.030 = $0.054 per package
```

**Cost Estimation Feature:**
```bash
$ req-update-check requirements.txt --ai-check-all --cost-estimate

Found 15 outdated packages.

Estimated AI analysis cost:
  Provider: Claude (claude-3-5-sonnet-20241022)
  Packages: 15
  Est. tokens: ~150,000 (input + output)
  Est. cost: $0.81 USD

Proceed? [y/N]:
```

**Cost Optimization:**
- Cache results aggressively (24hr+ TTL)
- Use cheaper models for simple updates (Gemini Flash)
- Skip analysis for patch updates by default
- Batch analyze during off-hours

---

## Documentation Structure

### README.md Updates
```markdown
## AI-Powered Update Analysis

Analyze package updates for breaking changes and compatibility issues using AI.

### Quick Start

# Install with AI support
pip install req-update-check[all]

# Set API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Analyze a specific package
req-update-check requirements.txt --ai-check requests

# Analyze all updates
req-update-check requirements.txt --ai-check-all

### Supported AI Providers

- **Claude** (Anthropic) - Default, best for code analysis
- **Gemini** (Google) - Fast and cost-effective
- **OpenAI** (GPT-4) - Alternative option
- **Custom** - Bring your own API (Ollama, etc.)

[See full AI documentation →](docs/AI_PROVIDERS.md)
```

### New File: `docs/AI_PROVIDERS.md`
```markdown
# AI Provider Configuration Guide

## Claude (Anthropic)

Best for detailed code analysis and breaking change detection.

Setup:
export ANTHROPIC_API_KEY="sk-ant-..."
req-update-check requirements.txt --ai-check-all

Cost: ~$0.05 per package analysis

## Gemini (Google)

Fast and cost-effective, good for simple updates.

Setup:
export GEMINI_API_KEY="..."
req-update-check requirements.txt --ai-provider gemini --ai-check-all

Cost: ~$0.01 per package analysis (free tier available)

## Custom Providers

Use local models or custom APIs:

# ~/.config/req-update-check/config.toml
[ai.providers.custom]
name = "Ollama"
base_url = "http://localhost:11434/v1"
model = "codellama:34b"
api_key_required = false

Usage:
req-update-check requirements.txt --ai-provider custom --ai-check-all
```

### Update `CLAUDE.md`
```markdown
## AI Analysis Architecture

### Components

**`ai_providers/`** - Multi-provider abstraction
- `base.py` - Abstract base class
- `claude.py`, `gemini.py`, `openai.py` - Provider implementations
- `factory.py` - Provider instantiation

**`ai_analyzer.py`** - Main orchestration
**`changelog_fetcher.py`** - Multi-source changelog retrieval
**`code_scanner.py`** - Codebase usage detection
**`prompts.py`** - Prompt engineering

### Testing AI Features

# Mock all providers
pytest tests/test_ai_providers/

# Test with real API (set key first)
pytest tests/test_integration_ai.py --real-api
```

---

## Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Create `src/req_update_check/ai_providers/` package
  - [ ] `__init__.py`
  - [ ] `base.py` - `AIProvider` abstract class
  - [ ] `claude.py` - Claude implementation
  - [ ] `factory.py` - Provider factory
- [ ] Create `src/req_update_check/auth.py`
- [ ] Create `src/req_update_check/exceptions.py`
- [ ] Update `src/req_update_check/cli.py`
  - [ ] Add `--ai-check` argument
  - [ ] Add `--api-key` argument
  - [ ] Initialize provider if AI enabled
- [ ] Update `pyproject.toml`
  - [ ] Add `anthropic` to `[project.optional-dependencies]`
- [ ] Write tests
  - [ ] `tests/test_ai_providers/test_base.py`
  - [ ] `tests/test_ai_providers/test_claude.py`
  - [ ] `tests/test_ai_providers/test_factory.py`
  - [ ] `tests/test_auth.py`

### Phase 2: Analysis Components
- [ ] Create `src/req_update_check/changelog_fetcher.py`
  - [ ] Direct URL fetching
  - [ ] GitHub releases API
  - [ ] Version range extraction
  - [ ] Caching
- [ ] Create `src/req_update_check/code_scanner.py`
  - [ ] Import finder
  - [ ] Usage example extractor
  - [ ] Usage hash for cache invalidation
- [ ] Create `src/req_update_check/prompts.py`
  - [ ] System prompt
  - [ ] User prompt template
- [ ] Create `src/req_update_check/ai_analyzer.py`
  - [ ] `ChangelogAnalyzer` class
  - [ ] `AnalysisResult` dataclass
  - [ ] Pipeline orchestration
- [ ] Create `src/req_update_check/formatting.py`
  - [ ] `format_ai_analysis()` function
- [ ] Update `src/req_update_check/core.py`
  - [ ] Add `ai_provider` parameter to `__init__`
  - [ ] Add `analyze_update_with_ai()` method
  - [ ] Update `report()` to show AI analysis
- [ ] Write tests
  - [ ] `tests/test_changelog_fetcher.py`
  - [ ] `tests/test_code_scanner.py`
  - [ ] `tests/test_prompts.py`
  - [ ] `tests/test_analyzer.py`
  - [ ] `tests/test_formatting.py`

### Phase 3: Multi-Provider Support
- [ ] Create `src/req_update_check/ai_providers/gemini.py`
- [ ] Create `src/req_update_check/ai_providers/openai.py`
- [ ] Create `src/req_update_check/ai_providers/custom.py`
- [ ] Update `src/req_update_check/ai_providers/factory.py`
  - [ ] Add all providers to registry
  - [ ] Handle provider-specific config
- [ ] Create `src/req_update_check/config.py`
  - [ ] Load config from TOML
  - [ ] Merge with defaults
- [ ] Update `src/req_update_check/auth.py`
  - [ ] Support all provider env vars
  - [ ] Load from config file
- [ ] Update `src/req_update_check/cli.py`
  - [ ] Add `--ai-provider` argument
  - [ ] Add `--ai-model` argument
- [ ] Update `pyproject.toml`
  - [ ] Add `google-generativeai` to optional deps
  - [ ] Add `openai` to optional deps
- [ ] Write tests
  - [ ] `tests/test_ai_providers/test_gemini.py`
  - [ ] `tests/test_ai_providers/test_openai.py`
  - [ ] `tests/test_ai_providers/test_custom.py`
  - [ ] `tests/test_config.py`

### Phase 4: Polish & Documentation
- [ ] Error handling
  - [ ] Retry logic with exponential backoff
  - [ ] Rate limit handling
  - [ ] Graceful degradation
  - [ ] Clear error messages
- [ ] Features
  - [ ] Cost estimation (`--cost-estimate`)
  - [ ] Progress indicators for multiple packages
  - [ ] User consent prompt
  - [ ] Skip packages list in config
- [ ] Documentation
  - [ ] Update `README.md` with AI features
  - [ ] Create `docs/AI_PROVIDERS.md`
  - [ ] Update `CLAUDE.md` with architecture
  - [ ] Add example outputs
  - [ ] Security best practices
- [ ] Testing
  - [ ] Integration tests with mocked APIs
  - [ ] Manual testing with real packages
  - [ ] Security review
  - [ ] Performance testing
- [ ] Release
  - [ ] Version bump to 0.3.0
  - [ ] Changelog update
  - [ ] PyPI release

---

## Success Metrics

### User Experience
- ✅ Clear, actionable recommendations in < 30 seconds
- ✅ Cost < $0.10 per package analysis
- ✅ Cache hit rate > 80% for repeated checks
- ✅ Zero false positives on "safe" recommendations

### Technical
- ✅ Test coverage > 90%
- ✅ Support 3+ AI providers
- ✅ Graceful degradation (no crashes on API failures)
- ✅ Response time < 30s for single package analysis
- ✅ Compatible with Python 3.9+

### Adoption
- ✅ Documentation for all providers
- ✅ Example configs for common setups
- ✅ Clear error messages for all failure modes
- ✅ Works with existing caching infrastructure

---

## Future Enhancements (Post-MVP)

### Short Term (v0.4.0)
1. Batch parallel analysis with progress bar
2. Interactive package selection mode
3. Export reports (JSON, HTML)
4. GitHub PR comments integration

### Medium Term (v0.5.0)
1. Auto-fix mode (generate code patches)
2. Plugin system for custom providers
3. Multi-language support (package.json, Cargo.toml)
4. Team cache sharing

### Long Term (v1.0.0)
1. Machine learning on user feedback
2. Predictive upgrade risk scores
3. Automated regression testing
4. IDE extensions (VS Code, PyCharm)

---

## Appendix: Provider Comparison

| Provider | Model | Cost/Analysis | Speed | Code Understanding | Free Tier |
|----------|-------|---------------|-------|-------------------|-----------|
| Claude | Sonnet 3.5 | ~$0.05 | Medium | Excellent | No |
| Gemini | 2.0 Flash | ~$0.01 | Fast | Good | Yes (limited) |
| OpenAI | GPT-4o | ~$0.08 | Medium | Excellent | No |
| Custom | Varies | $0 (local) | Varies | Varies | N/A |

**Recommendation:**
- **Default:** Claude (best accuracy)
- **Budget:** Gemini (good balance)
- **Privacy:** Custom with Ollama (local)
- **Enterprise:** OpenAI (compliance reasons)
