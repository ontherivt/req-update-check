from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("req_update_check")


@dataclass
class ImportLocation:
    """Represents a single import location in the codebase"""

    file: str
    line: int
    content: str


@dataclass
class UsageReport:
    """Report of how a package is used in the codebase"""

    package_name: str
    files_count: int
    import_locations: list[ImportLocation]
    usage_examples: list[str]

    def to_prompt_text(self) -> str:
        """
        Format usage report for inclusion in AI prompt

        Returns:
            Formatted text suitable for AI analysis
        """
        if not self.import_locations:
            return f"Package '{self.package_name}' not found in codebase."

        text = f"Found {self.files_count} files using '{self.package_name}':\n\n"

        # Show import locations (limit to 10)
        text += "Import locations:\n"
        for imp in self.import_locations[:10]:
            text += f"  • {imp.file}:{imp.line} - {imp.content}\n"

        if len(self.import_locations) > 10:
            text += f"  ... and {len(self.import_locations) - 10} more\n"

        # Show usage examples
        if self.usage_examples:
            text += "\nUsage examples:\n"
            text += "\n\n".join(self.usage_examples)

        return text


class CodebaseScanner:
    """Scans codebase for package usage patterns"""

    def __init__(self, root_path: str = "."):
        """
        Initialize codebase scanner

        Args:
            root_path: Root directory to scan (defaults to current directory)
        """
        self.root_path = Path(root_path).resolve()

    def find_package_usage(self, package_name: str) -> UsageReport:
        """
        Find how the package is used in the codebase

        Args:
            package_name: Name of the package to search for

        Returns:
            UsageReport with import locations and usage examples
        """
        logger.debug(f"Scanning codebase for '{package_name}' usage")

        # Find Python files
        py_files = self._find_python_files()
        logger.debug(f"Found {len(py_files)} Python files to scan")

        # Search for imports
        imports = self._find_imports(package_name, py_files)
        logger.debug(f"Found {len(imports)} import locations")

        # Extract usage examples
        examples = self._extract_usage_examples(package_name, imports)

        return UsageReport(
            package_name=package_name,
            files_count=len({imp.file for imp in imports}),
            import_locations=imports,
            usage_examples=examples,
        )

    def get_usage_hash(self, package_name: str) -> str:
        """
        Generate hash of package usage for cache invalidation
        Hash is based on: files using package and their modification times

        Args:
            package_name: Name of the package

        Returns:
            8-character hash string
        """
        py_files = self._find_python_files()
        imports = self._find_imports(package_name, py_files)

        hash_input = []
        for imp in imports:
            file_path = self.root_path / imp.file
            try:
                mtime = file_path.stat().st_mtime if file_path.exists() else 0
                hash_input.append(f"{imp.file}:{mtime}")
            except Exception:  # noqa: BLE001, S112
                continue

        # Generate hash
        hash_str = "".join(sorted(hash_input))
        return hashlib.md5(hash_str.encode()).hexdigest()[:8]  # noqa: S324

    def _find_python_files(self) -> list[Path]:
        """
        Find all Python files in the codebase

        Returns:
            List of Python file paths
        """
        try:
            # Exclude common directories that shouldn't be scanned
            excludes = {".git", ".venv", "venv", "__pycache__", "node_modules", ".tox", "build", "dist", ".eggs"}

            py_files = []
            for file in self.root_path.rglob("*.py"):
                # Check if any parent directory is in excludes
                if any(part in excludes for part in file.parts):
                    continue
                py_files.append(file)

            return py_files  # noqa: TRY300
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Error finding Python files: {e}")
            return []

    def _find_imports(self, package_name: str, files: list[Path]) -> list[ImportLocation]:
        """
        Find all imports of the package in given files

        Args:
            package_name: Package name to search for
            files: List of files to search

        Returns:
            List of import locations
        """
        locations = []

        # Normalize package name - handle both hyphenated and underscored
        package_variants = [
            package_name,
            package_name.replace("-", "_"),
            package_name.replace("_", "-"),
        ]

        # Create search patterns
        patterns = []
        for variant in set(package_variants):  # Remove duplicates
            patterns.extend(
                [
                    f"import {variant}",
                    f"from {variant}",
                    f"import {variant}.",
                    f"from {variant}.",
                ],
            )

        for file in files:
            try:
                with open(file, encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        # Strip leading whitespace for pattern matching
                        stripped = line.lstrip()

                        # Check if any pattern matches
                        for pattern in patterns:
                            if stripped.startswith(pattern):
                                try:
                                    relative_path = str(file.relative_to(self.root_path))
                                except ValueError:
                                    relative_path = str(file)

                                locations.append(
                                    ImportLocation(
                                        file=relative_path,
                                        line=line_num,
                                        content=line.strip(),
                                    ),
                                )
                                break  # Only count each line once
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Error reading {file}: {e}")
                continue

        return locations

    def _extract_usage_examples(
        self,
        package_name: str,
        imports: list[ImportLocation],
    ) -> list[str]:
        """
        Extract code snippets showing how package is used
        Limits to ~50 lines total to keep prompt size manageable

        Args:
            package_name: Package name
            imports: List of import locations

        Returns:
            List of formatted code examples
        """
        examples = []
        max_examples = 5
        lines_per_example = 10

        for imp in imports[:max_examples]:
            try:
                file_path = self.root_path / imp.file
                with open(file_path, encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()

                    # Get context around the import
                    start = max(0, imp.line - 3)  # 2 lines before
                    end = min(len(lines), imp.line + lines_per_example - 2)  # 8 lines after

                    snippet = "".join(lines[start:end])

                    # Format with file location and code block
                    example = f"{imp.file}:{imp.line}\n```python\n{snippet}```"
                    examples.append(example)
            except Exception as e:  # noqa: BLE001, PERF203
                logger.debug(f"Error extracting example from {imp.file}: {e}")
                continue

        return examples
