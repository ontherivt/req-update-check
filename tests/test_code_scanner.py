from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from req_update_check.code_scanner import CodebaseScanner
from req_update_check.code_scanner import ImportLocation
from req_update_check.code_scanner import UsageReport


class TestCodebaseScanner(unittest.TestCase):
    """Tests for codebase scanning functionality"""

    def setUp(self):
        """Create temporary test directory with Python files"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_root = Path(self.temp_dir.name)

        # Create test files
        (self.test_root / "test1.py").write_text(
            "import requests\nfrom requests import get\n\nresponse = requests.get('http://example.com')\n"
        )

        (self.test_root / "test2.py").write_text("from flask import Flask\nimport flask\n\napp = Flask(__name__)\n")

        (self.test_root / "test3.py").write_text("# No imports here\nprint('hello')\n")

        # Create subdirectory
        subdir = self.test_root / "subdir"
        subdir.mkdir()
        (subdir / "test4.py").write_text("import numpy as np\nfrom numpy import array\n")

    def tearDown(self):
        """Clean up temporary directory"""
        self.temp_dir.cleanup()

    def test_find_package_usage_found(self):
        """Test finding package usage when it exists"""
        scanner = CodebaseScanner(str(self.test_root))
        report = scanner.find_package_usage("requests")

        self.assertEqual(report.package_name, "requests")
        self.assertEqual(report.files_count, 1)
        self.assertEqual(len(report.import_locations), 2)  # Two import statements

    def test_find_package_usage_not_found(self):
        """Test finding package that doesn't exist in codebase"""
        scanner = CodebaseScanner(str(self.test_root))
        report = scanner.find_package_usage("nonexistent")

        self.assertEqual(report.package_name, "nonexistent")
        self.assertEqual(report.files_count, 0)
        self.assertEqual(len(report.import_locations), 0)

    def test_find_package_usage_multiple_files(self):
        """Test finding package used in subdirectories"""
        scanner = CodebaseScanner(str(self.test_root))
        report = scanner.find_package_usage("flask")

        self.assertEqual(report.package_name, "flask")
        self.assertGreater(report.files_count, 0)

    def test_find_python_files_excludes_venv(self):
        """Test that .venv and other directories are excluded"""
        # Create .venv directory with Python file
        venv_dir = self.test_root / ".venv"
        venv_dir.mkdir()
        (venv_dir / "test.py").write_text("import something\n")

        scanner = CodebaseScanner(str(self.test_root))
        py_files = scanner._find_python_files()

        # Should not include .venv files
        venv_files = [f for f in py_files if ".venv" in str(f)]
        self.assertEqual(len(venv_files), 0)

    def test_usage_report_to_prompt_text_with_imports(self):
        """Test formatting usage report for AI prompt"""
        locations = [
            ImportLocation(file="test.py", line=1, content="import requests"),
            ImportLocation(file="test.py", line=2, content="from requests import get"),
        ]

        report = UsageReport(
            package_name="requests",
            files_count=1,
            import_locations=locations,
            usage_examples=["test.py:1\n```python\nimport requests\n```"],
        )

        text = report.to_prompt_text()

        self.assertIn("requests", text)
        self.assertIn("1 files", text)
        self.assertIn("test.py:1", text)
        self.assertIn("import requests", text)

    def test_usage_report_to_prompt_text_not_found(self):
        """Test formatting when package not found"""
        report = UsageReport(
            package_name="missing",
            files_count=0,
            import_locations=[],
            usage_examples=[],
        )

        text = report.to_prompt_text()

        self.assertIn("not found", text)
        self.assertIn("missing", text)

    def test_get_usage_hash(self):
        """Test generating usage hash for cache invalidation"""
        scanner = CodebaseScanner(str(self.test_root))
        hash1 = scanner.get_usage_hash("requests")

        # Hash should be consistent
        hash2 = scanner.get_usage_hash("requests")
        self.assertEqual(hash1, hash2)

        # Hash for different package should be different
        hash3 = scanner.get_usage_hash("flask")
        self.assertNotEqual(hash1, hash3)

    def test_find_imports_with_hyphenated_package(self):
        """Test finding imports for packages with hyphens (converted to underscores)"""
        # Create file with package that has underscore
        (self.test_root / "test_hyphen.py").write_text("from google_auth import credentials\n")

        scanner = CodebaseScanner(str(self.test_root))

        # Should find it when searching with hyphen
        report = scanner.find_package_usage("google-auth")
        self.assertGreater(len(report.import_locations), 0)


if __name__ == "__main__":
    unittest.main()
