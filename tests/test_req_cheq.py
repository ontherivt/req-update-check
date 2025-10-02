import importlib
import sys
import unittest
from unittest.mock import mock_open
from unittest.mock import patch

from src.req_update_check import core
from src.req_update_check.cache import FileCache
from src.req_update_check.cli import main
from src.req_update_check.core import Requirements


class TestFileCache(unittest.TestCase):
    def setUp(self):
        self.cache = FileCache(cache_dir=".test-cache")
        self.test_key = "test-key"
        self.test_value = {"data": "test"}

    def tearDown(self):
        self.cache.clear()

    def test_set_and_get(self):
        self.cache.set(self.test_key, self.test_value)
        result = self.cache.get(self.test_key)
        self.assertEqual(result, self.test_value)

    def test_expired_cache(self):
        with patch("time.time", return_value=100):
            self.cache.set(self.test_key, self.test_value)

        with patch("time.time", return_value=5000):
            result = self.cache.get(self.test_key)
            self.assertIsNone(result)

    def test_invalid_cache(self):
        cache_file = self.cache.cache_dir / f"{self.test_key}.json"
        cache_file.write_text("invalid json")
        result = self.cache.get(self.test_key)
        self.assertIsNone(result)


class TestRequirements(unittest.TestCase):
    def setUp(self):
        self.req_content = """
requests==2.26.0
flask==2.0.1
# comment line
pytest==6.2.4  # inline comment
"""
        self.toml_content = """
[project]
dependencies = [
    "requests==2.26.0",
    "flask==2.0.1",
]

[dependency-groups]
group1 = ["pytest==6.2.4"]
group2 = ["numpy==1.21.0"]

"""

        self.mock_index = {
            "projects": [
                {"name": "requests"},
                {"name": "flask"},
                {"name": "pytest"},
            ],
        }
        self.mock_versions = {
            "versions": ["2.26.0", "2.27.0", "2.28.0"],
        }

        self.requirements = Requirements("requirements.txt", allow_cache=False)

    @patch.object(Requirements, "get_index")
    @patch("builtins.open", new_callable=mock_open)
    def test_get_packages(self, mock_file, mock_get_index):
        mock_file.return_value.readlines.return_value = self.req_content.split("\n")
        req = Requirements("requirements.txt", allow_cache=False)
        req.check_packages()
        expected = [
            ["requests", "2.26.0"],
            ["flask", "2.0.1"],
            ["pytest", "6.2.4"],
        ]
        self.assertEqual(req.packages, expected)

    @unittest.skipIf(sys.version_info < (3, 11), "Test requires Python 3.11 or newer")
    @patch.object(Requirements, "get_index")
    @patch("builtins.open", new_callable=mock_open)
    def test_get_packages__toml(self, mock_file, mock_get_index):
        mock_file.return_value.read.return_value = self.toml_content.encode("utf-8")
        req = Requirements("pyproject.toml", allow_cache=False)
        req.check_packages()
        expected = [
            ["requests", "2.26.0"],
            ["flask", "2.0.1"],
            ["pytest", "6.2.4"],
            ["numpy", "1.21.0"],
        ]
        self.assertEqual(req.packages, expected)

    def test_get_packages__toml__before_python_311(self):
        # Make tomllib "unavailable" and reload so TOMLLIB is recomputed.
        with patch.dict(sys.modules, {"tomllib": None}):
            importlib.reload(core)
            self.assertFalse(core.TOMLLIB, "Expected TOMLLIB to be False after reload")

            # Patch *after* reload, and patch the class on the reloaded module.
            with (
                patch.object(core.Requirements, "get_index"),
                patch("builtins.open", new_callable=mock_open) as mock_file,
            ):
                with self.assertRaises(SystemExit) as cm:
                    # IMPORTANT: call the reloaded class
                    req = core.Requirements("pyproject.toml", allow_cache=False)
                    req.check_packages()

                self.assertEqual(cm.exception.code, 1)
                mock_file.assert_not_called()

    @patch("requests.get")
    def test_get_index(self, mock_get):
        mock_get.return_value.json.side_effect = [self.mock_index] + [self.mock_versions] * 3
        with patch("builtins.open", new_callable=mock_open) as mock_file:
            mock_file.return_value.readlines.return_value = self.req_content.split("\n")
            req = Requirements("requirements.txt", allow_cache=False)
            req.check_packages()
            self.assertEqual(req.package_index, {"requests", "flask", "pytest"})

    @patch.object(Requirements, "get_index")
    @patch("requests.get")
    def test_get_latest_version(self, mock_get, mock_get_index):
        mock_get.return_value.json.return_value = self.mock_versions
        with patch("builtins.open", new_callable=mock_open) as mock_file:
            mock_file.return_value.readlines.return_value = self.req_content.split("\n")
            req = Requirements("requirements.txt", allow_cache=False)
            latest = req.get_latest_version("requests")
            self.assertEqual(latest, "2.28.0")

    @patch.object(Requirements, "get_index")
    def test_check_major_minor(self, mock_get_index):
        with patch("builtins.open", new_callable=mock_open) as mock_file:
            mock_file.return_value.readlines.return_value = self.req_content.split("\n")
            req = Requirements("requirements.txt", allow_cache=False)

            self.assertEqual(req.check_major_minor("1.0.0", "2.0.0"), "major")
            self.assertEqual(req.check_major_minor("1.0.0", "1.1.0"), "minor")
            self.assertEqual(req.check_major_minor("1.0.0", "1.0.1"), "patch")

    def test_optional_dependencies(self):
        package = ["psycopg2[binary]", "2.9.1"]
        with self.assertLogs("req_update_check", level="INFO") as cm:
            self.requirements.check_package(package)
        self.assertIn("Skipping optional packages 'binary' from psycopg2", cm.output[0])

        package = ["psycopg2", "2.9.1"]
        with self.assertLogs("req_update_check", level="INFO") as cm:
            self.requirements.check_package(package)

        self.assertNotIn("Skipping optional packages", cm.output[0])


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.req_content = "requests==2.26.0\nflask==2.0.1\n"
        self.requirements_file = "requirements.txt"

    @patch("sys.argv", ["req-check", "requirements.txt"])
    @patch("builtins.print")
    @patch("src.req_update_check.cli.Requirements")
    def test_main_default_args(self, mock_requirements, mock_print):
        mock_instance = mock_requirements.return_value
        main()
        mock_requirements.assert_called_with(
            "requirements.txt",
            allow_cache=True,
            cache_dir=None,
        )
        mock_instance.check_packages.assert_called_once()
        mock_instance.report.assert_called_once()

    @patch("sys.argv", ["req-check", "requirements.txt", "--no-cache"])
    @patch("builtins.print")
    @patch("src.req_update_check.cli.Requirements")
    def test_main_no_cache(self, mock_requirements, mock_print):
        main()
        mock_requirements.assert_called_with(
            "requirements.txt",
            allow_cache=False,
            cache_dir=None,
        )

    @patch(
        "sys.argv",
        ["req-check", "requirements.txt", "--cache-dir", "/custom/cache"],
    )
    @patch("builtins.print")
    @patch("src.req_update_check.cli.Requirements")
    def test_main_custom_cache_dir(self, mock_requirements, mock_print):
        main()
        mock_requirements.assert_called_with(
            "requirements.txt",
            allow_cache=True,
            cache_dir="/custom/cache",
        )


if __name__ == "__main__":
    unittest.main()
