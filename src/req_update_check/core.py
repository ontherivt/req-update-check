from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import requests

from .ai_analyzer import ChangelogAnalyzer
from .cache import FileCache
from .formatting import format_ai_analysis

logger = logging.getLogger("req_update_check")

try:
    import tomllib

    TOMLLIB = True
except ModuleNotFoundError:
    TOMLLIB = False


class Requirements:
    pypi_index = "https://pypi.python.org/simple/"
    pypi_package_base = "https://pypi.python.org/project/"
    pypi_json_api = "https://pypi.org/pypi/"
    headers = {"Content-Type": "json", "Accept": "application/vnd.pypi.simple.v1+json"}

    def __init__(
        self,
        path: str,
        allow_cache: bool = True,
        cache_dir: str | None = None,
        ai_provider=None,
    ):
        self._index = False
        self._get_packages = False

        self.path = path
        self.packages = None
        self.package_index = set()
        self.allow_cache = allow_cache
        self.updates = []
        cache_dir = cache_dir or ".req-check-cache"
        self.cache = FileCache(cache_dir) if allow_cache else None
        self.ai_provider = ai_provider

        # Initialize AI analyzer if provider is available
        self.ai_analyzer = None
        if ai_provider:
            # Get codebase path (directory containing the requirements file)
            codebase_path = str(Path(path).parent.resolve())
            self.ai_analyzer = ChangelogAnalyzer(
                provider=ai_provider,
                cache=self.cache,
                codebase_path=codebase_path,
            )

    def get_index(self):
        if self._index:
            return
        self._index = True
        if self.allow_cache and self.cache:
            package_index = self.cache.get("package-index")
            if package_index:
                self.package_index = set(package_index)
                return

        res = requests.get(self.pypi_index, headers=self.headers, timeout=10)
        package_index = res.json()["projects"]
        for package in package_index:
            self.package_index.add(package["name"])

        if self.cache:
            self.cache.set("package-index", list(self.package_index))

    def get_packages(self):
        if self._get_packages:
            return None
        self._get_packages = True
        self.get_index()
        try:
            # if it's a toml file, we should handle it differently
            if self.path.endswith(".toml"):
                if not TOMLLIB:
                    msg = "tomllib is not available before python 3.11, cannot parse pyproject.toml files."
                    logger.info(msg)
                    sys.exit(1)
                with open(self.path, "rb") as f:
                    file_data = tomllib.load(f)
                    if "project" not in file_data or "dependencies" not in file_data["project"]:
                        msg = f"File {self.path} is not a valid pyproject.toml file."
                        logger.info(msg)
                        sys.exit(1)
                    requirements = file_data["project"]["dependencies"]
                    # also grab dependency groups
                    if "dependency-groups" in file_data:
                        for reqs in file_data["dependency-groups"].values():
                            requirements.extend(reqs)
            else:
                with open(self.path) as file:
                    requirements = file.readlines()
        except FileNotFoundError:
            msg = f"File {self.path} not found."
            logger.info(msg)
            sys.exit(1)

        packages = []
        for req in requirements:
            if req.startswith("#") or req in ["", "\n"]:
                continue
            # remove inline comments
            req_ = req.split("#")[0]
            packages.append(req_.strip().split("=="))

        self.packages = packages
        return packages

    def get_latest_version(self, package_name):
        if self.allow_cache and self.cache:
            latest_version = self.cache.get(f"package:{package_name}")
            if latest_version:
                return latest_version

        res = requests.get(f"{self.pypi_index}{package_name}/", headers=self.headers, timeout=10)
        versions = res.json()["versions"]
        # start from the end and find the first version that is not a pre-release
        for version in reversed(versions):
            if not any(x in version for x in ["a", "b", "rc"]):
                if self.cache:
                    self.cache.set(f"package:{package_name}", version)
                return version
        return None

    def check_packages(self):
        self.get_packages()
        for package in self.packages:
            self.check_package(package)

    def check_package(self, package: list[str, str]):
        expected_length = 2
        if len(package) == expected_length:
            package_name, package_version = package
        else:
            return

        # check for optional dependencies
        if "[" in package_name:
            package_name, optional_deps = package_name.split("[")
            logger.info(f"Skipping optional packages '{optional_deps.replace(']', '')}' from {package_name}")

        # check if package is in the index
        if package_name not in self.package_index:
            msg = f"Package {package_name} not found in the index."
            logger.info(msg)
            return

        latest_version = self.get_latest_version(package_name)
        if latest_version != package_version:
            level = self.check_major_minor(package_version, latest_version)
            self.updates.append(
                (package_name, package_version, latest_version, level),
            )

    def report(self, ai_check_packages: list[str] | None = None):
        if not self.updates:
            logger.info("All packages are up to date.")
            return

        logger.info("The following packages need to be updated:\n")
        analyzing_all = ai_check_packages == ["*"]

        for idx, package in enumerate(self.updates):
            # Add separator line before each package (except the first)
            if idx > 0:
                separator = "\n" + "=" * 80 + "\n"
                logger.info(separator)

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
            should_analyze = ai_check_packages is not None and (
                ai_check_packages == ["*"] or package_name in ai_check_packages
            )

            if should_analyze and self.ai_analyzer:
                logger.info("\n\t🤖 Analyzing with AI...")
                analysis = self._analyze_update_with_ai(
                    package_name,
                    current_version,
                    latest_version,
                    level,
                    links,
                )
                if analysis:
                    logger.info(format_ai_analysis(analysis))

                # Add delay between API calls when analyzing all packages to avoid rate limits
                if analyzing_all and idx < len(self.updates) - 1:
                    time.sleep(1)  # 1 second delay between packages

    def get_package_info(self, package_name: str) -> dict:
        """Get package information using PyPI JSON API."""
        if self.allow_cache and self.cache:
            info = self.cache.get(f"package-info:{package_name}")
            if info:
                return info

        try:
            res = requests.get(f"{self.pypi_json_api}{package_name}/json", timeout=10)
            res.raise_for_status()
            data = res.json()

            info = {}
            project_info = data.get("info", {})
            project_urls = project_info.get("project_urls", {})

            # Try to get homepage from multiple sources
            homepage = project_info.get("home_page") or project_urls.get("Homepage")
            if homepage and homepage != "UNKNOWN":
                info["homepage"] = homepage

            # Try to get changelog from project URLs
            for key in ["Changelog", "Change Log", "Changes", "Release Notes", "Releases"]:
                changelog = project_urls.get(key)
                if changelog:
                    info["changelog"] = changelog
                    break

            if self.cache:
                self.cache.set(f"package-info:{package_name}", info)
        except (requests.RequestException, KeyError, ValueError):
            return {}
        else:
            return info

    def check_major_minor(self, current_version, latest_version):
        current_major, current_minor, _current_patch, *_ = current_version.split(".") + ["0"] * 3
        latest_major, latest_minor, _latest_patch, *_ = latest_version.split(".") + ["0"] * 3

        if current_major != latest_major:
            return "major"
        if current_minor != latest_minor:
            return "minor"
        return "patch"

    def _analyze_update_with_ai(
        self,
        package_name: str,
        current_version: str,
        latest_version: str,
        update_level: str,
        package_info: dict,
    ):
        """
        Perform AI analysis on a package update

        Args:
            package_name: Name of the package
            current_version: Current version
            latest_version: Latest version
            update_level: Type of update (major/minor/patch)
            package_info: Package info dict with homepage/changelog URLs

        Returns:
            AnalysisResult or None if analysis fails
        """
        if not self.ai_analyzer:
            return None

        try:
            changelog_url = package_info.get("changelog")
            homepage_url = package_info.get("homepage")

            # Use the full analyzer pipeline (Phase 2 implementation)
            return self.ai_analyzer.analyze_update(
                package_name=package_name,
                current_version=current_version,
                latest_version=latest_version,
                update_level=update_level,
                changelog_url=changelog_url,
                homepage_url=homepage_url,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"AI analysis failed for {package_name}: {e}")
            return None
