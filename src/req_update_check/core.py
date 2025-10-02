from __future__ import annotations

import logging
import subprocess
import sys
from shutil import which

import requests

from .cache import FileCache

logger = logging.getLogger("req_update_check")

try:
    import tomllib

    TOMLLIB = True
except ModuleNotFoundError:
    TOMLLIB = False


def get_pip_path():
    pip_path = which("pip")
    if not pip_path:
        logger.error("pip executable not found")
        return None
    return pip_path


class Requirements:
    pypi_index = "https://pypi.python.org/simple/"
    pypi_package_base = "https://pypi.python.org/project/"
    headers = {"Content-Type": "json", "Accept": "application/vnd.pypi.simple.v1+json"}

    def __init__(
        self,
        path: str,
        allow_cache: bool = True,
        cache_dir: str | None = None,
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

    def report(self):
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
            msg += "\n"
            logger.info(msg)

    def get_package_info(self, package_name: str) -> dict:
        """Get package information using pip show command."""
        if self.allow_cache and self.cache:
            info = self.cache.get(f"package-info:{package_name}")
            if info:
                return info
        pip_path = get_pip_path()
        if not pip_path:
            return {}
        try:
            # ruff: noqa: S603
            result = subprocess.run(
                [pip_path, "show", package_name, "--verbose"],
                capture_output=True,
                text=True,
                check=True,
            )

            info = {}

            for line in result.stdout.split("\n"):
                if "Home-page: " in line:
                    info["homepage"] = line.split(": ", 1)[1].strip()

                if "Changelog," in line or "change-log, " in line.lower():
                    info["changelog"] = line.split(", ", 1)[1].strip()

                if "Homepage, " in line:
                    info["homepage"] = line.split(", ", 1)[1].strip()
            if self.cache:
                self.cache.set(f"package-info:{package_name}", info)
            else:
                return info
        except subprocess.CalledProcessError:
            return {}

    def check_major_minor(self, current_version, latest_version):
        current_major, current_minor, _current_patch, *_ = current_version.split(".") + ["0"] * 3
        latest_major, latest_minor, _latest_patch, *_ = latest_version.split(".") + ["0"] * 3

        if current_major != latest_major:
            return "major"
        if current_minor != latest_minor:
            return "minor"
        return "patch"
