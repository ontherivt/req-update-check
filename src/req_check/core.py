import json
import subprocess
import sys
from typing import Optional
import requests
from .cache import FileCache


class Requirements:
    pypi_index = "https://pypi.python.org/simple/"
    pypi_package_base = "https://pypi.python.org/project/"
    headers = {"Content-Type": "json", 'Accept': 'application/vnd.pypi.simple.v1+json'}

    def __init__(self, path: str, allow_cache: bool = True, cache_dir: Optional[str] = None):
        self.path = path
        self.packages = self.get_packages()
        self.package_index = set()
        self.allow_cache = allow_cache
        self.updates = []
        cache_dir = cache_dir or ".req-check-cache"
        self.cache = FileCache(cache_dir) if allow_cache else None

        self.get_index()

    def get_index(self):
        if self.allow_cache and self.cache:
            package_index = self.cache.get("package-index")
            if package_index:
                self.package_index = set(package_index)
                return

        res = requests.get(self.pypi_index, headers=self.headers)
        package_index = res.json()['projects']
        for package in package_index:
            self.package_index.add(package['name'])

        if self.cache:
            self.cache.set("package-index", list(self.package_index))

    def get_packages(self):
        try:
            with open(self.path, 'r') as file:
                requirements = file.readlines()
        except FileNotFoundError:
            print(f"File {self.path} not found.")
            sys.exit(1)

        packages = []
        for req in requirements:
            if req.startswith("#") or req == "" or req == "\n":
                continue
            # remove inline comments
            req = req.split("#")[0]
            packages.append(req.strip().split("=="))

        return packages

    def get_latest_version(self, package_name):
        if self.allow_cache and self.cache:
            latest_version = self.cache.get(f"package:{package_name}")
            if latest_version:
                return latest_version

        res = requests.get(f"{self.pypi_index}{package_name}/", headers=self.headers)
        versions = res.json()['versions']
        # start from the end and find the first version that is not a pre-release
        for version in reversed(versions):
            if not any([x in version for x in ["a", "b", "rc"]]):
                if self.cache:
                    self.cache.set(f"package:{package_name}", version)
                return version

    def check_packages(self):
        for package in self.packages:
            if len(package) == 2:
                package_name, package_version = package
            else:
                continue

            # check if package is in the index
            if package_name not in self.package_index:
                print(f"Package {package_name} not found in the index.")
                continue

            latest_version = self.get_latest_version(package_name)
            if latest_version != package_version:
                level = self.check_major_minor(package_version, latest_version)
                self.updates.append((package_name, package_version, latest_version, level))

    def report(self):
        if not self.updates:
            print("All packages are up to date.")
            return

        print("The following packages need to be updated:\n")
        for package in self.updates:
            package_name, current_version, latest_version, level = package
            print(f"{package_name}: {current_version} -> {latest_version} [{level}]")
            print(f"    Pypi page: {self.pypi_package_base}{package_name}/")
            links = self.get_package_info(package_name)
            if links:
                if links.get("homepage"):
                    print(f"    Homepage: {links['homepage']}")
                if links.get("changelog"):
                    print(f"    Changelog: {links['changelog']}")
            print()

    def get_package_info(self, package_name: str) -> dict:
        """Get package information using pip show command."""
        if self.allow_cache and self.cache:
            info = self.cache.get(f"package-info:{package_name}")
            if info:
                return info
        try:
            result = subprocess.run(
                ['pip', 'show', package_name, '--verbose'],
                capture_output=True,
                text=True,
                check=True
            )

            info = {}

            for line in result.stdout.split('\n'):
                if "Home-page: " in line:
                    info['homepage'] = line.split(': ', 1)[1].strip()

                if "Changelog," in line or "change-log, " in line.lower():
                    info['changelog'] = line.split(', ', 1)[1].strip()

                if "Homepage, " in line:
                    info['homepage'] = line.split(', ', 1)[1].strip()
            if self.cache:
                self.cache.set(f"package-info:{package_name}", info)
            return info
        except subprocess.CalledProcessError:
            return {}

    def check_major_minor(self, current_version, latest_version):
        current_major, current_minor, current_patch, *_ = current_version.split(".") + ["0"] * 3
        latest_major, latest_minor, latest_patch, *_ = latest_version.split(".") + ["0"] * 3

        if current_major != latest_major:
            return "major"
        if current_minor != latest_minor:
            return "minor"
        return "patch"

