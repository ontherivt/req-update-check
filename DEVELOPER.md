# Developer Guide

This document contains information for developers working on `req-update-check`, including release processes, CI/CD workflows, and development best practices.

## Table of Contents

- [Release Process](#release-process)
  - [Automated Releases with Release Please](#automated-releases-with-release-please)
  - [Manual Release Process](#manual-release-process)
- [CI/CD Workflows](#cicd-workflows)
- [Development Workflow](#development-workflow)
- [Publishing to PyPI](#publishing-to-pypi)

## Release Process

This project supports two release methods: automated releases using Release Please (recommended) and manual releases.

### Automated Releases with Release Please

Release Please automates the release process based on Conventional Commits. It creates and maintains a release PR that tracks changes and updates the changelog.

#### How It Works

1. **Commit Convention**: Use [Conventional Commits](https://www.conventionalcommits.org/) format for your commits:
   ```
   feat: add new feature
   fix: bug fix
   docs: documentation changes
   chore: maintenance tasks
   ```

2. **Release PR Creation**: When commits are merged to `main`, Release Please automatically:
   - Creates/updates a release PR
   - Updates version in `pyproject.toml`
   - Generates/updates `CHANGELOG.md`
   - Determines version bump based on commit types:
     - `fix:` → patch version (0.2.0 → 0.2.1)
     - `feat:` → minor version (0.2.0 → 0.3.0)
     - `BREAKING CHANGE:` or `!` → major version (0.2.0 → 1.0.0)

3. **Releasing**: When you merge the Release Please PR:
   - A new GitHub release is created
   - A git tag (e.g., `v0.2.1`) is created
   - The manual release workflow is triggered (see `.github/workflows/release.yml`)

#### Configuration

Release Please is configured in `.github/workflows/release-please.yml`:

```yaml
- uses: googleapis/release-please-action@v4.3.0
  with:
    release-type: python  # Automatically handles pyproject.toml versioning
```

#### Example Workflow

```bash
# 1. Make changes with conventional commits
git commit -m "feat: add support for poetry.lock files"
git commit -m "fix: handle missing package metadata gracefully"

# 2. Push to main (via PR)
git push origin feature-branch
# Merge PR to main

# 3. Release Please creates/updates a release PR automatically

# 4. Review and merge the release PR when ready

# 5. GitHub release is created automatically with release notes
```

### Manual Release Process

If you need to create a release manually (not recommended for regular releases):

#### Prerequisites

- Write access to the repository
- Appropriate version number decided
- All changes merged to `main` branch

#### Steps

1. **Update Version Number**

   Edit `pyproject.toml` and update the version:
   ```toml
   [project]
   version = "0.2.1"  # Update this line
   ```

2. **Commit and Merge Version Change**

   ```bash
   # On a feature branch or directly on main (if you have permission)
   git add pyproject.toml
   git commit -m "chore: bump version to 0.2.1"

   # If on a feature branch, push and create PR
   git push origin feature-branch
   # Then merge PR to main via GitHub

   # If working directly on main (ensure you have write access)
   git push origin main
   ```

3. **Create and Push Git Tag**

   ```bash
   # Switch to main and pull latest (ensures you're tagging the merged commit)
   git checkout main
   git pull origin main

   # Create annotated tag
   git tag -a v0.2.1 -m "Release v0.2.1"

   # Push tag to GitHub (this triggers the release workflow)
   git push origin refs/tags/v0.2.1

   # Alternative: push all tags
   # git push origin --tags
   ```

4. **Automatic Release Creation**

   The `.github/workflows/release.yml` workflow triggers on tag push and automatically:
   - Creates a GitHub release
   - Generates release notes from commits
   - Attaches release assets

5. **Verify Release**

   - Check [GitHub Releases](https://github.com/ontherivt/req-update-check/releases) for the new release
   - Verify the release notes are accurate

#### Manual Release Workflow Details

The release workflow (`.github/workflows/release.yml`) is configured as:

```yaml
on:
  push:
    tags:
      - 'v*'  # Triggers on any tag starting with 'v'

jobs:
  release:
    steps:
      - uses: softprops/action-gh-release@v1
        with:
          draft: false
          prerelease: false
          generate_release_notes: true  # Auto-generates from commits
```

## CI/CD Workflows

### Tests Workflow

Located at `.github/workflows/tests.yml`, this workflow:

- **Triggers**: On push to `main` or PRs targeting `main`
- **Python Versions**: Tests against 3.9, 3.10, 3.11, 3.12, 3.13
- **Steps**:
  1. Install dependencies
  2. Run `ruff check` and `ruff format --check`
  3. Run tests with coverage (`coverage run -m unittest discover`)
  4. Submit coverage to Coveralls

### Release Please Workflow

Located at `.github/workflows/release-please.yml`:

- **Triggers**: On push to `main` branch
- **Purpose**: Creates/updates release PRs based on conventional commits
- **Permissions**: Requires `contents: write` and `pull-requests: write`

### Release Workflow

Located at `.github/workflows/release.yml`:

- **Triggers**: On tag push matching `v*` pattern
- **Purpose**: Creates GitHub releases with auto-generated notes

## Development Workflow

### Setting Up

```bash
# Clone repository
git clone https://github.com/ontherivt/req-update-check.git
cd req-update-check

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
python -m unittest

# Run with coverage
coverage run -m unittest discover
coverage report
coverage xml

# Run specific test
python -m unittest tests.test_req_cheq.TestRequirements.test_get_packages
```

### Code Quality

```bash
# Check formatting and linting
ruff check .
ruff format --check .

# Auto-fix issues
ruff check --fix .
ruff format .
```

### Branch Strategy

- `main`: Stable branch, all releases cut from here
- Feature branches: Create from `main`, PR back to `main`
- Use conventional commits for automatic changelog generation

### Commit Message Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature (minor version bump)
- `fix`: Bug fix (patch version bump)
- `docs`: Documentation changes
- `chore`: Maintenance tasks
- `refactor`: Code refactoring
- `test`: Test changes
- `ci`: CI/CD changes

Breaking changes:
```
feat!: breaking API change

BREAKING CHANGE: explain the breaking change
```

## Publishing to PyPI

Publishing to PyPI is currently a manual process (not handled by CI/CD).

### Prerequisites

1. PyPI account with access to the `req-update-check` project
2. Install build tools:
   ```bash
   pip install build twine
   ```

### Publishing Steps

```bash
# 1. Ensure you're on the tagged release commit
git checkout v0.2.1

# 2. Clean previous builds
rm -rf dist/ build/ *.egg-info

# 3. Build distribution files
python -m build

# 4. Check distribution
twine check dist/*

# 5. Upload to PyPI (you'll be prompted for credentials)
twine upload dist/*

# Or upload to TestPyPI first to verify
twine upload --repository testpypi dist/*
```

### Verification

After publishing:

```bash
# Install from PyPI to verify
pip install --upgrade req-update-check

# Check version
req-update-check --version
```

## Troubleshooting

### Release Please Not Creating PR

- Verify commits follow conventional commit format
- Check workflow permissions in repository settings
- Review workflow logs in Actions tab

### Manual Release Tag Not Triggering Workflow

- Ensure tag follows `v*` pattern (e.g., `v0.2.1`, not `0.2.1`)
- Verify tag was pushed to GitHub: `git ls-remote --tags origin`
- Check workflow file syntax is valid

### Tag Push Error: "src refspec matches more than one"

If you get this error when pushing a tag:
```
error: src refspec v0.2.1 matches more than one
```

This means you have both a branch and a tag with the same name. Use the full ref path:
```bash
git push origin refs/tags/v0.2.1
```

Or delete the conflicting branch if it's no longer needed:
```bash
git branch -d v0.2.1  # Delete local branch
git push origin --delete v0.2.1  # Delete remote branch
```

### Test Failures in CI

- Ensure all tests pass locally first
- Check Python version compatibility (especially for pyproject.toml parsing)
- Review test logs in Actions tab for detailed error messages

## Additional Resources

- [Release Please Documentation](https://github.com/googleapis/release-please)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [PyPI Publishing Guide](https://packaging.python.org/tutorials/packaging-projects/)
