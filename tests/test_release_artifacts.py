"""Meta-tests: the release polish doesn't silently drift.

These assertions don't exercise any code — they guard the written
artifacts (CHANGELOG, docs/, deploy/, CONTRIBUTING) against rot. If
someone renames a file, bumps the version without updating the
CHANGELOG, or removes a doc the README cross-links, CI fails loudly
instead of the repo quietly becoming inconsistent.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
CONTRIBUTING = REPO_ROOT / "CONTRIBUTING.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"
DOCS_METHODOLOGY = REPO_ROOT / "docs" / "methodology.md"
DOCS_SAMPLE_OUTPUTS = REPO_ROOT / "docs" / "sample-outputs.md"
DEPLOY_FLY = REPO_ROOT / "deploy" / "fly.toml"
DEPLOY_README = REPO_ROOT / "deploy" / "README.md"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text()


@pytest.fixture(scope="module")
def changelog() -> str:
    return CHANGELOG.read_text()


@pytest.fixture(scope="module")
def pyproject() -> str:
    return PYPROJECT.read_text()


# ---------------------------------------------------------------------------
# Presence — every artifact the README promises actually exists
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        CHANGELOG,
        CONTRIBUTING,
        DOCS_METHODOLOGY,
        DOCS_SAMPLE_OUTPUTS,
        DEPLOY_FLY,
        DEPLOY_README,
    ],
)
def test_artifact_exists(path):
    assert path.is_file(), f"missing release artifact: {path.relative_to(REPO_ROOT)}"


# ---------------------------------------------------------------------------
# Version / CHANGELOG coherence
# ---------------------------------------------------------------------------


def test_pyproject_version_has_changelog_entry(pyproject, changelog):
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.MULTILINE)
    assert match, "couldn't find version in pyproject.toml"
    version = match.group(1)
    # A level-2 header of the form `## [<version>]` must exist.
    assert re.search(
        rf"^##\s+\[{re.escape(version)}\]", changelog, flags=re.MULTILINE
    ), f"CHANGELOG is missing an entry for version {version}"


def test_changelog_is_keepachangelog_style(changelog):
    # Title + at least one version section + link reference at the bottom.
    assert changelog.startswith("# Changelog")
    assert re.search(r"^##\s+\[\d+\.\d+\.\d+\]", changelog, flags=re.MULTILINE)


# ---------------------------------------------------------------------------
# Cross-references — every relative link in README/docs/deploy resolves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source_file",
    [README, DOCS_METHODOLOGY, DOCS_SAMPLE_OUTPUTS, DEPLOY_README, CHANGELOG, CONTRIBUTING],
)
def test_relative_links_resolve(source_file):
    text = source_file.read_text()
    # `[label](target)` — skip absolute URLs, anchors, and mailto.
    for match in re.finditer(r"\]\(([^)]+)\)", text):
        target = match.group(1).split("#", 1)[0].split(" ", 1)[0]
        if not target:
            continue
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        resolved = (source_file.parent / target).resolve()
        assert resolved.exists(), (
            f"{source_file.relative_to(REPO_ROOT)} links to missing path: {target}"
        )


# ---------------------------------------------------------------------------
# Extras stay coherent across pyproject, Dockerfile, README
# ---------------------------------------------------------------------------


_EXPECTED_EXTRAS = {
    "social",
    "ml-extra",
    "transformers",
    "tracking",
    "explain",
    "postgres",
    "crypto",
    "dev",
}


def test_pyproject_declares_expected_extras(pyproject):
    declared = set(re.findall(r"^([a-z][a-z0-9-]*)\s*=\s*\[", pyproject, flags=re.MULTILINE))
    missing = _EXPECTED_EXTRAS - declared
    assert not missing, f"pyproject missing extras: {missing}"


def test_dockerfile_installs_production_extras():
    text = (REPO_ROOT / "Dockerfile").read_text()
    match = re.search(r'pip install\s+"\.\[([^\]]+)\]"', text)
    assert match, "Dockerfile is missing its `pip install '.[...]'` line"
    installed = {e.strip() for e in match.group(1).split(",")}
    for required in ("social", "ml-extra", "tracking", "postgres", "crypto", "explain"):
        assert required in installed, f"Dockerfile drops extra: {required}"


# ---------------------------------------------------------------------------
# Roadmap sanity — the README must not advertise unshipped work
# ---------------------------------------------------------------------------


def test_readme_has_no_unchecked_roadmap_items(readme):
    # v0.1 ships everything, so any `- [ ]` checkbox left in the README is a
    # drift signal — either work slipped, or the README wasn't updated. The
    # roadmap itself lives in CHANGELOG.md now; this guards against someone
    # reintroducing a Roadmap section with unchecked items.
    unchecked = re.findall(r"^- \[ \] .+$", readme, flags=re.MULTILINE)
    assert not unchecked, f"unchecked roadmap items in README: {unchecked}"


def test_readme_advertises_shipped_version(readme, pyproject):
    # The README should at least mention the current version somewhere.
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.MULTILINE)
    assert match
    version = match.group(1)
    major_minor = ".".join(version.split(".")[:2])
    assert f"v{major_minor}" in readme or version in readme, (
        f"README doesn't mention the current release (v{version})"
    )
