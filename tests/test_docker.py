"""Structural tests for Dockerfile, .dockerignore, and docker-compose.yml.

These assert invariants we care about (multi-stage build, non-root runtime,
HEALTHCHECK present, profile-gated sidecars, shared volume mounts, etc.)
without requiring a docker daemon — the CI ``docker`` job exercises the
actual build.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")


REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = REPO_ROOT / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    assert DOCKERFILE.is_file(), f"Dockerfile missing at {DOCKERFILE}"
    return DOCKERFILE.read_text()


@pytest.fixture(scope="module")
def dockerignore_text() -> str:
    assert DOCKERIGNORE.is_file(), f".dockerignore missing at {DOCKERIGNORE}"
    return DOCKERIGNORE.read_text()


@pytest.fixture(scope="module")
def compose() -> dict:
    assert COMPOSE_FILE.is_file(), f"docker-compose.yml missing at {COMPOSE_FILE}"
    return yaml.safe_load(COMPOSE_FILE.read_text())


# ---------------------------------------------------------------------------
# Dockerfile invariants
# ---------------------------------------------------------------------------


def test_dockerfile_is_multi_stage(dockerfile_text):
    stages = re.findall(r"^FROM\s+.+\s+AS\s+(\w+)", dockerfile_text, flags=re.MULTILINE)
    assert "builder" in stages
    assert "runtime" in stages


def test_dockerfile_uses_slim_python_base(dockerfile_text):
    # Slim base keeps the runtime image small; we rely on it in README claims.
    assert "python:${PYTHON_VERSION}-slim-bookworm" in dockerfile_text
    assert re.search(r"ARG\s+PYTHON_VERSION=3\.11", dockerfile_text)


def test_dockerfile_copies_venv_from_builder(dockerfile_text):
    # Runtime stage must copy the prebuilt venv so we don't ship build tools.
    assert re.search(
        r"COPY\s+--from=builder(?:\s+--chown=[^ ]+)?\s+/opt/venv\s+/opt/venv",
        dockerfile_text,
    )


def test_dockerfile_runs_as_non_root(dockerfile_text):
    assert re.search(r"useradd\s+.*--uid\s+10001\b", dockerfile_text)
    # Any USER directive pointing at the sentinel account.
    assert re.search(r"^USER\s+sentinel\b", dockerfile_text, flags=re.MULTILINE)


def test_dockerfile_has_healthcheck(dockerfile_text):
    assert "HEALTHCHECK" in dockerfile_text
    assert "sentinel version" in dockerfile_text


def test_dockerfile_declares_data_volume(dockerfile_text):
    assert re.search(r'^VOLUME\s+\["\s*/data\s*"\]', dockerfile_text, flags=re.MULTILINE)


def test_dockerfile_entrypoint_uses_tini_and_sentinel(dockerfile_text):
    # tini reaps zombies for the long-running scheduler daemon.
    assert '"/usr/bin/tini"' in dockerfile_text
    assert re.search(r'ENTRYPOINT\s+\[[^\]]*"sentinel"', dockerfile_text)


def test_dockerfile_default_cmd_runs_scheduler(dockerfile_text):
    assert re.search(
        r'CMD\s+\[\s*"schedule"\s*,\s*"run"\s*,\s*"--forever"\s*\]',
        dockerfile_text,
    )


def test_dockerfile_installs_production_extras(dockerfile_text):
    # These extras must stay in the runtime install list so the image supports
    # Postgres/Timescale, crypto ingestion, gradient boosting, MLflow, and
    # SHAP-based explanations. Losing one silently breaks the advertised
    # `docker compose` flows.
    install_line = re.search(r'pip install\s+"\.\[([^\]]+)\]"', dockerfile_text)
    assert install_line, "couldn't find the `pip install '.[...]'` line"
    extras = {e.strip() for e in install_line.group(1).split(",")}
    for required in ("social", "ml-extra", "tracking", "postgres", "crypto", "explain"):
        assert required in extras, f"Dockerfile runtime image missing [{required}]"


def test_dockerfile_sets_sentinel_env(dockerfile_text):
    assert "SENTINEL_DB_PATH=/data/sentinel.duckdb" in dockerfile_text
    assert "PATH=\"/opt/venv/bin:${PATH}\"" in dockerfile_text


# ---------------------------------------------------------------------------
# .dockerignore invariants
# ---------------------------------------------------------------------------


def test_dockerignore_excludes_secrets_and_state(dockerignore_text):
    lines = {line.strip() for line in dockerignore_text.splitlines() if line.strip()}
    for needle in (".git", ".env", "*.duckdb", "mlruns", "tests", "__pycache__"):
        assert needle in lines, f".dockerignore should exclude {needle!r}"


def test_dockerignore_excludes_docker_meta_to_avoid_recursion(dockerignore_text):
    # Including Dockerfile/compose files in the build context is wasteful and
    # invalidates the cache on every edit to docs-adjacent Dockerfile lines.
    assert "Dockerfile*" in dockerignore_text
    assert "docker-compose*.yml" in dockerignore_text


# ---------------------------------------------------------------------------
# docker-compose.yml invariants
# ---------------------------------------------------------------------------


def test_compose_has_expected_services(compose):
    services = compose.get("services", {})
    assert set(services) >= {"sentinel", "postgres", "mlflow"}


def test_compose_default_service_builds_from_dockerfile(compose):
    sentinel = compose["services"]["sentinel"]
    build = sentinel.get("build")
    assert build is not None, "sentinel service must build the local Dockerfile"
    if isinstance(build, dict):
        assert build.get("context") == "."
        assert build.get("dockerfile") == "Dockerfile"
    assert sentinel.get("image") == "sentinel:latest"


def test_compose_sentinel_service_has_no_profiles(compose):
    # The sentinel service must run on plain `docker compose up` — no profile.
    sentinel = compose["services"]["sentinel"]
    assert not sentinel.get("profiles"), (
        "sentinel must be active in the default profile — move sidecars to named profiles instead"
    )


def test_compose_postgres_is_profile_gated(compose):
    postgres = compose["services"]["postgres"]
    assert postgres.get("profiles") == ["postgres"]
    # Timescale image so the hypertable path works; the schema soft-falls-back
    # if the extension is absent, so plain Postgres would still be valid.
    assert "timescaledb" in postgres.get("image", "").lower()


def test_compose_mlflow_is_profile_gated(compose):
    mlflow = compose["services"]["mlflow"]
    assert mlflow.get("profiles") == ["mlflow"]
    assert "mlflow" in mlflow.get("image", "").lower()


def test_compose_sentinel_depends_on_postgres_optionally(compose):
    sentinel = compose["services"]["sentinel"]
    depends = sentinel.get("depends_on", {})
    # `required: false` is what keeps the default (DuckDB) profile working
    # even though postgres isn't spun up.
    assert depends.get("postgres", {}).get("condition") == "service_healthy"
    assert depends["postgres"].get("required") is False


def test_compose_sentinel_has_healthcheck(compose):
    hc = compose["services"]["sentinel"].get("healthcheck", {})
    test_cmd = hc.get("test") or []
    assert any("sentinel" in str(arg) for arg in test_cmd)


def test_compose_env_vars_have_defaults(compose):
    env = compose["services"]["sentinel"].get("environment", {})
    backend = env.get("SENTINEL_STORAGE_BACKEND", "")
    assert "duckdb" in str(backend), (
        "SENTINEL_STORAGE_BACKEND must default to duckdb so the zero-setup flow works"
    )
    # DB path lives inside the persistent /data volume.
    assert env.get("SENTINEL_DB_PATH") == "/data/sentinel.duckdb"


def test_compose_mounts_persistent_volumes(compose):
    sentinel = compose["services"]["sentinel"]
    mounts = sentinel.get("volumes", [])
    joined = "\n".join(mounts)
    assert "sentinel-data:/data" in joined, "DuckDB + CLI outputs must persist"
    assert "sentinel-mlruns:/app/mlruns" in joined, (
        "MLflow local runs must persist across container restarts"
    )

    declared = set((compose.get("volumes") or {}).keys())
    for vol in ("sentinel-data", "sentinel-mlruns", "sentinel-pg", "mlflow-data"):
        assert vol in declared, f"volume {vol!r} referenced but not declared at top level"


def test_compose_sentinel_passes_through_credentials(compose):
    env = compose["services"]["sentinel"].get("environment", {})
    for key in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "TWITTER_BEARER_TOKEN"):
        assert key in env, f"{key} must be passed through to the container"


# ---------------------------------------------------------------------------
# Build-context sanity
# ---------------------------------------------------------------------------


def test_dockerfile_copied_paths_exist(dockerfile_text):
    # Every local COPY source (post --chown / --from filter) must exist in the
    # build context; otherwise `docker build` fails with a misleading error.
    copy_pattern = re.compile(r"^COPY\s+(?!--from=)(.*)$", flags=re.MULTILINE)
    for match in copy_pattern.finditer(dockerfile_text):
        args = match.group(1).split()
        # Drop any --chown/--chmod flags.
        args = [a for a in args if not a.startswith("--")]
        if len(args) < 2:
            continue
        sources, _dest = args[:-1], args[-1]
        for src in sources:
            # Strip quoting and trailing slashes.
            src = src.strip('"').rstrip("/")
            if src in {".", "./"}:
                continue
            candidate = REPO_ROOT / src
            assert candidate.exists(), (
                f"Dockerfile COPY references missing path: {src}"
            )
