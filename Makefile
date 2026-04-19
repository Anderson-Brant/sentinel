# Sentinel — developer shortcuts.
#
# All targets are .PHONY; nothing here depends on file timestamps.
# Run `make` or `make help` to see what's available.

.DEFAULT_GOAL := help
.PHONY: help install install-all lint format test test-cov verify clean \
        docker-build docker-up docker-down docker-logs \
        compose-validate release-check

PYTHON ?= python
PIP    ?= $(PYTHON) -m pip

# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------

help:  ## Show this help message.
	@awk 'BEGIN {FS = ":.*?## "; printf "\nUsage: make <target>\n\n"} \
	     /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2} \
	     /^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0, 5)}' $(MAKEFILE_LIST)

##@ Installation

install:  ## Editable install with dev extras only.
	$(PIP) install -e ".[dev]"

install-all:  ## Editable install with every optional extra (heavy).
	$(PIP) install -e ".[dev,social,ml-extra,tracking,postgres,crypto,explain,transformers]"

##@ Code quality

lint:  ## Run ruff on src + tests.
	ruff check src tests

format:  ## Autoformat with ruff.
	ruff format src tests
	ruff check --fix src tests

test:  ## Run the pytest suite.
	pytest -q

test-cov:  ## Run pytest with coverage.
	pytest --cov=sentinel --cov-report=term-missing

verify:  ## Run every sandbox verify_*.py script at the repo root.
	@set -e; for f in ../verify_*.py; do \
	    echo "--- $$f ---"; \
	    $(PYTHON) "$$f"; \
	done

##@ Docker / Compose

docker-build:  ## Build the sentinel image locally.
	docker build -t sentinel:latest .

docker-up:  ## Start the sentinel service in the background (DuckDB profile).
	docker compose up -d sentinel

docker-down:  ## Stop all compose services and remove orphans.
	docker compose down --remove-orphans

docker-logs:  ## Tail the sentinel scheduler logs.
	docker compose logs -f sentinel

compose-validate:  ## Validate all three compose profiles (no daemon needed).
	docker compose config --quiet
	docker compose --profile postgres config --quiet
	docker compose --profile mlflow  config --quiet

##@ Release

release-check:  ## Sanity-check a release commit: lint + test + verify artifacts.
	$(MAKE) lint
	$(MAKE) test
	pytest -q tests/test_docker.py tests/test_release_artifacts.py

##@ Cleanup

clean:  ## Remove caches + build artifacts (keeps data/ and mlruns/).
	rm -rf build dist .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info"  -prune -exec rm -rf {} +
