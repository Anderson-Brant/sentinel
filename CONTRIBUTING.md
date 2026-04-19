# Contributing to Sentinel

Thanks for your interest. This guide covers how the codebase is
organized, how to run the test matrix, and how to extend the platform
cleanly along the three axes it's designed for: **new data sources**,
**new models**, and **new feature blocks**.

## Getting set up

```bash
git clone https://github.com/Anderson-Brant/sentinel.git
cd sentinel
pip install -e ".[dev,ml-extra,tracking,explain]"

# Sanity check — all three should pass:
ruff check src tests
pytest -q
sentinel demo SPY
```

`make help` lists the convenience targets (`make test`, `make lint`,
`make docker-build`, `make verify`).

## Repository layout

Everything worth touching lives under `src/sentinel/`. Tests mirror the
module structure in `tests/`, and each non-trivial module has a
companion `verify_*.py` at the repo root — a standalone script that
exercises the module in environments where `pytest` or the full dep
set isn't available (the CI runs pytest; the sandbox runs verify).
See `docs/methodology.md` for the evaluation protocol the platform is
built around.

```
src/sentinel/
├── cli.py              Typer CLI entrypoint — the only user surface
├── config.py           Pydantic settings (merges YAML + env + defaults)
├── ingestion/          Data adapters: yfinance, ccxt, reddit, twitter
├── storage/            Store protocol + DuckDB + Postgres backends
├── features/           Technical + sentiment feature blocks
├── models/             Registry + baselines + GBM adapters
├── evaluation/         Walk-forward / rolling-origin CV
├── backtest/           Strategy simulation + vol-targeted sizing
├── scheduling/         Job specs + scheduler loop + registry
├── reporting/          Rich tables
└── utils/              Logging, paths
```

## Testing

Three overlapping layers — don't skip any of them when contributing:

- **pytest** (`pytest -q`). Primary test suite. Every new module should
  land with tests here. Aim for the same style as the existing suites:
  small fixtures, no network, injected fakes rather than patched
  globals.
- **verify\_\<module\>.py** at the repo root. A standalone assertion
  script that works in a dep-poor sandbox by stubbing heavy optional
  dependencies (pydantic, duckdb, ccxt, transformers, etc.). Use
  `verify_crypto.py` or `verify_docker.py` as templates. These exist so
  that a reviewer without Docker or MLflow can still confirm the
  module's invariants end-to-end.
- **CI** (`.github/workflows/ci.yml`). pytest + ruff on Python 3.11 and
  3.12, plus a docker build + hadolint + smoke-test job, plus a
  compose-profile validation job.

If you change a module, re-run its `verify_*.py` locally before pushing
— CI does not run them directly, so the repo-root sandbox is the
convention.

## Extending along the three axes

### Adding a new ingestion source

1. New file under `src/sentinel/ingestion/<source>.py`. Define a
   `Protocol` for the external client at the top so tests can inject a
   fake without mocking the library. Look at `ingestion/crypto.py`
   (`ExchangeClient`) for the template.
2. Return a DataFrame shape-compatible with the existing `prices`
   table if the source is bar data, or add a dedicated table via the
   storage backends if not (see how `reddit_posts` and `tweets` were
   added).
3. Wire a `@ingest_app.command("<source>")` into `cli.py`, following
   the signature of `ingest crypto` or `ingest twitter`.
4. Register a scheduler kind in `scheduling/registry.py` with per-item
   failure isolation (don't let one bad symbol abort the batch).
5. Add to `.env.example` if the source needs credentials, and extend
   `config/default.yaml` with an `ingestion.<source>` block.
6. Tests in `tests/test_<source>.py`; sandbox verify at
   `verify_<source>.py`.

### Adding a new model

1. Create `src/sentinel/models/<model>.py` with a
   `class <Model>Adapter` implementing the same interface as
   `LogisticAdapter` (`fit`, `predict`, `predict_proba`, `name`,
   `default_params`).
2. Register it in `models/registry.py` so `sentinel train --model
   <name>` resolves it.
3. Make any heavy deps lazy-imported with a helpful error message if
   missing — follow `models/xgboost_adapter.py` for the pattern.
4. Add the dep to `pyproject.toml` as an optional extra
   (`[project.optional-dependencies]`). Never add a heavy dep to the
   base `dependencies` list.
5. Add it to the Dockerfile's runtime install line if it's part of the
   expected deploy set.
6. Tests: walk-forward parity with an existing model on synthetic
   data. Sandbox verify exercises `fit` + `predict` without the
   optional dep when feasible (stubbed).

### Adding a new feature block

1. New file under `src/sentinel/features/<block>.py`. Export a
   `build_<block>_features(prices, config)` function returning a
   DataFrame whose columns are all prefixed consistently
   (`tech_*`, `reddit_*`, `twitter_*`). Prefixes are load-bearing — the
   ablation harness partitions features by prefix.
2. Wire it into the feature-build pipeline (`features/build.py`) behind
   a toggle on the config.
3. Crucially: enforce `closed="left"` on all rolling stats. The same
   bar cannot contaminate its own prediction. If you skip this the
   backtest will look great and be useless.
4. Tests must include a lookahead assertion: assert that shifting the
   output one bar forward and recomputing changes nothing.

## Style

- `ruff check src tests` is canon. Configure your editor to run ruff on
  save and you'll never think about style.
- Type hints on public functions; `from __future__ import annotations`
  at the top of every file.
- Prose comments should explain *why*, not *what*. The code explains
  what. The comment should be the reason the code looks odd.
- No vendored models, no committed datasets, no committed credentials.
  `.dockerignore` and `.gitignore` enforce this — please don't go
  around them.

## Design rules that are non-negotiable

These come from `docs/methodology.md`, but they're worth repeating in
the contributor context:

1. **No lookahead, ever.** Features and vol estimates use past bars
   only, shifted forward by one before they reach the model.
2. **Walk-forward only.** Any reported metric is a mean ± std across
   folds. Single-fold numbers are not results.
3. **Beat the naive baseline** (`predict_majority`, `predict_prev_sign`,
   `buy_and_hold`) by more than one standard deviation across folds
   before claiming anything. A model that doesn't beat both naive rules
   is reported as "no edge detected," not hidden.

## Reporting issues & proposing changes

- Bug reports: include the invocation, the observed behavior, the
  expected behavior, and your Python + platform versions. A small
  repro in `verify_<something>.py` style is ideal.
- Feature proposals: open an issue before a PR for anything larger
  than a bug fix. The three-axis framing above is the frame — if your
  idea doesn't fit it, that's worth discussing first.
- PRs: keep them to one logical change; include tests, verify script
  updates, and README/CHANGELOG edits; run `ruff check` + `pytest` +
  the relevant `verify_*.py` before pushing.
