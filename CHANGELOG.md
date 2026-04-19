# Changelog

All notable changes to Sentinel will be documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/).

## [0.1.1] - 2026-04-19
### Fixed
- `sentinel demo` no longer crashes with `AttributeError: 'OptionInfo' object 
has no attribute 'decode'`. The demo wrapper now passes explicit values for every 
Typer-option-defaulted parameter when it calls sub-commands as plain Python 
functions, so unresolved `OptionInfo` sentinels never reach MLflow.


## [0.1.0] — 2026-04-18

Initial public release. Sentinel is feature-complete against its v0.1
roadmap: a working multi-source ingest → features → train → evaluate →
backtest loop with proper time-series discipline, pluggable storage, two
gradient-boosted model adapters, vol-targeted sizing, ablation/regime
analysis, MLflow tracking, SHAP importance, a declarative scheduler with a
durable run log, and a containerized deployment story.

### Added

#### Ingestion
- Equities & ETF OHLCV via `yfinance` (`sentinel ingest prices`).
- Crypto OHLCV via CCXT with paginated history, overlap dedupe, and
  yfinance-style symbol normalization (`BTC-USD` ↔ exchange `BTC/USDT`,
  with USDC/DAI/BUSD/TUSD also collapsing to `-USD` for storage). Same
  `prices` table as equities, so the rest of the pipeline is asset-class
  agnostic. (`sentinel ingest crypto`, `[crypto]` extra.)
- Reddit ingestion via `praw` with cashtag + whitelist mention extraction
  (`sentinel ingest reddit`, `[social]` extra).
- X/Twitter ingestion via `tweepy` v2 with cashtag query builder,
  idempotent writes, and engagement-weighted sentiment
  (`sentinel ingest twitter`, `[social]` extra).

#### Storage
- Pluggable `Store` protocol with two backends:
  - **DuckDB** (default) — single-file, zero-setup local analytics.
  - **Postgres / TimescaleDB** — opt-in via
    `SENTINEL_STORAGE_BACKEND=postgres`, `[postgres]` extra. Hypertables
    on `prices` / `reddit_posts` / `tweets` when the Timescale extension
    is available; soft-falls back to plain Postgres tables otherwise.
- Schema is created on first connect; feature columns are added
  dynamically with `ALTER TABLE ADD COLUMN` as new feature blocks come
  online — no migrations needed for the v0.1 surface.

#### Features
- Technical block: returns, SMA/EMA, realized vol, momentum, volume.
- Sentiment block: VADER scoring with per-date rollups, mention
  z-scores, and rolling means. Reddit and Twitter live in parallel
  `reddit_*` / `twitter_*` columns so either source can be ablated
  independently.
- finBERT scorer (`ProsusAI/finbert`) as a VADER-compatible drop-in via
  the `[transformers]` extra (lazy-imported `transformers` + `torch`).
- Target generation for directional and regression problems.

#### Models & evaluation
- Baselines: logistic regression, random forest.
- Gradient-boosted trees: XGBoost, LightGBM (`[ml-extra]` extra,
  lazy-imported).
- Walk-forward / rolling-origin cross-validation as the default eval
  protocol.
- Vol-targeted position sizing in the backtest:
  `size = target_vol / realized_vol`, capped by `--max-leverage`,
  1-bar shifted to avoid lookahead.
- Backtest output: equity curve, Sharpe, max drawdown, hit rate,
  vs.-buy-and-hold comparison.

#### Analysis surfaces
- Ablation harness (`sentinel ablate`): trains tech-only, sentiment-only,
  and hybrid variants on identical walk-forward splits.
- Regime detection + regime-sliced reporting (`sentinel regimes`):
  vol terciles + bull/bear SMA crossover.
- Feature importance (`sentinel explain`): permutation (dep-free) and
  SHAP (`[explain]` extra).
- Rich console reporting across all of the above.

#### Tracking
- MLflow integration behind `--track` on `train` and `backtest`. Logs
  parameters, metrics, and the trained model artifact per run.
  `[tracking]` extra.

#### Scheduling
- Declarative `scheduler.jobs` YAML config.
- Job kinds: `ingest-prices`, `ingest-reddit`, `ingest-twitter`,
  `ingest-crypto`, `score-sentiment`, `build-features`.
- `sentinel schedule run/status/history` CLI.
- Durable `job_runs` log on the active store. Failures stay "due" and
  retry on the next tick; one bad job never aborts the loop.

#### Deployment
- Multi-stage `Dockerfile` (slim Python 3.11, non-root user uid 10001,
  `tini` for signal handling, `HEALTHCHECK` via `sentinel version`).
- `docker-compose.yml` with the `sentinel` service in the default
  profile (DuckDB) and opt-in `postgres` / `mlflow` profile sidecars.
  Single-image, env-var-driven backend selection.
- CI extends to a `docker` job (hadolint + buildx + smoke-test) and a
  `compose` job that validates all three profiles.
- Fly.io deployment recipe in `deploy/fly.toml`.

### Documentation
- Sample CLI output for every major command in `docs/sample-outputs.md`.
- Experimental design + intended study protocol in `docs/methodology.md`.

[0.1.1]: https://github.com/Anderson-Brant/sentinel/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Anderson-Brant/sentinel/releases/tag/v0.1.0