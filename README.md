# Sentinel

**Long-term stock analysis from the command line.** One command grades a ticker on quality, valuation, price history, insider activity, and competitive position, each with a line of evidence. Underneath it sits a full ML research pipeline: multi-source ingest, walk-forward evaluation, costed backtests.

[![CI](https://github.com/Anderson-Brant/sentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/Anderson-Brant/sentinel/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

> **v0.3.0 shipped.** The `sentinel analyze` long-term scorecard is complete: quality, valuation, price history, insiders, and competitive position all grade, with related-ticker discovery. The v0.1 loop still runs end-to-end: equities (via `yfinance`) and crypto (via CCXT) with Reddit + X/Twitter sentiment as parallel optional blocks, pluggable storage (DuckDB default, Postgres / TimescaleDB opt-in), multi-stage Docker image + Fly.io deploy recipe.

## What it does

Sentinel has two lenses on the same data.

The first is the one you'd actually open on a weekend: `sentinel analyze` produces a dense scorecard for long-term investing decisions. Not a price prediction, not a buy signal. A structured read on whether a business is good, whether the price is high by its own standards, what the stock has done for holders over a decade, and what the people running it are doing with their own shares. You make the call; the tool's job is to make sure you're looking at the right numbers.

The second is a short-horizon prediction research pipeline, and it's built around the ways that kind of project usually lies to its author:

- **Multi-source by design.** Equities, crypto, Reddit, and X/Twitter feed the same feature table, and every sentiment block is cleanly separable for ablation.
- **No leakage.** Walk-forward / rolling-origin CV is the only accepted evaluation protocol. Features only use information available at time *t*.
- **Baselines first.** Every model is compared against `predict_majority`, `predict_prev_sign`, and buy-and-hold. If the fancy model can't beat the naive rule, that's a finding, not a failure to hide.
- **Honest evaluation.** Ablations (sentiment on/off), regime slicing (vol terciles × bull/bear), realistic transaction costs, vol-targeted sizing.

The scorecard's grading rules are in [`docs/analyze.md`](docs/analyze.md). The evaluation protocol and the rules for what counts as a finding vs. noise are in [`docs/methodology.md`](docs/methodology.md).

---

## See it in action

```
$ sentinel analyze NVDA

NVDA · NVIDIA Corporation · Semiconductors · $5.1T mcap · as of 2026-07-11

Quality        B+    ROIC 65%. Gross margin 71%, variable. Net cash. Revenue +100%/yr.
Valuation      A     P/E 32 (0th pct vs own history). PEG 0.2.
Price hist     A     10y CAGR 67%. Max DD 90% (recovered in 49mo).
Insiders       B-    Net selling 0.0% of shares over 6mo (neutral). 0 buys / 14 sells.
Competitive    A+    Revenue +100%/yr vs sector +10%. Op margin 60% vs sector 22%.

Composite: A-
Related: SPY, MSFT, AAPL, AMZN, TSLA
```

Every row drills down: `sentinel analyze NVDA --detail valuation` shows all the ratios and percentiles behind that A.

The prediction pipeline runs end-to-end with one command:

```bash
$ sentinel demo SPY
[00:00:01] ingest.prices      SPY  2015-01-02 → 2026-04-17   rows=2847  source=yfinance
[00:00:03] features.build     SPY  with_sentiment=False       rows=2820 cols=18
[00:00:05] evaluate           SPY  walk-forward folds=10 window=252 step=56
[00:00:06] backtest           SPY  cost_bps=2.0 sizing=unit
          summary: cagr=0.071 sharpe=0.88 max_dd=-0.144 hit_rate=0.523 vs_bh=+0.8% cagr
```

```
$ sentinel ablate SPY

                  Ablation: SPY, walk-forward folds=10
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┓
┃ Variant           ┃ Acc.   ┃ LogL   ┃ Sharpe ┃ Max DD   ┃ vs. B&H ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━┩
│ technical-only    │ 0.523  │ 0.688  │  0.94  │  -0.144  │  -0.023 │
│ sentiment-only    │ 0.508  │ 0.692  │  0.41  │  -0.231  │  -0.182 │
│ hybrid            │ 0.529  │ 0.687  │  1.02  │  -0.138  │  +0.009 │
└───────────────────┴────────┴────────┴────────┴──────────┴─────────┘
```

More captured output from every major command lives in [`docs/sample-outputs.md`](docs/sample-outputs.md).

## Quickstart

```bash
pip install -e ".[dev]"

sentinel analyze AAPL                                # long-term scorecard (docs/analyze.md)
sentinel analyze AAPL --detail quality               # drill into one row
sentinel demo SPY                                    # prediction pipeline, end to end
sentinel ingest prices SPY --start 2015-01-01
sentinel ingest crypto BTC-USD --start 2020-01-01    # CCXT, Binance by default
sentinel features build SPY --with-sentiment
sentinel train    SPY --model xgboost --track        # log params to MLflow
sentinel backtest SPY --vol-target 0.10 --max-leverage 2.0
sentinel ablate   SPY                                # tech vs. sentiment vs. hybrid
sentinel regimes  SPY                                # when does the strategy work?
sentinel explain  SPY --model xgboost --method shap
```

Requires Python 3.11+. Install the optional extras you need: `social`, `ml-extra` (XGBoost/LightGBM), `tracking` (MLflow), `explain` (SHAP), `postgres`, `crypto`.

## Capabilities

| Layer | What you get |
|---|---|
| **Analysis** | `sentinel analyze TICKER`: long-term scorecard, letter grade + one line of evidence per dimension. Quality (ROIC, margins, leverage, growth), valuation (ratios + percentile vs own history), price history (multi-horizon CAGR, drawdown/recovery), insider activity, competitive position, plus correlation-based related tickers ([docs/analyze.md](docs/analyze.md)) |
| **Ingestion** | Equities via `yfinance`; crypto via `ccxt` (any exchange, BTC-USD ↔ BTC/USDT symbol normalization); Reddit via `praw` with cashtag / whitelist extraction; X/Twitter via `tweepy` v2 with engagement-weighted sentiment |
| **Storage** | Pluggable `Store` protocol. DuckDB (default, zero-setup). Postgres / TimescaleDB opt-in; hypertables when the extension is live, plain tables otherwise |
| **Features** | Technical (returns, SMA/EMA, realized vol, momentum, volume z-scores); sentiment (VADER rollups per source; posts after the close roll to the next trading day so day-t features only see pre-close posts); prefixed blocks so ablation partitions cleanly |
| **Models** | Logistic + Random Forest baselines; XGBoost + LightGBM via lazy-imported `[ml-extra]` |
| **Evaluation** | Walk-forward / rolling-origin CV; directional + regression targets; ablation harness; regime-sliced performance (vol terciles × bull/bear SMA crossover) |
| **Backtest** | Signal to equity curve to Sharpe / Sortino / max DD / hit rate; transaction costs; vol-targeted sizing with leverage cap, 1-bar shifted |
| **Tracking** | MLflow behind `--track` on train + backtest; params, metrics, artifacts logged per run |
| **Explainability** | Permutation importance (dep-free) + SHAP via `[explain]`; Rich top-N table via `sentinel explain` |
| **Scheduling** | Declarative `scheduler.jobs` YAML; `ingest-{prices,reddit,twitter,crypto}` / `score-sentiment` / `build-features` kinds; durable `job_runs` log; failures retry, never abort the loop |
| **Deployment** | Multi-stage Docker image (slim Python, non-root, tini, healthcheck); docker-compose with DuckDB default + opt-in `postgres` / `mlflow` profiles; Fly.io recipe |

Release-by-release notes are in [`CHANGELOG.md`](CHANGELOG.md).

## Running scheduled jobs

Sentinel ships a declarative scheduler that turns the CLI commands into recurring jobs. Declare them under `scheduler.jobs` in your YAML config:

```yaml
scheduler:
  tick_seconds: 30
  jobs:
    - name: daily-prices
      kind: ingest-prices
      interval: 1d
      params:
        symbols: [SPY, AAPL, MSFT, NVDA, TSLA]
    - name: crypto-daily                       # CCXT, no API key needed for public data
      kind: ingest-crypto
      interval: 1d
      params:
        symbols: [BTC-USD, ETH-USD, SOL-USD]
        exchange: binance
    - name: wsb-hourly
      kind: ingest-reddit
      interval: 1h
      params:
        whitelist: [SPY, AAPL, MSFT, NVDA, TSLA]
        limit: 200
    - name: rebuild-features
      kind: build-features
      interval: 1d
      params:
        symbols: [SPY, AAPL, MSFT, NVDA, TSLA]
        with_sentiment: true
```

Then drive it from the CLI:

```bash
sentinel schedule run --once                   # run all due jobs one pass and exit
sentinel schedule run --forever                # daemon loop; Ctrl-C to stop
sentinel schedule status                       # per-job: last run, next due
sentinel schedule history --limit 20           # recent runs across all jobs
```

Every run (success, error, or skipped) is appended to a durable `job_runs` table. A failing job stays "due" and retries on the next tick; one bad job never aborts the loop.

## Switching to Postgres / TimescaleDB

The CLI talks to storage through a `Store` protocol, so the backend is a one-env-var switch:

```bash
pip install -e ".[postgres]"                   # or: pip install 'psycopg[binary]'

export SENTINEL_STORAGE_BACKEND=postgres
export SENTINEL_POSTGRES_DSN='postgresql://user:pass@host:5432/sentinel'

sentinel ingest prices SPY                     # identical CLI, Postgres backend
```

Schema is created on first connect. `prices`, `reddit_posts`, and `tweets` become Timescale hypertables when the extension is available, and soft-fall-back to plain Postgres tables otherwise. Feature columns are added dynamically (`ALTER TABLE ADD COLUMN`) as new feature blocks come online, no migrations.

## Twitter / X credentials

Set `TWITTER_BEARER_TOKEN` in your environment or `.env` before running `sentinel ingest twitter`. The adapter uses the v2 recent-search endpoint via `tweepy` (install the `social` extra). With a whitelist, Sentinel builds a cashtag query like `($SPY OR $AAPL OR $TSLA) -is:retweet lang:en`; pass `--query` to supply a raw v2 query instead.

## Crypto ingestion (CCXT)

Crypto OHLCV flows through the same `prices` table as equities. Symbols are stored in yfinance-style (`BTC-USD`, `ETH-USD`) regardless of which stablecoin the exchange actually quotes in, so `sentinel features build BTC-USD` and the rest of the pipeline work without special-casing crypto.

```bash
pip install -e '.[crypto]'                     # installs ccxt

sentinel ingest crypto BTC-USD                 # Binance, 1d, start from config
sentinel ingest crypto ETH-USD --start 2021-01-01 --exchange coinbase
sentinel ingest crypto SOL-USD --quote USDC    # trade via USDC instead of USDT
```

Public OHLCV endpoints on most CCXT exchanges (Binance, Coinbase, Kraken, ...) require no API key. The adapter paginates through `fetch_ohlcv` in batches of 1000 bars, deduplicates overlapping timestamps, and maps exchange-side `BTC/USDT` to storage-side `BTC-USD` automatically. USDC, DAI, BUSD, and TUSD quotes also normalize to `-USD`.

## Running in production (Docker)

Sentinel ships a multi-stage `Dockerfile` and a `docker-compose.yml` with opt-in sidecars, so the same image drives local experiments, a Postgres-backed deployment, and a full tracking setup.

```bash
docker compose up -d sentinel                  # default: scheduler daemon + DuckDB
docker compose --profile postgres up -d        # + TimescaleDB sidecar
docker compose --profile mlflow up -d          # + MLflow tracking server (localhost:5000)
docker compose run --rm sentinel demo SPY      # one-shot CLI run
```

The container runs as a non-root user (uid 10001), uses `tini` to reap zombies from the scheduler daemon, and exposes a Docker `HEALTHCHECK` that calls `sentinel version`. State persists across restarts via named volumes. Credentials pass through from your shell or `.env`.

For single-machine cloud deployment, see [`deploy/fly.toml`](deploy/fly.toml), a Fly.io recipe that runs the scheduler daemon on a persistent volume, with a drop-in path to Postgres when you outgrow DuckDB. [`deploy/README.md`](deploy/README.md) has notes for other platforms.

## Architecture

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Ingestion   │──▶│   Storage    │──▶│   Features   │
│  (yfinance,  │   │  (DuckDB or  │   │  (technical, │
│   ccxt,      │   │   Postgres/  │   │   sentiment) │
│   reddit,    │   │   Timescale) │   └──────┬───────┘
│   twitter)   │   └──────────────┘          │
└──────────────┘                             ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Reporting   │◀──│   Backtest   │◀──│    Models    │
│  (Rich CLI)  │   │  (strategy → │   │  (sklearn,   │
└──────────────┘   │   equity)    │   │   xgboost,   │
                   └──────────────┘   │   lightgbm)  │
                                      └──────────────┘
```

```
src/sentinel/
├── cli/                Typer CLI, one module per domain
├── config.py           Pydantic settings
├── ingestion/          yfinance + ccxt + reddit + twitter adapters
├── storage/            Pluggable Store (DuckDB default, Postgres/Timescale opt-in)
├── fundamental/        Scorecard rows: quality, valuation, price history, insiders, competitive
├── analyze/            Scorecard assembly + rendering
├── features/           Technical, sentiment, target generation
├── models/             Baselines + GBM adapters + registry
├── evaluation/         Walk-forward / rolling-origin CV
├── backtest/           Strategy simulation + vol-targeted sizing
├── scheduling/         Job specs + scheduler loop + registry
├── reporting/          Rich tables & summaries
└── utils/              Logging, paths
```

## Further reading

- [`CHANGELOG.md`](CHANGELOG.md): what shipped, release by release.
- [`docs/analyze.md`](docs/analyze.md): the scorecard's grading rules, data sources, and their limits. Read this before trusting a letter grade.
- [`docs/methodology.md`](docs/methodology.md): universe, walk-forward protocol, and decision rules for what counts as a finding. **Read this before interpreting any backtest number.**
- [`docs/sample-outputs.md`](docs/sample-outputs.md): captured Rich output for every major CLI command.
- [`deploy/`](deploy/): single-machine deployment recipes (Fly.io today).
- [`CONTRIBUTING.md`](CONTRIBUTING.md): how the codebase is organized and how to extend it (new data sources, new models, new feature blocks).

## Risks and honest caveats

Markets are noisy; relationships decay; social hype is often *reactive* rather than predictive; backtests can look great and still be fake if evaluation is sloppy. The scorecard's grades are only as good as free data allows: yfinance statements go back four or five years, sector baselines are coarse, and none of it substitutes for reading a filing. Sentinel is a research and engineering project, not a trading system, not financial advice. Any real-money decision based on its output alone would be irresponsible.

## License

MIT. See [LICENSE](LICENSE).
