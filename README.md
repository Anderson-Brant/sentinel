# Sentinel

**Market intelligence & stock prediction platform.** Sentinel ingests market data and (soon) social sentiment, engineers predictive features, trains and evaluates models with proper time-series discipline, and produces research-grade outputs via a clean CLI.

> **Status:** Early-stage. The MVP loop (ingest → features → train → evaluate) runs end-to-end on equities via `yfinance`. Sentiment ingestion, backtesting, MLflow tracking, and Postgres/TimescaleDB support are scaffolded as roadmap items — see [Roadmap](#roadmap).

---

## Why

Price movement is the result of many interacting forces: trend, momentum, volatility regime, volume behavior, cross-asset correlation, and crowd attention. Most "stock predictor" projects use one data source, no time-aware evaluation, and a single model. Sentinel is designed to avoid those mistakes from the start:

- Multi-source by design (prices today; Reddit/X/news on the roadmap)
- Proper time-based splits + walk-forward validation (no leakage)
- Multiple model families compared against baselines + naive rules
- Honest evaluation: ablations (sentiment on/off), regime slicing, realistic metrics

---

## What's in the MVP today

| Layer | Status | Notes |
|---|---|---|
| Market data ingestion (yfinance) | ✅ Working | Equities, ETFs, indexes |
| Social ingestion (Reddit / X) | 🟡 Stubbed | Interfaces + config keys; adapters TODO |
| Storage (DuckDB + Parquet) | ✅ Working | Local analytics, zero-setup |
| Storage (Postgres / Timescale) | 🟡 Stubbed | Config toggle planned |
| Feature engineering — technical | ✅ Working | Returns, SMA/EMA, vol, momentum, volume |
| Feature engineering — sentiment | 🟡 Stubbed | VADER adapter pending |
| Target generation | ✅ Working | Directional + regression targets |
| Baseline models | ✅ Working | Logistic, Random Forest |
| Walk-forward evaluation | ✅ Working | Rolling-origin, no leakage |
| Backtesting engine | 🟡 Stubbed | Strategy → equity curve next |
| Console reporting (Rich) | ✅ Working | Tables, model comparison |
| MLflow experiment tracking | 🟡 Stubbed | Wiring behind a config flag |

---

## Quickstart

```bash
# 1. Install (editable)
pip install -e ".[dev]"

# 2. Run the end-to-end demo on SPY
sentinel demo SPY

# Or step through the pipeline:
sentinel ingest prices SPY --start 2015-01-01
sentinel features build SPY
sentinel train SPY --model logistic
sentinel evaluate SPY
sentinel predict SPY
```

Requires Python 3.11+.

---

## Architecture

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Ingestion   │──▶│   Storage    │──▶│   Features   │
│  (yfinance,  │   │  (DuckDB /   │   │  (technical, │
│   reddit*,   │   │   Parquet)   │   │   sentiment*)│
│   twitter*)  │   └──────────────┘   └──────┬───────┘
└──────────────┘                             │
                                             ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Reporting   │◀──│  Backtest*   │◀──│    Models    │
│  (Rich CLI)  │   │  (strategy → │   │  (sklearn,   │
└──────────────┘   │   equity)    │   │   xgboost*)  │
                   └──────────────┘   └──────────────┘

* = scaffolded, not yet implemented
```

Package layout (src layout):

```
src/sentinel/
├── cli.py              Typer CLI entrypoint
├── config.py           Pydantic settings
├── ingestion/          Market + social data adapters
├── storage/            DuckDB store + schemas
├── features/           Technical, sentiment, target generation
├── models/             Baselines + registry
├── evaluation/         Walk-forward / rolling-origin CV
├── backtest/           Strategy simulation (stub)
├── reporting/          Rich tables & summaries
└── utils/              Logging, paths
```

---

## Design principles

**No leakage.** All splits are time-based. Walk-forward validation is the default. Features only use information available at time *t*.

**Baselines first.** Every model is compared against a naive baseline (e.g., "predict majority class", "predict yesterday's sign"). If the fancy model can't beat the naive rule, that's a useful result, not a failure to hide.

**Ablations built in.** When sentiment lands, the pipeline will train three variants — technical-only, sentiment-only, hybrid — so we can measure whether sentiment *actually adds value* rather than just assuming it does.

**Reproducibility.** Runs are seeded; configs are declarative (YAML); MLflow integration (planned) records params, metrics, and artifacts per run.

**CLI-first.** No notebook-only workflows. If a workflow can't be expressed as a CLI command, it won't ship.

---

## Roadmap

**Near-term**
- [ ] Reddit (`praw`) ingestion adapter + ticker mention extraction
- [ ] Twitter/X ingestion adapter
- [ ] VADER sentiment features + rolling sentiment aggregates
- [ ] Backtest engine: signal → trades → equity curve → Sharpe/drawdown/win-rate
- [ ] MLflow tracking behind `--track` flag
- [ ] XGBoost / LightGBM models

**Medium-term**
- [ ] Postgres / TimescaleDB storage backend (config toggle)
- [ ] Ablation harness (tech-only vs sentiment-only vs hybrid)
- [ ] Regime detection + regime-sliced performance reporting
- [ ] Crypto OHLCV adapter

**Longer-term**
- [ ] Finance-tuned transformer sentiment (finBERT or similar)
- [ ] SHAP / permutation importance reporting
- [ ] Scheduled ingestion + intraday refresh
- [ ] Dockerized services + cloud deployment

---

## Risks & honest caveats

Markets are noisy; relationships decay; social hype is often *reactive* rather than predictive; backtests can look great and still be fake if evaluation is sloppy. Sentinel is a research and engineering project — not a trading system, not financial advice. Any real-money decision based on its output would be irresponsible.

---

## License

MIT — see [LICENSE](LICENSE).
