# Methodology

This document describes the experimental design Sentinel is **intended
to execute** — the universe, time ranges, evaluation protocol, and
decision rules we use to call something a "real" finding vs. noise.

The platform is feature-complete against this design. A study writeup
— actual findings from running the protocol below — is a separate
artifact and is not included in this v0.1 release. Treating the platform
and the study as different deliverables keeps the engineering claims
honest: *this ships, this works, this is what it's for*.

---

## Problem framing

Sentinel is a **short-horizon directional predictor** for liquid
equities, ETFs, and major crypto pairs. The primary target is the sign
of the next-bar return at daily resolution (`horizon=1`, `shift=1`). A
regression target (next-bar log return) is supported for models that
prefer it, but the evaluation contract is binary classification + a
simulated long/short strategy in the backtest.

**What we are trying to answer, in order of importance:**

1. Do sentiment features (Reddit, X/Twitter, finBERT) add **durable
   out-of-sample edge** over a technical-only feature stack?
2. If they do, **where** — which regimes, which symbols, which
   horizons? Edge that only appears in one vol tercile is a different
   claim than edge that appears everywhere.
3. Can gradient-boosted models extract signal that the logistic
   baseline misses, after we pay for the added capacity with the usual
   overfitting risks?
4. Does vol-targeted sizing improve Sharpe net of transaction costs,
   without relying on lookahead for the vol estimate?

Everything in the platform — ablation harness, regime slicer, walk-forward
evaluator, SHAP/permutation importance, backtest with costs — exists to
answer *one of these four questions without lying*.

---

## Universe

**Equities & ETFs (primary).** A curated liquid set:

- Broad-market ETFs: `SPY`, `QQQ`, `IWM`, `DIA`
- Sector ETFs: `XLF`, `XLK`, `XLE`, `XLV`
- Mega-cap singles: `AAPL`, `MSFT`, `NVDA`, `TSLA`, `AMZN`, `GOOGL`, `META`
- Known meme/retail-attention names: `GME`, `AMC`, `PLTR`

The mega-cap and meme slice matters: sentiment signal is *expected* to
be stronger in retail-driven names and weaker in index ETFs dominated
by passive flows. A pipeline that can't tell the difference isn't
finding anything.

**Crypto (secondary).** `BTC-USD`, `ETH-USD`, `SOL-USD` via CCXT/Binance.
Crypto lives in the same `prices` table and runs the same pipeline, but
is reported separately because the base rates, costs, and regime
structure are different enough that pooling would mask effects.

**Deliberately out of scope for v0.1:** options, futures, FX,
international equities, microcaps. These bring enough structural
differences (listing effects, expiry, overnight gaps) that the
walk-forward + vol-sizing machinery would need extension.

---

## Time range & splits

**Training span:** 2015-01-01 through the most recent complete trading
day. For symbols with shorter histories (e.g., `PLTR`), use the full
available range and annotate the row count in the results.

**Evaluation protocol:** walk-forward / rolling-origin cross-validation
is the **only** accepted out-of-sample protocol in this project. A
fixed train/test split is not reported and not considered evidence of
anything.

Default fold geometry:

- `window = 252` trading days (~1 year training window)
- `step   =  56` trading days (~quarterly refit)
- `min_train = 504` days (~2 years) before the first test fold

Every metric reported is the mean ± std across folds. Single-fold
numbers exist in intermediate artifacts but are **never** what gets
quoted as a result.

---

## Features

Features are organized into **ablation-friendly blocks** so that we can
measure each source's contribution in isolation:

| Block | Features | Toggle |
|---|---|---|
| `technical` | returns (1/5/21d), SMA 20/50/200, EMA 12/26, realized vol (21d), RSI-14, volume z-score (21d), SMA-50/SMA-200 crossover | always on |
| `reddit`    | mention count, mention z-score (7/21d), VADER sentiment mean (1/3/7d), engagement-weighted sentiment | `--with-sentiment` |
| `twitter`   | mention count, mention z-score, VADER sentiment mean (1/3/7d), engagement-weighted sentiment | `--with-sentiment` |
| `finbert`   | optional drop-in replacement for VADER on both reddit + twitter columns | `--scorer finbert` |

All features use information available at time *t* only. Rolling
statistics are computed with `closed="left"`; the same-day bar never
contaminates its own prediction. This is enforced at the feature layer,
not left to the modeler.

---

## Models & baselines

**Models under evaluation:**

- `logistic` — regularized logistic regression (L2, C tuned per fold)
- `random_forest` — sklearn RF, shallow (`max_depth=6`), many trees
- `xgboost` — gradient-boosted trees, early-stopping on a walk-forward
  validation slice
- `lightgbm` — same role as xgboost, different bias profile

**Baselines we must beat:**

- `predict_majority` — always predict the training-set majority class
- `predict_prev_sign` — predict the sign of the prior-day return
- `buy_and_hold` — for the backtest comparison

A model that can't beat *both* baselines on at least one of
{accuracy, log-loss, Sharpe} across the walk-forward fold means is
reported as **no edge detected**, not as a modeling failure to hide.
This is the most important rule in the entire project.

---

## What counts as a finding

A result is reported as a **finding** only if it clears all four:

1. **Walk-forward mean beats both naive baselines** by at least 1 std
   across folds.
2. **Ablation gap is directional and material.** If `hybrid` ≥
   `technical-only` by ≥ 0.3 Sharpe and ≥ 1% accuracy, sentiment
   contributed. Smaller gaps get reported as "not detectable here."
3. **Importance is concentrated in the features we think matter.** A
   model that "works" but whose SHAP importance is dominated by a
   noise feature gets a skeptical footnote, not a victory lap.
4. **The backtest survives realistic costs** (`--cost-bps 2.0` on
   equities, `--cost-bps 10.0` on crypto) and vol targeting.

Anything that clears 1–3 but fails 4 is reported as "signal exists,
strategy doesn't." That's a useful result. Anything that fails 1 is
noise and is reported as such.

---

## What we explicitly will not claim

- That Sentinel predicts markets, in any general sense.
- That any single-symbol backtest result generalizes.
- That sentiment adds edge uniformly — if it helps, we expect it to
  help in specific regimes (attention-driven names, high-vol bears)
  and not in others, and we will report it that way.
- Anything based on a single train/test split.
- Any t-statistic from a backtest without a multiple-testing correction
  across the grid of symbols × models × feature sets.

---

## Failure modes we actively look for

- **Lookahead via feature construction.** Walked-through per block
  during development; the ablation harness triggers a re-check if a
  newly added feature produces anomalously high importance.
- **Lookahead via the vol target.** The realized-vol estimate used for
  sizing is computed strictly from past bars and shifted one bar
  forward. Any change to the sizing module requires a synthetic-data
  sanity check that vol-targeted Sharpe ≤ oracle-Sharpe + ε.
- **Surviving-company bias.** The universe is a fixed list of current
  tickers; we *will* have survivorship bias on equities. For v0.1 this
  is flagged in any writeup; v0.2 would add delisted-ticker support.
- **Sentiment sample collapse.** Reddit/Twitter ingestion can return
  very few posts for low-attention symbols on quiet days. Features
  computed on fewer than 3 posts/day are masked to NaN before
  aggregation and reported as coverage in the results.
- **Overfit via scheduler refit.** The scheduler has the power to
  refit daily. For the study we refit quarterly (step=56 days) and
  note deviations from this cadence.

---

## What a v0.1 study writeup would include

When this platform is used to produce actual findings — a separate,
future artifact — the minimum contents are:

- Universe, time range, and total bar count per symbol
- Walk-forward fold geometry (`window`, `step`, `min_train`, `folds`)
- For each symbol × model × feature set cell: mean ± std of
  {accuracy, log-loss, Sharpe, max DD, hit rate} across folds
- Regime-sliced performance table (as in the `regimes` CLI output)
- SHAP importance table for the best-performing cell per symbol
- Backtest equity curves for the best technical-only and hybrid cells
  per symbol, with costs and vol targeting applied
- An honest "did it work" verdict by the rules in *What counts as a
  finding* above — **per symbol, not pooled**

The shape of each of those outputs is already captured in
[`sample-outputs.md`](./sample-outputs.md). The platform produces them;
the study would be what we run through it.
