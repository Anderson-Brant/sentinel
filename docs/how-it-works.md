# Sentinel: How It Works

A plain-English tour of what Sentinel actually does under the hood: the pipeline, the database, every feature column, what the models do, and how to read every number the tool prints.

If you just want to run commands, see [`usage.md`](usage.md). This document is for people who want to understand *why* the commands produce what they produce.

## Part 1: The big picture

Sentinel answers one question: **"Is this stock more likely to go up or down tomorrow?"**

It does that in six stages. Each stage's output is the next stage's input.

```
┌─────────┐   ┌──────────┐   ┌──────┐   ┌──────────┐   ┌─────────┐   ┌─────────┐
│ INGEST  │──▶│ FEATURES │──▶│ TRAIN│──▶│ EVALUATE │──▶│ BACKTEST│──▶│ PREDICT │
└─────────┘   └──────────┘   └──────┘   └──────────┘   └─────────┘   └─────────┘
 Yahoo        Technical     Learn       Check how      Simulate      One real
 Finance      indicators    patterns    it would       actually      prediction
 to DB        from prices   in the      have done      trading on    for today
                            features    on unseen      those
                                        data           predictions
```

When you run `sentinel demo SYMBOL`, it walks through all six in sequence. When you run the sub-commands individually, it does one stage at a time using the data already stored from previous runs.

## Part 2: The database

Everything is persisted to a single DuckDB file at `data/sentinel.duckdb` (DuckDB is a local SQL database, like SQLite but designed for analytics). You can poke around in it with the DuckDB CLI or any SQL client.

### 2.1 The `prices` table

Raw OHLCV data from Yahoo Finance. One row per `(symbol, date)`.

| Column      | Type    | What it is                                                                 |
| ----------- | ------- | -------------------------------------------------------------------------- |
| `symbol`    | VARCHAR | Ticker like `SPY`, `AAPL`, `BTC-USD`, etc. Part of the primary key.        |
| `date`      | DATE    | Trading day. Part of the primary key.                                      |
| `open`      | DOUBLE  | Price at market open.                                                      |
| `high`      | DOUBLE  | Highest trade of the day.                                                  |
| `low`       | DOUBLE  | Lowest trade of the day.                                                   |
| `close`     | DOUBLE  | Last trade of the day. This is what most calculations use.                 |
| `adj_close` | DOUBLE  | Close adjusted for dividends and splits. Available but not used for now.   |
| `volume`    | DOUBLE  | Total shares traded that day.                                              |

### 2.2 The `features` table

The computed technical indicators that feed the model. One row per `(symbol, date)`. Built by `sentinel features build`.

There are **~30 feature columns** plus two target columns. See [Part 3](#part-3--the-features-explained) for what each one means.

### 2.3 The `reddit_posts` and `tweets` tables

Stored social-media posts mentioning tickers. Only populated if you run `sentinel ingest reddit` or `sentinel ingest twitter`. Each row carries sentiment scores from VADER (or, optionally, finBERT).

| Column               | What it is                                                     |
| -------------------- | -------------------------------------------------------------- |
| `post_id` / `tweet_id` | Unique ID from the source platform.                          |
| `created_ts`         | When the post/tweet was created (UTC).                         |
| `title` / `body` / `text` | Post content.                                             |
| `score`, `num_comments` | Reddit engagement metrics.                                  |
| `like_count`, `retweet_count`, etc. | Twitter engagement metrics.                     |
| `sentiment_compound` | Overall sentiment score, from -1 (very negative) to +1 (very positive). |
| `sentiment_pos/neg/neu` | Probability the post is positive/negative/neutral.          |

### 2.4 The `mentions` table

Cross-references which posts mentioned which tickers. `(post_id, symbol, source)`. A single Reddit post mentioning both `$AAPL` and `$MSFT` produces two rows here.

### 2.5 The `job_runs` table

Log of every scheduled job that's run. Used by `sentinel schedule status` and `sentinel schedule history`.

| Column         | What it is                                                       |
| -------------- | ---------------------------------------------------------------- |
| `job_name`     | Identifier from your `scheduler.jobs` config.                    |
| `started_at`   | When the job started.                                            |
| `finished_at`  | When it ended.                                                   |
| `status`       | `ok` or `failed`.                                                |
| `rows_written` | How much data the job produced.                                  |
| `error`        | Error message if `status = failed`, `NULL` otherwise.            |

## Part 3: The features explained

Each feature is a number computed from past prices that the model can use to predict the future. **Every feature only uses data from *before* the current date**, so there's no peeking at the future (that's called look-ahead bias and it's the #1 way backtests lie to you).

### 3.1 Returns (4 columns)

| Column         | Formula                                   | What it means                                      |
| -------------- | ----------------------------------------- | -------------------------------------------------- |
| `ret_1d`       | `(close_t / close_{t-1}) - 1`             | Percent change from yesterday's close to today's.  |
| `ret_5d`       | `(close_t / close_{t-5}) - 1`             | Percent change over the past week.                 |
| `ret_20d`      | `(close_t / close_{t-20}) - 1`            | Percent change over the past ~month.               |
| `log_ret_1d`   | `ln(close_t) - ln(close_{t-1})`           | Log return, more well-behaved for statistics.      |

**Why:** Short-term returns capture momentum/reversal. If a stock has been going up for 20 days, that's informative (in either direction, depending on whether the regime favors momentum or mean-reversion).

### 3.2 Moving averages and crossovers (4 columns)

| Column                | What it is                                                           |
| --------------------- | -------------------------------------------------------------------- |
| `close_to_sma_5`      | `close / SMA_5 - 1`. How far above/below the 5-day simple moving avg. |
| `close_to_sma_50`     | Same for 50-day SMA (classic medium-term trend signal).              |
| `close_to_sma_200`    | Same for 200-day SMA (classic long-term trend signal).               |
| `sma_50_over_200`     | 1 if the 50-day is above the 200-day, else 0. The **golden cross** indicator. |

**Note:** Sentinel drops the raw `sma_X` columns and keeps only the normalized `close_to_sma_X` ratios. Raw SMAs don't generalize: a $1000 stock and a $10 stock have wildly different SMA values, so the model would just memorize price levels. Ratios are regime-agnostic.

**Why:** "Price above its moving average" is the canonical trend-following signal. It's been used since the 1930s and it still works in backtests more often than it doesn't.

### 3.3 Momentum (6 columns)

| Column                | What it is                                                  |
| --------------------- | ----------------------------------------------------------- |
| `mom_5d`, `mom_20d`   | Rate of change over 5 and 20 days. Same as `ret_5d`/`ret_20d` mathematically. |
| `dist_to_high_5d`     | `close / max(close over last 5 days) - 1`. Negative unless at new 5-day high. |
| `dist_to_high_20d`    | Same for 20-day high.                                       |
| `dist_to_low_5d`      | `close / min(close over last 5 days) - 1`. Positive unless at new 5-day low. |
| `dist_to_low_20d`     | Same for 20-day low.                                        |

**Why:** Breakouts (close = new high) and breakdowns (close = new low) are distinct market states. A stock making new 20-day highs behaves differently than one in the middle of its recent range, even if recent *returns* are identical.

### 3.4 Volatility (4 columns)

| Column                   | What it is                                                              |
| ------------------------ | ----------------------------------------------------------------------- |
| `vol_5d`, `vol_20d`      | Standard deviation of daily log returns over the past 5 / 20 days.      |
| `hl_range_pct`           | Today's `(high - low) / close`. A cheap intraday-range proxy.          |
| `hl_range_pct_5d_mean`   | 5-day mean of `hl_range_pct`. Smoothed intraday volatility.            |

**Why:** Market regimes differ. Predictive signals work differently in calm vs. chaotic markets. Giving the model a read on current volatility lets it adapt its behavior.

### 3.5 Volume (5 columns)

| Column                | What it is                                                              |
| --------------------- | ----------------------------------------------------------------------- |
| `vol_avg_5d`, `vol_avg_20d` | Rolling mean of share volume.                                      |
| `rel_vol_5d`          | `today_volume / vol_avg_5d`. >1 means today is heavier than normal.     |
| `rel_vol_20d`         | Same against 20-day average.                                            |
| `signed_rel_vol_20d`  | `sign(ret_1d) × rel_vol_20d`. Big green candle on high volume = strong positive, big red candle on high volume = strong negative. |

**Why:** Volume confirms price moves. A 3% rally on 5× normal volume means something different from a 3% rally on a quiet day. The signed version gives the model directional conviction.

### 3.6 Sentiment features (optional, only if you ran Reddit/Twitter ingestion)

Built by `sentinel score-sentiment` + feature join. Structured so Reddit and Twitter live in parallel columns, so you can ablate either source independently.

| Column (pattern)                           | What it is                                          |
| ------------------------------------------ | --------------------------------------------------- |
| `reddit_mention_count`, `twitter_mention_count` | Number of posts/tweets mentioning this ticker today. |
| `reddit_sentiment_mean`, `twitter_sentiment_mean` | Mean VADER compound score that day.           |
| `reddit_sentiment_weighted`                | Mean score weighted by post engagement (upvotes + comments). |
| `twitter_sentiment_weighted`               | Same but by tweet likes/retweets/etc.               |
| `reddit_mention_zscore_7d`                 | Today's mention count standardized against its 7-day distribution. A "today is unusual" signal. |
| `*_rolling_mean_7d`                        | 7-day smoothed versions to reduce noise.           |

**Why:** Social sentiment can lead price moves (meme-stock era demonstrated this) but it's noisy. The z-score and smoothing variants are the model's chance to filter signal from chatter.

### 3.7 Targets (2 columns)

These are what the model is trying to **predict**, not features.

| Column             | What it is                                                                    |
| ------------------ | ----------------------------------------------------------------------------- |
| `target_direction` | 1 if `close_{t+horizon} > close_t`, else 0. The classification target.        |
| `target_return`    | `close_{t+horizon} / close_t − 1`. The regression target (not used by default). |

`horizon` defaults to 1 day, i.e. "will the close tomorrow be higher than today's close?"

**Important:** The last `horizon` rows of the feature table are dropped before training, because their targets are unknown (the future hasn't happened yet).

### 3.8 What gets dropped

After all features are computed, rows are dropped if they contain **any** NaN values. This typically removes the first ~200 days (the 200-day SMA needs 200 days of warmup) and the last 1 day (forward return unknown). You'll see this in the log output: `Feature table: 2639 rows (dropped 200 with NaN), 33 cols`.

## Part 4: The models explained

Sentinel supports four model types. All of them are classifiers that predict a probability the stock goes up.

### 4.1 `logistic`: Logistic Regression

**The baseline.** Finds a linear combination of features that best separates "up days" from "down days," then squashes the result through a sigmoid to produce a probability between 0 and 1.

**Pros:** Fast, interpretable, hard to overfit. Coefficients tell you exactly which features the model is using and how.

**Cons:** Can only learn linear relationships. "RSI above 70 AND volume spike" is a pattern logistic regression can't natively represent.

**When it wins:** When your features are already well-engineered and the underlying signal is mostly linear. Often close to tree models on noisy financial data, because noise drowns out nonlinear structure.

### 4.2 `random_forest`: Random Forest

Trains a bunch (100 by default) of **decision trees** on random subsets of the data and features, then averages their predictions. Each tree can learn nonlinear rules like "if `close_to_sma_200 > 0` and `rel_vol_20d > 1.5`, vote up."

**Pros:** Handles nonlinearity and feature interactions for free. No need to scale features. Reasonably robust.

**Cons:** Slower than logistic. Can overfit if trees are too deep.

### 4.3 `xgboost`: Extreme Gradient Boosting

A different way of combining trees: instead of averaging independent trees (random forest), **build each new tree specifically to correct the mistakes of the previous trees**. This is called "boosting."

**Pros:** Typically the single best off-the-shelf classifier in tabular ML competitions. Highly tunable. Built-in regularization fights overfitting.

**Cons:** More hyperparameters to get wrong. Slower to train than random forest.

**Requires:** `pip install "sentinel[ml-extra]"` (XGBoost is a heavyweight dependency).

### 4.4 `lightgbm`: LightGBM

Same idea as XGBoost (gradient boosting) but with a different tree-growing strategy (**leaf-wise** instead of level-wise) that's usually much faster on large datasets.

**Pros:** Often 5 to 10× faster than XGBoost with comparable accuracy.

**Cons:** Slightly more prone to overfitting on small datasets. Requires the same `[ml-extra]`.

### 4.5 How features get prepped

- **Logistic regression** gets a `StandardScaler` in front of it, so features are z-scored (mean 0, std 1) and the coefficients are comparable.
- **Tree models** get the features raw, because trees split on thresholds and don't care about absolute scales.

You can see this in the output of `build_classifier`: logistic has `["scaler", "clf"]` as its pipeline steps, trees just have `["clf"]`.

## Part 5: Training and evaluating

### 5.1 `sentinel train`: what happens

1. Reads the feature table for the symbol.
2. Takes the **last `test_fraction`** rows (default 20%) as the holdout set and trains on the earlier 80%.
3. Fits the pipeline (scaler, if any, then classifier) on the training rows.
4. Predicts on the holdout and computes three metrics:
   - **Holdout accuracy**: fraction of predictions that got the direction right.
   - **Holdout F1**: harmonic mean of precision and recall. Punishes imbalanced failure modes (e.g., "always predicts up").
   - **Holdout ROC-AUC**: area under the ROC curve. Ranges 0.5 (random) to 1.0 (perfect). Reads as "probability a randomly chosen up-day ranks higher than a randomly chosen down-day."
5. Also computes the **baseline accuracy**: what you'd get by always guessing the majority class. If the model can't beat the baseline, it's not learning anything real.
6. Saves the trained model to `models/<SYMBOL>__<model_name>.pkl` for later use by `predict` and `explain`.

**Reality check:** For daily direction prediction on liquid US equities, holdout accuracies of 51 to 55% are typical for genuinely predictive models. Anything claiming 60%+ on this task should be treated with suspicion: either look-ahead bias, data leakage, or the test period happened to be easy.

### 5.2 `sentinel evaluate`: walk-forward cross-validation

A single train/test split can lie to you. Maybe your holdout happened to be an unusually easy or hard period. Walk-forward evaluation repeats the train/test cycle many times, sliding forward through history.

```
Train on [2015..2018] -> test on [2018..2019]
Train on [2015..2019] -> test on [2019..2020]
Train on [2015..2020] -> test on [2020..2021]
...and so on.
```

You get one accuracy/F1/AUC per split, and the report shows the **mean and spread** across all splits. That's a much more honest estimate of how the model performs in practice, because it's effectively out-of-sample testing on the entire history.

### 5.3 Reading the evaluation output

- **Mean accuracy ± std**: average across splits and how much it varied. High variance across splits means the model is inconsistent.
- **Per-split table**: accuracy in each time slice. Watch for models that have one great split propping up the average.
- **Confusion matrix**: breakdown of true positives, false positives, true negatives, false negatives. Tells you whether the model is biased toward "always up" or "always down."

## Part 6: Backtesting, i.e. turning predictions into a strategy

Predictions are nice but they don't make money. The backtest answers: "if you had traded on these predictions every day, paying realistic costs, what would you have ended up with?"

### 6.1 The trading rules

- **Go long** when the model's predicted P(up) > `long_threshold` (default 0.55).
- **Go short** (if `--allow-short`) when P(up) < `short_threshold` (default 0.45).
- **Exit to cash** in between.
- **Positions shift by one bar**, so today's prediction drives tomorrow's position. You can't trade on a signal you only know after the close.
- **Transaction costs** are charged on every change in `|position|` at `cost_bps` per basis point (default 2bps per round-trip unit of position change). 2bps is roughly realistic for large-cap US equities via a modern broker.

### 6.2 Optional: volatility targeting

If you pass `--vol-target 0.10`, positions are sized so that `position × realized_vol ≈ 10%` annualized target vol. When the market is calm, you lever up toward `--max-leverage`. When it's wild, you size down. This makes the strategy's risk profile more stable across different regimes.

Without this flag, positions are fixed-size at {−1, 0, +1}.

### 6.3 Reading the backtest output

| Metric                  | What it means                                                                                      | Good/bad                                                                 |
| ----------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `total_return`          | Cumulative PnL over the whole period.                                                              | Higher than benchmark is good.                                           |
| `annualized_return`     | Total return re-expressed as a yearly rate.                                                        | Same.                                                                    |
| `annualized_vol`        | Std dev of daily returns × √252. How bouncy the equity curve is.                                   | Lower-for-same-return is good.                                           |
| `sharpe`                | `annualized_return / annualized_vol`. Risk-adjusted return.                                        | >1 is decent, >2 is rare-and-great, <0.5 is basically random.           |
| `max_drawdown`          | Worst peak-to-trough loss the strategy ever sat through.                                           | Smaller magnitude is better. Know your own pain tolerance.               |
| `win_rate`              | Fraction of trades that made money.                                                                | 50%+ is fine; profitable strategies can have <50% win rates if winners > losers. |
| `n_trades`              | Number of round-trip trades.                                                                       | More = more cost friction. Lower = maybe the strategy isn't picky enough. |
| `exposure`              | Fraction of time the strategy was in a position (vs. cash).                                        | Indicates how opportunistic the strategy is.                             |
| `turnover`              | Sum of `|position changes|` per period. How much trading.                                          | High turnover + low Sharpe = costs are eating you.                      |
| `benchmark_*`           | Same metrics for buy-and-hold the underlying.                                                      | Compare directly.                                                        |

### 6.4 Honesty checklist

Sentinel's backtest intentionally avoids the three classic ways backtests lie:

1. **No look-ahead bias**: positions are shifted by one bar, so today's decision uses only information available at yesterday's close.
2. **No training-on-test contamination**: probabilities come from walk-forward splits, not a model fit on the whole history.
3. **Realistic costs**: every position change is charged.

You can still fool yourself by (a) curve-fitting thresholds to a specific symbol, or (b) cherry-picking symbols that happened to work. The `ablate` and `regimes` commands exist partly to push back on those failure modes.

## Part 7: Feature importance (`sentinel explain`)

After you've trained a model, which features actually mattered? Two methods:

### 7.1 Permutation importance (default)

For each feature: shuffle its values across the dataset, ask the model to predict on the scrambled data, and measure how much accuracy drops. Big drop means the model was relying on that feature. No drop means the feature was useless.

**Pros:** Works with any model. Doesn't require installing extra libraries.

**Cons:** Slow on large datasets (needs `n_repeats` predictions per feature). Correlated features split their importance.

### 7.2 SHAP importance (`--method shap`)

Uses game-theoretic Shapley values to attribute each prediction to each feature. The summed magnitude across all predictions tells you overall feature impact.

**Pros:** More principled than permutation, especially with correlated features. Per-prediction explanations possible.

**Cons:** Requires `pip install "sentinel[explain]"`. Much slower on large datasets, so Sentinel subsamples to `max_samples` points by default.

### 7.3 Reading the output

Features ranked by mean absolute importance. Top features are what the model is really relying on. If the top-ranked feature is something suspicious (e.g., a feature that shouldn't theoretically predict anything), that's a hint you might have a bug or a data leak.

## Part 8: Regime analysis (`sentinel regimes`)

A strategy can be profitable on average but terrible in specific conditions (e.g., great in trending markets, awful in ranging markets). This command slices the backtest by regime.

### 8.1 Volatility regimes

Rolling 20-day volatility is split into terciles: **low / medium / high**. The backtest is partitioned into those three buckets and the metrics are recomputed per bucket.

"Sharpe 1.5 overall but 2.2 in low-vol and -0.3 in high-vol" is a very different story from "1.5 everywhere."

### 8.2 Trend regimes

**Bull** when `SMA_50 > SMA_200`, **bear** otherwise (the classic golden/death cross). Strategies often have asymmetric performance across this: momentum works better in bulls, mean-reversion in bears.

### 8.3 Why this matters

If you're about to deploy capital, knowing where your strategy works (and doesn't) is crucial. "This strategy is Sharpe 1.5 in the 70% of market conditions where the 50d is above the 200d" is a deployable insight. "It averages 1.5 overall" isn't.

## Part 9: Ablation (`sentinel ablate`)

Requires Reddit and/or Twitter data already ingested + scored.

Trains **three** models on the same walk-forward splits:

1. **Technicals only**: the 30-ish price-based features.
2. **Sentiment only**: just the Reddit/Twitter columns.
3. **Hybrid**: everything combined.

Compares their walk-forward metrics side-by-side. Answers the question: **does social sentiment add out-of-sample predictive value, or is it noise?**

**Reality check:** For most liquid large-caps, sentiment adds ~nothing. For meme stocks and smaller caps during periods of heavy social attention, it can help. The ablation exists specifically so you don't fool yourself into thinking sentiment works when it doesn't.

## Part 10: How to reason about whether a result is real

The single most important habit when running ML on financial data:

**Assume your first good result is wrong.**

Things to check before believing a backtest:

1. **Does it walk-forward?** If you trained on 2015 to 2023 and tested on 2015 to 2023, the numbers are meaningless. Sentinel's `evaluate` does walk-forward correctly, but if you wire something custom, check this first.
2. **Is there look-ahead?** Does any feature use information from *after* the prediction date? (E.g., a z-score computed over the full history including future data.) Sentinel's features are written to avoid this but if you add new ones, triple-check.
3. **Are costs realistic?** Backtests with zero transaction cost produce amazing Sharpe ratios that evaporate in production. 2bps per leg (what Sentinel defaults to) is realistic for liquid US equities with a modern broker.
4. **Does it work on symbols you didn't tune for?** If you tuned thresholds on SPY and the strategy only works on SPY, it's overfit.
5. **Does it survive regime slicing?** A strategy that's great in one volatility bucket and negative in another might just be long-volatility in disguise.
6. **Is the edge plausibly exploitable?** 55% directional accuracy on daily SPY with low turnover and 1.5 Sharpe is realistic. 65% with 3.0 Sharpe is not. Something is wrong.

The point of Sentinel isn't to find a magic strategy. It's to give you a scaffolding where, when you do find something, you have reason to believe it's real.
