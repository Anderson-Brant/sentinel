# Sentinel — Usage Guide

A step-by-step walkthrough of how to actually use Sentinel. No prior knowledge assumed. Copy-paste every command exactly as shown.

Everything below happens in the **Terminal** app on macOS (press `Cmd+Space`, type "Terminal", hit Enter).

---

## Part 1 — First-time setup (do this once)

You only need to do this section **the very first time** you set up Sentinel on your computer. Skip to Part 2 if you've already done it.

### 1.1 Open a terminal and go to the project folder

```
cd /Users/brant/Documents/Claude/Projects/Sentinel
```

After running this, your terminal prompt should end with `Sentinel %`.

### 1.2 Create a virtual environment

A "virtual environment" is a sandboxed Python just for this project. It keeps Sentinel's dependencies from interfering with anything else on your computer.

```
python3 -m venv .venv
```

This creates a hidden folder called `.venv` inside your project. It takes a few seconds. Nothing prints on success.

### 1.3 Turn on the virtual environment

```
source .venv/bin/activate
```

You'll know it worked because your prompt now starts with `(.venv)`. For example:

```
(.venv) brant@Brants-MacBook-Pro Sentinel %
```

This `(.venv)` prefix means "I am using the project's sandboxed Python." As long as you see it, you're in the right place.

### 1.4 Install Sentinel

```
python3 -m pip install -e .
```

This downloads all the libraries Sentinel needs (pandas, scikit-learn, yfinance, etc.) and then installs Sentinel itself. It takes 1–3 minutes the first time.

The `-e` means "editable" — if you change the source code, the changes take effect immediately without reinstalling.

When it finishes you'll see `Successfully installed sentinel-0.1.1 ...`. Done with setup.

---

## Part 2 — Every time you want to use Sentinel

Whenever you close the terminal and come back later, run these two commands first:

```
cd /Users/brant/Documents/Claude/Projects/Sentinel
source .venv/bin/activate
```

You should see `(.venv)` at the start of your prompt. If you don't, Sentinel won't work.

When you're done using Sentinel, you can either close the terminal or type:

```
deactivate
```

---

## Part 3 — The one command that does everything

This is the fastest way to see what Sentinel can do. It ingests data, builds features, trains a model, evaluates it, backtests it as a trading strategy, and prints a prediction — all in one shot.

```
sentinel demo SPY
```

`SPY` is the ticker for the S&P 500 ETF. You can swap it for any stock:

```
sentinel demo AAPL        # Apple
sentinel demo TSLA        # Tesla
sentinel demo MSFT        # Microsoft
sentinel demo NVDA        # Nvidia
sentinel demo BTC-USD     # Bitcoin
```

This takes 30 seconds to 2 minutes depending on the stock. When it finishes you'll see a prediction table at the bottom — that's the model's guess for whether the stock goes up or down tomorrow.

---

## Part 4 — Running the steps individually

`sentinel demo` runs six steps back-to-back. Here's how to run each one by itself, which is useful when you want more control (e.g., a fancier model, a different backtest threshold, comparing strategies).

Assume throughout that `SYMBOL` is a stock ticker like `AAPL`. Replace it with whatever you actually want.

### Step 1 — Download the price history

```
sentinel ingest prices AAPL
```

Downloads ~10 years of daily OHLCV (open/high/low/close/volume) from Yahoo Finance and stores it in a local database at `data/sentinel.duckdb`.

You only need to re-run this when you want fresh data — otherwise the stored prices are reused.

### Step 2 — Build features

```
sentinel features build AAPL
```

Computes ~30 technical indicators (moving averages, RSI, momentum, volatility, volume ratios, etc.) and a "target" column (did the stock go up tomorrow? 1 or 0). These get saved to the database.

### Step 3 — Train a model

```
sentinel train AAPL
```

Trains a logistic regression (the default) to predict whether the stock goes up tomorrow, using the features you just built. Prints the accuracy on a held-out test set.

You can pick a different model with `--model`:

```
sentinel train AAPL --model random_forest
sentinel train AAPL --model xgboost
sentinel train AAPL --model lightgbm
```

Trained models are saved to disk so you can reuse them without retraining.

### Step 4 — Evaluate the model

```
sentinel evaluate AAPL
```

Does a **walk-forward** evaluation — it repeatedly trains on past data and tests on the next chunk of future data, sliding forward in time. This simulates how the model would have actually performed in real trading. You get an honest estimate of accuracy, F1, ROC-AUC, and a few other metrics.

Same `--model` flag works here:

```
sentinel evaluate AAPL --model xgboost
```

### Step 5 — Backtest the model as a trading strategy

```
sentinel backtest AAPL
```

Takes the model's predictions and simulates actually trading on them: go long when the model says "up" with confidence above 55%, exit otherwise. Charges transaction costs. Compares against buy-and-hold.

Prints total return, Sharpe ratio, max drawdown, number of trades, and more.

Useful flags:

```
sentinel backtest AAPL --long-threshold 0.6 --short-threshold 0.4 --allow-short
```

- `--long-threshold` → how confident the model must be before going long (default 0.55)
- `--short-threshold` → how pessimistic before going short (default 0.45, only used with `--allow-short`)
- `--allow-short` → let the strategy bet against the stock too
- `--cost-bps 2.0` → transaction cost per trade in basis points (default 2)
- `--vol-target 0.10` → size positions to target 10% annualized volatility

### Step 6 — Generate the latest prediction

```
sentinel predict AAPL
```

Uses your saved model to produce a prediction for the most recent trading day — "probability the stock goes up tomorrow."

---

## Part 5 — Bonus commands

### See which version is installed

```
sentinel version
```

### Explain which features the model cares about

```
sentinel explain AAPL
```

Ranks features by how much the model actually uses them. Default method is permutation importance. Try `--method shap` for a more detailed view (requires the optional `shap` library installed).

### Analyze performance by market regime

```
sentinel regimes AAPL
```

Slices backtest performance by volatility bucket (low/medium/high) and trend (bull/bear). Answers the question "when does this strategy actually work?"

### Compare sentiment vs technicals (requires Reddit data)

```
sentinel ingest reddit --whitelist AAPL
sentinel score-sentiment
sentinel ablate AAPL
```

Pulls Reddit posts mentioning the ticker, scores their sentiment, and compares three models: technicals-only, sentiment-only, and both combined. Tells you whether sentiment actually adds predictive value.

### Get help on any command

Add `--help` to any command to see all its options:

```
sentinel --help
sentinel backtest --help
sentinel ingest prices --help
```

---

## Part 6 — Reading the output

### Training output

After `sentinel train AAPL` you'll see:

```
✓ Trained logistic on AAPL. Holdout accuracy = 0.528 (baseline = 0.520).
```

- **Holdout accuracy** — how often the model guessed the direction right on data it hadn't seen during training
- **Baseline** — what you'd get by always guessing "up" (the majority class). If your model isn't beating this, it's not learning anything useful

### Backtest output

The backtest table shows strategy metrics side-by-side with buy-and-hold:

- **Total return** — percent gained/lost over the entire test period
- **Annualized return** — what that would be per year
- **Sharpe** — risk-adjusted return. Higher = more return per unit of volatility. Anything above 1 is decent, above 2 is great (and rare)
- **Max drawdown** — worst peak-to-trough loss. Lower magnitude (closer to 0) is better
- **Win rate** — fraction of trades that made money
- **N trades** — total number of round-trip trades
- **Exposure** — fraction of time the strategy was in a position (vs. sitting in cash)

**Reality check:** any honest strategy on public equity data with realistic costs will struggle to beat buy-and-hold. If you see a backtest claiming Sharpe 3+ with low turnover, something's wrong (usually look-ahead bias or overfitting). Sentinel's evaluation is walk-forward specifically to prevent that, so the numbers you see are closer to reality than most backtest results floating around online.

---

## Part 7 — Common problems and fixes

### `zsh: command not found: sentinel`

You forgot to activate the venv. Run:

```
source .venv/bin/activate
```

### `externally-managed-environment` error from pip

You ran `pip install` outside the venv. Activate it first, then try again. If the venv doesn't exist yet, redo Part 1.

### `BinderException: table features has X columns but Y values were supplied`

Your local database has a stale schema from a previous run with different columns. Wipe it and re-ingest:

```
rm data/sentinel.duckdb
sentinel demo SPY
```

You'll lose any previously ingested data, but the demo rebuilds everything from scratch.

### `No features for SYMBOL` error

You skipped a step. Make sure you've run these in order:

```
sentinel ingest prices SYMBOL
sentinel features build SYMBOL
sentinel train SYMBOL
```

### `No saved <model> model for SYMBOL` when running `predict` or `explain`

You need to train that model first:

```
sentinel train SYMBOL --model xgboost
sentinel predict SYMBOL --model xgboost
```

The model name must match between train and predict/explain.

### `sentinel demo SPY` hangs on "Fetching SPY from yfinance..."

Yahoo Finance is rate-limiting or down. Wait a few minutes and try again. If it keeps happening, check `https://finance.yahoo.com/quote/SPY` in a browser — if that doesn't load either, it's a Yahoo outage, not your fault.

---

## Part 8 — What to do next

Once the basics work, interesting experiments to try:

1. **Run the demo on 5–10 different tickers.** See which ones the model seems to understand and which it doesn't. Tech mega-caps usually work better than small-caps with low volume.

2. **Compare models.** Run `sentinel train SYMBOL --model logistic`, then `--model random_forest`, then `--model xgboost`. Check `sentinel evaluate` results for each. Usually xgboost wins marginally, but not always.

3. **Tune the backtest.** Try higher thresholds like `--long-threshold 0.6` — more selective entries usually mean fewer trades but higher average quality. See what happens to Sharpe.

4. **Look at feature importance.** Run `sentinel explain AAPL` and see which indicators the model actually leans on. If it's leaning hard on one feature, that's worth investigating — could be signal, could be a data leak.

5. **Compare stocks vs crypto.** `sentinel demo BTC-USD` — different volatility profile, different cost structure (use `--cost-bps 10` for crypto to be realistic).

6. **Try the sentiment workflow.** If you have Reddit API credentials configured, the `ablate` command is the most interesting thing in the repo — it tells you whether social sentiment is actually adding predictive value or just noise.
