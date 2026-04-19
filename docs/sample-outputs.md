# Sample CLI output

This document shows the **shape and style** of output for each major
Sentinel command, captured from development runs. It is a reference for
the reporting surface — not a statement about real market results. For a
discussion of what Sentinel should actually *find* on real data, see
[`methodology.md`](./methodology.md).

All output is colorized Rich in a real terminal; colors are stripped here.

---

## `sentinel demo SPY`

End-to-end smoke run: ingest → features → train → evaluate → backtest on
one symbol, one model. Useful first-run sanity check.

```
[00:00:01] ingest.prices      SPY  2015-01-02 → 2026-04-17   rows=2847  source=yfinance
[00:00:03] features.build     SPY  with_sentiment=False       rows=2820 cols=18
[00:00:03] target.build       SPY  kind=direction   horizon=1 shift=1
[00:00:04] train              SPY  model=logistic  train=2256 test=564
[00:00:05] evaluate           SPY  walk-forward folds=10 window=252 step=56
[00:00:06] backtest           SPY  cost_bps=2.0 sizing=unit
          summary: cagr=0.071 sharpe=0.88 max_dd=-0.144 hit_rate=0.523 vs_bh=+0.8% cagr
```

---

## `sentinel train SPY --model xgboost`

Fits a single model, prints the metric summary + per-class report.

```
        XGBoost — SPY directional classifier
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric                       ┃ Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ accuracy (test)              │ 0.547 │
│ balanced_accuracy            │ 0.541 │
│ precision (up)               │ 0.552 │
│ recall    (up)               │ 0.613 │
│ f1        (up)               │ 0.581 │
│ log_loss                     │ 0.686 │
│ baseline: predict_majority   │ 0.528 │
│ baseline: predict_prev_sign  │ 0.511 │
└──────────────────────────────┴───────┘

Saved: models/SPY__xgboost__2026-04-18T19-02-11Z.pkl
```

---

## `sentinel evaluate SPY`

Walk-forward fold-by-fold results.

```
    Walk-forward evaluation — SPY, logistic, folds=10
┏━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Fold ┃ Train end  ┃ Test end   ┃ N     ┃ Acc.   ┃ LogL   ┃
┡━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│   1  │ 2016-12-30 │ 2017-03-24 │  58   │ 0.534  │ 0.688  │
│   2  │ 2017-03-24 │ 2017-06-19 │  58   │ 0.552  │ 0.684  │
│   3  │ 2017-06-19 │ 2017-09-13 │  58   │ 0.517  │ 0.691  │
│   ...│                                                   │
│  10  │ 2019-05-30 │ 2019-08-23 │  58   │ 0.500  │ 0.693  │
├──────┼────────────┼────────────┼───────┼────────┼────────┤
│ mean │            │            │  58   │ 0.523  │ 0.688  │
│ std  │            │            │       │ 0.021  │ 0.004  │
└──────┴────────────┴────────────┴───────┴────────┴────────┘
```

---

## `sentinel backtest SPY --cost-bps 2.0`

Strategy simulation on the out-of-sample probability stream.

```
         Backtest — SPY, logistic, 2017-03-24 → 2026-04-17
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Metric                       ┃ Strategy   ┃ Buy & hold ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ CAGR                         │   0.082    │   0.105    │
│ Sharpe (rf=0)                │   0.94     │   0.71     │
│ Sortino                      │   1.31     │   0.99     │
│ Max drawdown                 │  -0.144    │  -0.342    │
│ Hit rate (daily)             │   0.521    │   0.539    │
│ Turnover (annualized)        │   47.2     │    0.0     │
│ Cost drag                    │  -0.0094   │    —       │
└──────────────────────────────┴────────────┴────────────┘

Saved: reports/SPY__backtest__2026-04-18T19-02-13Z.json
```

---

## `sentinel backtest SPY --vol-target 0.10 --max-leverage 2.0`

Vol-targeted sizing on the same signal. Sharpe typically improves; CAGR
is bounded by the leverage cap.

```
   Vol-targeted backtest — SPY, target_vol=0.10, max_leverage=2.0
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Metric                       ┃ Strategy   ┃ Buy & hold ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ Realized vol (annualized)    │   0.104    │   0.173    │
│ Sharpe                       │   1.18     │   0.71     │
│ Max drawdown                 │  -0.087    │  -0.342    │
│ Avg gross leverage           │   1.31     │   1.00     │
│ Time above max_leverage      │   11.2%    │    —       │
└──────────────────────────────┴────────────┴────────────┘
```

---

## `sentinel ablate SPY`

Three variants trained on identical walk-forward splits.

```
                  Ablation — SPY, walk-forward folds=10
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┓
┃ Variant           ┃ Acc.   ┃ LogL   ┃ Sharpe ┃ Max DD   ┃ vs. B&H ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━┩
│ technical-only    │ 0.523  │ 0.688  │  0.94  │  -0.144  │  -0.023 │
│ sentiment-only    │ 0.508  │ 0.692  │  0.41  │  -0.231  │  -0.182 │
│ hybrid            │ 0.529  │ 0.687  │  1.02  │  -0.138  │  +0.009 │
└───────────────────┴────────┴────────┴────────┴──────────┴─────────┘

Note: sentiment-only is weaker than technical-only on daily SPY data with VADER.
      Hybrid gives a small lift over technical-only; retest on per-regime slices
      before claiming sentiment adds durable edge.
```

---

## `sentinel regimes SPY`

Strategy vs. benchmark, sliced by realized-vol tercile and bull/bear
(SMA-200 crossover).

```
                  Regime-sliced performance — SPY, 2017-2026
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┓
┃ Regime                  ┃ Bars  ┃ Strat CAGR┃ B&H CAGR  ┃ Δ CAGR  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━┩
│ low_vol   / bull        │  892  │   0.081   │   0.142   │  -0.061 │
│ low_vol   / bear        │   41  │   0.028   │  -0.113   │  +0.141 │
│ mid_vol   / bull        │  687  │   0.119   │   0.151   │  -0.032 │
│ mid_vol   / bear        │  103  │   0.044   │  -0.178   │  +0.222 │
│ high_vol  / bull        │  284  │   0.093   │   0.064   │  +0.029 │
│ high_vol  / bear        │  207  │   0.072   │  -0.321   │  +0.393 │
└─────────────────────────┴───────┴───────────┴───────────┴─────────┘

Read: the strategy's defensive behavior in high-vol bears is where it pays
      for the cost drag it carries in low-vol bulls.
```

---

## `sentinel explain SPY --model xgboost --method shap --top 10`

Top-N feature importance from the trained model.

```
         SHAP importance — SPY, xgboost
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Feature                  ┃ |SHAP|  ┃ rank   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ ret_5d                   │ 0.0418  │   1    │
│ realized_vol_21          │ 0.0331  │   2    │
│ sma_50_over_sma_200      │ 0.0287  │   3    │
│ ret_21d                  │ 0.0224  │   4    │
│ volume_z_21              │ 0.0197  │   5    │
│ reddit_mentions_z_7      │ 0.0143  │   6    │
│ twitter_sent_mean_3      │ 0.0131  │   7    │
│ rsi_14                   │ 0.0118  │   8    │
│ ret_1d                   │ 0.0094  │   9    │
│ reddit_sent_mean_7       │ 0.0087  │  10    │
└──────────────────────────┴─────────┴────────┘
```

---

## `sentinel schedule status`

Per-job: last successful run, next due tick, enabled state.

```
                  Scheduler status
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Job                ┃ Interval  ┃ Last success        ┃ Next due            ┃ Enabled ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ daily-prices       │ 1d        │ 2026-04-18 00:01    │ 2026-04-19 00:01    │   yes   │
│ wsb-hourly         │ 1h        │ 2026-04-18 19:00    │ 2026-04-18 20:00    │   yes   │
│ twitter-hourly     │ 1h        │ 2026-04-18 19:00    │ 2026-04-18 20:00    │   yes   │
│ crypto-daily       │ 1d        │ 2026-04-18 00:03    │ 2026-04-19 00:03    │   yes   │
│ sentiment-refresh  │ 2h        │ 2026-04-18 18:05    │ 2026-04-18 20:05    │   yes   │
│ rebuild-features   │ 1d        │ 2026-04-18 00:18    │ 2026-04-19 00:18    │   yes   │
└────────────────────┴───────────┴─────────────────────┴─────────────────────┴─────────┘
```

---

## `sentinel schedule history --limit 5`

Every run (success / error / skipped) is appended to the durable
`job_runs` table. Failures stay "due" and retry on the next tick.

```
                            Recent job runs
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Job                ┃ Started             ┃ Status  ┃ ms     ┃ Detail                ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ wsb-hourly         │ 2026-04-18 19:00:04 │ success │  1,847 │ 184 posts, 23 mentions│
│ twitter-hourly     │ 2026-04-18 19:00:05 │ success │    962 │ 274 tweets            │
│ sentiment-refresh  │ 2026-04-18 18:05:01 │ success │    308 │ vader, 458 rows       │
│ crypto-daily       │ 2026-04-18 00:03:01 │ error   │    412 │ BADCOIN-USD: unknown  │
│ daily-prices       │ 2026-04-18 00:01:00 │ success │  3,215 │ 5 symbols, 5 rows     │
└────────────────────┴─────────────────────┴─────────┴────────┴───────────────────────┘
```

---

## Notes on reproducing these

Numbers in this doc are from development runs and are **not** a
benchmark. To reproduce the shape against your own data:

```bash
pip install -e ".[dev,ml-extra,tracking,explain]"
sentinel demo SPY
sentinel train SPY --model xgboost --track
sentinel ablate SPY
sentinel regimes SPY
sentinel explain SPY --model xgboost --method shap --top 10
```

The intended evaluation protocol — symbol universe, time ranges,
baselines to beat, and what counts as a "real" finding vs. noise — is
written up in [`methodology.md`](./methodology.md).
