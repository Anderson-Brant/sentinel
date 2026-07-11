# `sentinel analyze`: the long-term scorecard

`sentinel analyze TICKER` produces a five-row scorecard for long-term equity
analysis: quality, valuation, price history, insiders, and competitive
position. Each row is a letter grade plus one line of evidence. It produces
structured information for a human decision. It does not say "buy".

```
$ sentinel analyze AAPL

AAPL · Apple Inc. · Consumer Electronics · $4.6T mcap · as of 2026-07-11

Quality        A     ROIC 58%. Gross margin 46%, stable. Net cash. Revenue +6%/yr.
Valuation      D+    P/E 38 (83rd pct vs own history). PEG 1.8.
Price hist     A+    10y CAGR 27%. Max DD 39% (recovered in 9mo).
Insiders       C     Net selling 0.2% of shares over 6mo (mild distribution). 1 buys / 14 sells.
Competitive    B     Revenue +6%/yr vs sector +10%. Op margin 32% vs sector 22%.

Composite: B
Related: MSFT, GOOGL, NVDA
```

## Options

| Flag | Meaning |
|---|---|
| `--detail quality` | ROIC, margins + stability, leverage, growth, numeric score |
| `--detail valuation` | All ratios + own-history percentiles |
| `--detail price` | Per-horizon CAGR, drawdown, recovery, Sharpe |
| `--detail insiders` | Net 6/12mo activity, buy/sell counts |
| `--detail competitive` | Growth and margin vs the sector baseline |
| `--start` | History start when prices need fetching (default `2000-01-01`) |
| `--offline` | No network. Stored prices only; only the price history row grades |
| `--no-related` | Skip the correlation-based related-tickers line |

Prices come from the store (`sentinel ingest prices` populates it); if the
symbol has never been ingested, `analyze` fetches from `--start` and persists
the result. Fundamentals and insider transactions come from yfinance at call
time. Related tickers are ranked by daily-return correlation against the
other symbols in your store, so the list gets better as you ingest more.

## How the grades work

Grades share one scale: `A+ A A- B+ B B- C+ C C- D+ D F`. Each row carries a
numeric 0-100 score (visible via `--detail`); the composite is the mean of
the numeric scores of the rows that could be computed, mapped back to a
letter. Rows without enough data show no grade and stay out of the composite.

### Quality

Seven metrics, each scored against absolute bands and combined with fixed
weights: ROIC (25%), gross margin level (15%) and stability (10%), operating
margin trend (10%), net debt / EBITDA (15%), revenue growth (15%) and growth
stability (10%). ROIC is operating income after a 21% notional tax over
invested capital (equity + debt - cash), latest fiscal year. Bands are
universal rather than sector-relative for now; sector-relative calibration
needs per-peer fundamentals ingestion, which arrives with the screener work.

### Valuation: vs the stock's own history

The current P/E, P/S, and P/FCF are ranked against the stock's own monthly
ratio history (annual per-share fundamentals treated as a step function under
monthly closes). The mean percentile maps to a grade on a scale calibrated so
the middle of a stock's own range reads as a middle grade: bottom 5% of its
range is A+, 50th percentile is B-, and only the most expensive decile grades
D or F. A PEG below 1 bumps the grade a notch; above 3 costs one.

Ratios needing positive denominators (P/E with negative earnings, P/FCF with
negative FCF) are omitted rather than reported as meaningless negatives.
EV/EBITDA and dividend yield are shown in `--detail valuation` but don't
enter the grade yet.

### Price history: absolute thresholds

The grade comes from CAGR over the longest fully-covered horizon (10y
preferred, falling back to 5y/3y/1y):

| 10y CAGR | Grade |
|---|---|
| ≥ 20% | A+ |
| ≥ 15% | A |
| ≥ 12% | A- |
| ≥ 10% | B+ |
| ≥ 8% | B |
| ≥ 6% | B- |
| ≥ 4% | C+ |
| ≥ 2% | C |
| ≥ 0% | C- |
| ≥ -3% | D+ |
| ≥ -8% | D |
| below | F |

A max drawdown deeper than 60% costs one notch. CAGR uses `adj_close` so
splits and dividends don't distort it.

### Insiders

Net insider buying minus selling over the trailing 6 months, as a percent of
shares outstanding (12-month figure shown in `--detail insiders`). Net buying
above 0.5% grades A ("strong accumulation"); net selling beyond 1.5% grades D
("heavy distribution"); the middle is graded in steps between. Routine grants
and awards are excluded; only classifiable open-market buys and sells count.
Data comes from Yahoo's Form 4 feed via yfinance. Direct SEC EDGAR ingestion
with a durable local table is the planned upgrade.

Insider selling is weak evidence in isolation (people sell for taxes, houses,
and diversification; they buy for one reason), which is why this row gets the
same weight as the others rather than more.

### Competitive

Revenue growth and operating margin compared against a static table of
sector medians (large-cap US, rough figures). Beating your sector's growth
and margin profile grades well; lagging it grades poorly. This is the
weakest row methodologically until real peer-set comparison ships: the
baseline table is coarse, and "sector median" hides a lot of dispersion.
Treat it as context, not signal.

## Data sources and their limits

Everything comes from yfinance (free, unofficial). Annual statements only go
back ~4-5 fiscal years, so "own history" percentiles and growth/stability
figures cover that window, not a full market cycle. Per-share history uses
current shares outstanding, so a large buyback or dilution skews older
ratios. Insider data reflects whatever Yahoo surfaces from Form 4 filings.
SEC EDGAR ingestion will extend and firm up all of this.
