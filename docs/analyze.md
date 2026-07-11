# `sentinel analyze`: the long-term scorecard

`sentinel analyze TICKER` produces a five-row scorecard for long-term equity
analysis: quality, valuation, price history, insiders, and competitive
position. Each row is a letter grade plus one line of evidence. It produces
structured information for a human decision. It does not say "buy".

As of v0.2.0 two rows are real (valuation, price history); quality lands in
v0.3, insiders in v0.5, competitive in v0.6.

```
$ sentinel analyze AAPL

AAPL · Apple Inc. · Consumer Electronics · $3.2T mcap · as of 2026-07-10

Quality        —    (pending, v0.3)
Valuation      C+   P/E 33 (78th pct vs own history). PEG 2.6.
Price hist     A    10y CAGR 18%. Max DD 38% (recovered in 21mo).
Insiders       —    (pending, v0.5)
Competitive    —    (pending, v0.6)

Composite: B+   (price history + valuation only)
```

## Options

| Flag | Meaning |
|---|---|
| `--detail price` | Expand the price-history row: per-horizon CAGR, drawdown, recovery, Sharpe, numeric score |
| `--detail valuation` | Expand the valuation row: all ratios + own-history percentiles |
| `--start` | History start when prices need fetching (default `2000-01-01`; earlier = longer CAGR windows) |
| `--offline` | No network. Uses stored prices only; the valuation row shows as unavailable |

Prices come from the store (`sentinel ingest prices` populates it); if the
symbol has never been ingested, `analyze` fetches from `--start` and persists
the result. Fundamentals come from yfinance at call time.

## How the grades work

Grades share one scale: `A+ A A- B+ B B- C+ C C- D+ D F`. Each row also
carries a numeric 0-100 score (visible via `--detail`); the composite is the
mean of the numeric scores of the rows that could be computed, mapped back to
a letter.

### Price history: absolute thresholds

Long-run price performance is graded against universal benchmarks, not
sector-relative. The grade comes from CAGR over the longest fully-covered
horizon (10y preferred, falling back to 5y/3y/1y):

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
splits and dividends don't distort it. When less than 10 years of history is
available the summary says so and the grade is based on what exists.

### Valuation: vs the stock's own history

The current P/E, P/S, and P/FCF are ranked against the stock's own monthly
ratio history (annual per-share fundamentals treated as a step function under
monthly closes). The mean percentile maps to a grade on a scale calibrated so
the middle of a stock's own range reads as a middle grade: bottom 5% of its
range is A+, 50th percentile is B-, and only the most expensive decile grades
D or F. A PEG below 1 bumps the grade a notch; above 3 costs one.

Sector-relative grading (per the methodology decision log) arrives with the
quality factor in v0.3. Until then "expensive vs itself" is the whole story,
so read the grade with that limitation in mind. EV/EBITDA and dividend yield
are computed and shown in `--detail valuation` but don't enter the grade yet.

Ratios needing positive denominators (P/E with negative earnings, P/FCF with
negative FCF) are omitted rather than reported as meaningless negatives. If
nothing survives, the row shows `insufficient fundamentals data` with no grade and the
composite is computed from the remaining rows.

## Data sources and their limits

Everything comes from yfinance (free, unofficial). Annual statements only go
back ~4-5 fiscal years, so "own history" percentiles cover that window, not a
full market cycle. Per-share history uses current shares outstanding, so a
large buyback or dilution skews older ratios. SEC EDGAR ingestion (v0.5, with
insiders) will extend and firm up the fundamentals history.
