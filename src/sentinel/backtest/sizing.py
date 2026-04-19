"""Position sizing: convert unit signals into vol-aware position weights.

The default backtest uses fixed-size positions in ``{-1, 0, +1}``. That's a
clean interpretability choice for showing "does the model pick direction
correctly?", but it's not how real portfolios are managed. A fixed unit
position takes a *lot* more risk in a high-vol regime than a low-vol one,
which means the strategy's P&L is dominated by whatever happens during
volatile periods — masking whether the edge is consistent.

Volatility targeting fixes this by scaling each position so that the
*expected* contribution to portfolio volatility is constant through time:

    size_t = min( target_vol_annual / realized_vol_annual_t , max_leverage )

where ``realized_vol_annual_t`` is the rolling std of daily returns,
annualized. This module provides the two pure-function primitives the
engine uses; the 1-bar look-ahead shift is still the engine's job so that
sizing, direction signal, and cost accounting all use the same shifted
position series.

Notes on look-ahead: ``realized_volatility`` at index ``t`` uses returns
*through* ``t`` (the ``.rolling(window).std()`` is right-edge aligned).
That is fine because the engine shifts the final position forward by one
bar before multiplying with returns — so the position held on day ``t+1``
only depends on information available by close of day ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def realized_volatility(
    returns: pd.Series,
    *,
    window: int = 20,
    periods_per_year: int = 252,
) -> pd.Series:
    """Rolling standard deviation of returns, annualized.

    Uses sample std (``ddof=1``) so a single-value window returns NaN
    rather than 0. The first ``window-1`` rows are NaN by construction.
    """
    if window < 2:
        raise ValueError(f"vol window must be >= 2, got {window}")
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be > 0, got {periods_per_year}")
    std = returns.rolling(window=window, min_periods=window).std(ddof=1)
    return std * np.sqrt(periods_per_year)


def vol_target_scale(
    realized_vol_annual: pd.Series,
    *,
    target_vol_annual: float,
    max_leverage: float = 1.0,
    min_denominator: float = 1e-6,
) -> pd.Series:
    """Per-bar position scale so that ``size * realized_vol == target_vol``.

    Parameters
    ----------
    realized_vol_annual:
        Annualized realized vol series (output of :func:`realized_volatility`).
    target_vol_annual:
        Desired portfolio vol, e.g. 0.10 for 10% annualized.
    max_leverage:
        Cap on the scale. 1.0 means "never leverage beyond a fully-invested
        unit position". Setting it higher lets the strategy lever up in
        quiet regimes.
    min_denominator:
        Floor on the vol denominator to avoid div-by-zero / infinities
        when an input window has identical returns.

    Returns
    -------
    A float series aligned to ``realized_vol_annual``. NaN rows (warmup)
    stay NaN — the caller is responsible for deciding how to handle those
    (the engine treats NaN → 0 position, i.e. "stay flat during warmup").
    """
    if target_vol_annual <= 0:
        raise ValueError(f"target_vol_annual must be > 0, got {target_vol_annual}")
    if max_leverage <= 0:
        raise ValueError(f"max_leverage must be > 0, got {max_leverage}")

    denom = realized_vol_annual.clip(lower=min_denominator)
    raw_scale = target_vol_annual / denom
    # Preserve NaN from warmup.
    scale = raw_scale.where(realized_vol_annual.notna(), other=np.nan)
    # Cap without touching NaN.
    capped = scale.clip(upper=max_leverage)
    return capped
