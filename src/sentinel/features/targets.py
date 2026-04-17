"""Prediction target generation.

Two canonical targets for the MVP:
    - ``target_direction``: 1 if forward-return > 0, else 0.
    - ``target_return``:    forward percentage return over ``horizon`` days.

Forward returns are computed with .shift(-horizon) and aligned to time t, which
is exactly what a model at time t would need to predict. We drop the trailing
``horizon`` rows (target unknown) in the pipeline.
"""

from __future__ import annotations

import pandas as pd


def forward_return(prices: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Percent return from close_t to close_{t+horizon}."""
    close = prices["close"]
    return close.pct_change(horizon).shift(-horizon)


def directional_target(prices: pd.DataFrame, horizon: int = 1) -> pd.Series:
    ret = forward_return(prices, horizon=horizon)
    return (ret > 0).astype("Int8").where(ret.notna())
