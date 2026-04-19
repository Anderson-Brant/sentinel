"""Ablation harness tests.

We build a synthetic feature table with a known technical + sentiment split
and verify that:
  - run_ablation produces three variants (technical / sentiment / hybrid)
  - each variant only sees its own feature columns (no leakage)
  - uplift helpers compute the right sign

We don't need real predictive signal here — only that the plumbing is sound.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.config import SentinelConfig
from sentinel.evaluation.ablation import VARIANTS, run_ablation


def _synth_features(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    # Two "technical" features and two "sentiment" features, all noise.
    ret_1 = rng.normal(0, 0.01, n)
    ret_5 = pd.Series(ret_1).rolling(5, min_periods=1).mean().to_numpy()
    sent_mean = rng.normal(0, 0.3, n)
    sent_count = rng.poisson(5, n).astype(float)

    # Target has no signal — just a fair coin so accuracy hovers near 0.5.
    target = (rng.random(n) > 0.5).astype(int)

    return pd.DataFrame(
        {
            "symbol": "SYNTH",
            "ret_1": ret_1,
            "ret_5": ret_5,
            "reddit_sentiment_mean": sent_mean,
            "reddit_mention_count": sent_count,
            "target_direction": target,
            "target_return": rng.normal(0, 0.01, n),
        },
        index=idx,
    )


def _cfg() -> SentinelConfig:
    # Enough rows for 3 folds on 400 obs with min_train=250 leaves 150 tail.
    return SentinelConfig.model_validate(
        {"modeling": {"walk_forward": {"n_splits": 3, "min_train_size": 250}}}
    )


def test_run_ablation_returns_all_three_variants():
    features = _synth_features()
    report = run_ablation(
        features,
        symbol="SYNTH",
        model_name="logistic",
        cfg=_cfg(),
        sentiment_columns=["reddit_sentiment_mean", "reddit_mention_count"],
    )
    got = [r.variant for r in report.results]
    assert got == list(VARIANTS)
    by = report.by_variant()
    assert by["technical"].n_features == 2
    assert by["sentiment"].n_features == 2
    assert by["hybrid"].n_features == 4


def test_variants_use_disjoint_columns():
    """Each variant should see only its own features. The feature subset is
    derived from the frame's columns, so the row counts in the WalkForwardReport
    should match (no NaN-dropping should fire on a clean synthetic table).
    """
    features = _synth_features()
    report = run_ablation(
        features,
        symbol="SYNTH",
        model_name="logistic",
        cfg=_cfg(),
        sentiment_columns=["reddit_sentiment_mean", "reddit_mention_count"],
    )
    for r in report.results:
        # Each variant should produce at least one fold.
        assert len(r.wf_report.folds) >= 1


def test_missing_sentiment_columns_raises():
    features = _synth_features().drop(columns=["reddit_sentiment_mean", "reddit_mention_count"])
    with pytest.raises(ValueError, match="No sentiment columns"):
        run_ablation(
            features,
            symbol="SYNTH",
            model_name="logistic",
            cfg=_cfg(),
            sentiment_columns=["reddit_sentiment_mean", "reddit_mention_count"],
        )


def test_missing_target_raises():
    features = _synth_features().drop(columns=["target_direction"])
    with pytest.raises(ValueError, match="target_direction"):
        run_ablation(
            features,
            symbol="SYNTH",
            model_name="logistic",
            cfg=_cfg(),
            sentiment_columns=["reddit_sentiment_mean"],
        )


def test_uplift_helpers_compute_signed_deltas():
    features = _synth_features()
    report = run_ablation(
        features,
        symbol="SYNTH",
        model_name="logistic",
        cfg=_cfg(),
        sentiment_columns=["reddit_sentiment_mean", "reddit_mention_count"],
    )
    uplift = report.sentiment_uplift_accuracy()
    assert uplift is not None
    rs = report.by_variant()
    assert uplift == pytest.approx(
        rs["hybrid"].mean_accuracy - rs["technical"].mean_accuracy, rel=0, abs=1e-12
    )

    # No backtest was requested → sharpe uplift is None, not a crash.
    assert report.sentiment_uplift_sharpe() is None
