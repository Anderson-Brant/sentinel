"""Tests for the finBERT sentiment scorer.

torch and transformers are heavyweight optional deps, so we stub both
via sys.modules. What we're actually pinning down:

    * Lazy import: FinBertScorer doesn't touch torch/transformers until
      instantiated, and raises a helpful ImportError when they're missing.
    * Label-to-VADER-dict mapping: compound = pos − neg, pos/neg/neu
      come from the softmaxed logits indexed by id2label.
    * Label map robustness: respects id2label ordering, rejects checkpoints
      missing 'positive' or 'negative'.
    * Batching: polarity_scores_batch groups texts into batch_size-sized
      chunks and invokes the model once per chunk.
    * score_posts integration: finBERT plugs into score_posts exactly like
      VADER without any pipeline changes.
"""

from __future__ import annotations

import math
import sys
import types
from typing import Any

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fake torch + transformers
# ---------------------------------------------------------------------------


def _install_fake_torch(monkeypatch, *, cuda_available: bool = False) -> dict[str, list]:
    """Install a stub torch module; return a call log."""
    log: dict[str, list] = {"no_grad_enter": 0, "softmax_calls": []}

    class _Tensor:
        def __init__(self, arr: np.ndarray):
            self._arr = np.asarray(arr, dtype=float)

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

        def __getitem__(self, idx):
            return _Tensor(self._arr[idx])

    class _NoGrad:
        def __enter__(self):
            log["no_grad_enter"] += 1
            return None

        def __exit__(self, *exc):
            return False

    def softmax(logits, dim=-1):
        log["softmax_calls"].append(dim)
        arr = logits._arr if isinstance(logits, _Tensor) else np.asarray(logits, dtype=float)
        shifted = arr - arr.max(axis=dim, keepdims=True)
        ex = np.exp(shifted)
        return _Tensor(ex / ex.sum(axis=dim, keepdims=True))

    class _Cuda:
        @staticmethod
        def is_available():
            return cuda_available

    torch_mod = types.ModuleType("torch")
    torch_mod.no_grad = lambda: _NoGrad()
    torch_mod.softmax = softmax
    torch_mod.cuda = _Cuda()
    torch_mod._Tensor = _Tensor  # exposed for fake model internals
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    return log


class _FakeEncoding(dict):
    """Mock of a transformers BatchEncoding — dict-like, with ``.to``."""

    def to(self, device):
        self["_device"] = device
        return self


def _install_fake_transformers(
    monkeypatch,
    *,
    id2label: dict[int, str] | None,
    forward: Any,
    tokenizer_log: list | None = None,
):
    """Install stub transformers. ``forward`` is a callable called with the
    encoded dict and must return an object with ``.logits`` (a ``_Tensor``)."""
    import sys as _sys

    torch_mod = _sys.modules["torch"]
    _Tensor = torch_mod._Tensor

    class _FakeTokenizer:
        def __init__(self):
            self.calls: list = tokenizer_log if tokenizer_log is not None else []

        def __call__(self, texts, padding=True, truncation=True, max_length=256, return_tensors="pt"):
            self.calls.append(
                {
                    "n": len(texts),
                    "padding": padding,
                    "truncation": truncation,
                    "max_length": max_length,
                    "return_tensors": return_tensors,
                }
            )
            return _FakeEncoding({"input_ids": _Tensor(np.zeros((len(texts), 4)))})

        @classmethod
        def from_pretrained(cls, name):
            return cls()

    class _Config:
        def __init__(self, id2label):
            self.id2label = id2label

    class _FakeModel:
        def __init__(self, cfg_map):
            self.config = _Config(cfg_map) if cfg_map is not None else types.SimpleNamespace()
            self._forward = forward
            self._moved_to: str | None = None

        def to(self, device):
            self._moved_to = device
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            # Wrap in a SimpleNamespace to expose ``.logits``.
            return types.SimpleNamespace(logits=self._forward(kwargs))

        @classmethod
        def from_pretrained(cls, name):
            return cls(id2label)

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoTokenizer = _FakeTokenizer
    transformers_mod.AutoModelForSequenceClassification = _FakeModel
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)


# ---------------------------------------------------------------------------
# Helpers — build logits that softmax to a controlled distribution
# ---------------------------------------------------------------------------


def _logits_for(*triples: tuple[float, float, float]) -> np.ndarray:
    """``triples`` are one (pos, neg, neu) tuple per row. Returns logits
    such that softmax(logits) ≈ those probabilities."""
    probs = np.asarray(triples, dtype=float)
    # Avoid log(0).
    probs = np.clip(probs, 1e-9, None)
    return np.log(probs)


# ---------------------------------------------------------------------------
# Tests — missing libs
# ---------------------------------------------------------------------------


def test_finbert_missing_torch_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    from sentinel.features.finbert import FinBertScorer

    with pytest.raises(ImportError, match="transformers"):
        FinBertScorer()


# ---------------------------------------------------------------------------
# Tests — single-text scoring
# ---------------------------------------------------------------------------


def test_polarity_scores_returns_vader_shape(monkeypatch):
    _install_fake_torch(monkeypatch)

    # id2label matches ProsusAI/finbert's published order.
    id2label = {0: "positive", 1: "negative", 2: "neutral"}

    def forward(kwargs):
        n = kwargs["input_ids"]._arr.shape[0]
        # All rows: strong positive.
        logits = _logits_for(*([(0.80, 0.10, 0.10)] * n))
        return sys.modules["torch"]._Tensor(logits)

    _install_fake_transformers(monkeypatch, id2label=id2label, forward=forward)

    from sentinel.features.finbert import FinBertScorer

    fb = FinBertScorer()
    out = fb.polarity_scores("earnings blew past every estimate")
    assert set(out) == {"compound", "pos", "neg", "neu"}
    assert math.isclose(out["pos"], 0.80, rel_tol=1e-6)
    assert math.isclose(out["neg"], 0.10, rel_tol=1e-6)
    assert math.isclose(out["neu"], 0.10, rel_tol=1e-6)
    assert math.isclose(out["compound"], 0.70, rel_tol=1e-6)


def test_polarity_scores_negative_compound(monkeypatch):
    _install_fake_torch(monkeypatch)
    id2label = {0: "positive", 1: "negative", 2: "neutral"}

    def forward(kwargs):
        n = kwargs["input_ids"]._arr.shape[0]
        logits = _logits_for(*([(0.15, 0.75, 0.10)] * n))
        return sys.modules["torch"]._Tensor(logits)

    _install_fake_transformers(monkeypatch, id2label=id2label, forward=forward)
    from sentinel.features.finbert import FinBertScorer

    out = FinBertScorer().polarity_scores("missed guidance, cut the dividend")
    assert math.isclose(out["compound"], -0.60, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Tests — label map robustness
# ---------------------------------------------------------------------------


def test_label_map_respects_config_ordering(monkeypatch):
    """A checkpoint with non-canonical ordering should still work."""
    _install_fake_torch(monkeypatch)
    # Flipped: 0=negative, 1=positive, 2=neutral
    id2label = {0: "negative", 1: "positive", 2: "neutral"}

    captured = {}

    def forward(kwargs):
        n = kwargs["input_ids"]._arr.shape[0]
        # Index 1 (positive in this flipped map) gets 0.80 probability.
        logits = _logits_for(*([(0.10, 0.80, 0.10)] * n))  # (idx0, idx1, idx2)
        captured["shape"] = logits.shape
        return sys.modules["torch"]._Tensor(logits)

    _install_fake_transformers(monkeypatch, id2label=id2label, forward=forward)
    from sentinel.features.finbert import FinBertScorer

    out = FinBertScorer().polarity_scores("great earnings")
    # positive is at idx 1, so pos probability is 0.80.
    assert math.isclose(out["pos"], 0.80, rel_tol=1e-6)
    assert math.isclose(out["neg"], 0.10, rel_tol=1e-6)
    assert math.isclose(out["compound"], 0.70, rel_tol=1e-6)


def test_label_map_rejects_missing_labels(monkeypatch):
    _install_fake_torch(monkeypatch)
    # Missing 'negative'.
    id2label = {0: "positive", 1: "neutral"}

    def forward(kwargs):
        return sys.modules["torch"]._Tensor(np.zeros((1, 2)))

    _install_fake_transformers(monkeypatch, id2label=id2label, forward=forward)
    from sentinel.features.finbert import FinBertScorer

    with pytest.raises(ValueError, match="negative"):
        FinBertScorer()


def test_label_map_falls_back_when_id2label_missing(monkeypatch):
    _install_fake_torch(monkeypatch)
    # id2label=None triggers the fallback ordering (pos, neg, neu).

    def forward(kwargs):
        n = kwargs["input_ids"]._arr.shape[0]
        logits = _logits_for(*([(0.80, 0.10, 0.10)] * n))
        return sys.modules["torch"]._Tensor(logits)

    _install_fake_transformers(monkeypatch, id2label=None, forward=forward)
    from sentinel.features.finbert import FinBertScorer

    out = FinBertScorer().polarity_scores("anything")
    assert math.isclose(out["pos"], 0.80, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Tests — batching
# ---------------------------------------------------------------------------


def test_polarity_scores_batch_respects_batch_size(monkeypatch):
    _install_fake_torch(monkeypatch)
    id2label = {0: "positive", 1: "negative", 2: "neutral"}

    tok_log: list = []

    def forward(kwargs):
        n = kwargs["input_ids"]._arr.shape[0]
        logits = _logits_for(*([(0.50, 0.30, 0.20)] * n))
        return sys.modules["torch"]._Tensor(logits)

    _install_fake_transformers(
        monkeypatch, id2label=id2label, forward=forward, tokenizer_log=tok_log
    )
    from sentinel.features.finbert import FinBertScorer

    fb = FinBertScorer(batch_size=3)
    # 7 texts → batches of 3, 3, 1.
    out = fb.polarity_scores_batch([f"text-{i}" for i in range(7)])
    assert len(out) == 7
    batch_sizes = [call["n"] for call in tok_log]
    assert batch_sizes == [3, 3, 1]


def test_polarity_scores_batch_handles_empty(monkeypatch):
    _install_fake_torch(monkeypatch)
    id2label = {0: "positive", 1: "negative", 2: "neutral"}

    def forward(kwargs):
        return sys.modules["torch"]._Tensor(np.zeros((1, 3)))

    _install_fake_transformers(monkeypatch, id2label=id2label, forward=forward)
    from sentinel.features.finbert import FinBertScorer

    assert FinBertScorer().polarity_scores_batch([]) == []


def test_polarity_scores_batch_coerces_non_strings(monkeypatch):
    """NaN and None should be scored as empty strings, not crash the tokenizer."""
    _install_fake_torch(monkeypatch)
    id2label = {0: "positive", 1: "negative", 2: "neutral"}

    seen = []

    def forward(kwargs):
        seen.append(kwargs["input_ids"]._arr.shape[0])
        n = kwargs["input_ids"]._arr.shape[0]
        return sys.modules["torch"]._Tensor(_logits_for(*([(0.34, 0.33, 0.33)] * n)))

    _install_fake_transformers(monkeypatch, id2label=id2label, forward=forward)
    from sentinel.features.finbert import FinBertScorer

    out = FinBertScorer(batch_size=8).polarity_scores_batch(["ok", None, float("nan")])
    assert len(out) == 3
    # All three rows processed — no crash on None/NaN.
    assert sum(seen) == 3


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_constructor_rejects_zero_batch_size(monkeypatch):
    _install_fake_torch(monkeypatch)
    id2label = {0: "positive", 1: "negative", 2: "neutral"}

    def forward(kwargs):
        return sys.modules["torch"]._Tensor(np.zeros((1, 3)))

    _install_fake_transformers(monkeypatch, id2label=id2label, forward=forward)
    from sentinel.features.finbert import FinBertScorer

    with pytest.raises(ValueError, match="batch_size"):
        FinBertScorer(batch_size=0)


def test_constructor_rejects_zero_max_length(monkeypatch):
    _install_fake_torch(monkeypatch)
    id2label = {0: "positive", 1: "negative", 2: "neutral"}

    def forward(kwargs):
        return sys.modules["torch"]._Tensor(np.zeros((1, 3)))

    _install_fake_transformers(monkeypatch, id2label=id2label, forward=forward)
    from sentinel.features.finbert import FinBertScorer

    with pytest.raises(ValueError, match="max_length"):
        FinBertScorer(max_length=0)


# ---------------------------------------------------------------------------
# score_posts integration — finBERT behaves like a VADER scorer
# ---------------------------------------------------------------------------


def test_finbert_integrates_with_score_posts(monkeypatch):
    _install_fake_torch(monkeypatch)
    id2label = {0: "positive", 1: "negative", 2: "neutral"}

    def forward(kwargs):
        n = kwargs["input_ids"]._arr.shape[0]
        return sys.modules["torch"]._Tensor(_logits_for(*([(0.70, 0.15, 0.15)] * n)))

    _install_fake_transformers(monkeypatch, id2label=id2label, forward=forward)
    from sentinel.features.finbert import FinBertScorer
    from sentinel.features.sentiment import score_posts

    posts = pd.DataFrame(
        [
            {"post_id": "p1", "title": "GME moon", "body": "great earnings"},
            {"post_id": "p2", "title": "TSLA dip", "body": "bad quarter"},
        ]
    )
    scorer = FinBertScorer()
    scored = score_posts(posts, scorer=scorer)
    assert list(scored.columns) == [
        "post_id",
        "sentiment_compound",
        "sentiment_pos",
        "sentiment_neg",
        "sentiment_neu",
    ]
    assert len(scored) == 2
    # compound ≈ 0.70 − 0.15 = 0.55 for every row
    assert all(abs(c - 0.55) < 1e-6 for c in scored["sentiment_compound"])


# ---------------------------------------------------------------------------
# get_scorer factory
# ---------------------------------------------------------------------------


def test_get_scorer_returns_finbert(monkeypatch):
    _install_fake_torch(monkeypatch)
    id2label = {0: "positive", 1: "negative", 2: "neutral"}

    def forward(kwargs):
        return sys.modules["torch"]._Tensor(np.zeros((1, 3)))

    _install_fake_transformers(monkeypatch, id2label=id2label, forward=forward)
    from sentinel.features.finbert import FinBertScorer, get_scorer

    s = get_scorer("finbert")
    assert isinstance(s, FinBertScorer)


def test_get_scorer_rejects_unknown():
    from sentinel.features.finbert import get_scorer

    with pytest.raises(ValueError, match="Unknown scorer"):
        get_scorer("sentimentsaurus-rex")
