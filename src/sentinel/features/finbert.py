"""finBERT sentiment scorer — a transformer-based alternative to VADER.

Why this exists
---------------
VADER is a rule-based lexicon scorer tuned on general social-media text.
It handles emojis and slang well, but it misses phrasing that only
makes sense in a financial context: "beat earnings", "guided down",
"priced in", "raised guidance", "missed the top line" all carry strong
directional signal that VADER scores as approximately neutral because
the constituent words are not in its sentiment lexicon.

finBERT (``ProsusAI/finbert``) is a BERT-base model fine-tuned on
financial news headlines. It produces probabilities over
``{positive, negative, neutral}``. This module wraps it in the same
``Scorer`` protocol the rest of the pipeline already expects from
VADER, so nothing downstream has to change.

Why the VADER-shape output
--------------------------
The existing ingestion → scoring → aggregation → features pipeline is
written against VADER's return shape (``compound / pos / neg / neu``).
Keeping the same shape means the same aggregation code, the same
ablation harness, and the same feature columns — the only thing that
changes is the *quality* of the per-post signal. That isolation is what
makes "VADER vs finBERT" a clean ablation rather than a refactor.

Mapping
-------
    pos = P(positive)
    neg = P(negative)
    neu = P(neutral)
    compound = P(positive) − P(negative)       # in [−1, +1]

Dependencies
------------
``torch`` and ``transformers`` are optional — both are lazy-imported in
the constructor. Sentinel still imports and runs without them.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from sentinel.utils.logging import get_logger

log = get_logger(__name__)


_INSTALL_HINT = (
    "finBERT requires `transformers` and `torch`. Install with "
    "`pip install 'transformers[torch]'` to enable --scorer finbert."
)

DEFAULT_MODEL = "ProsusAI/finbert"

# Fallback label ordering for ``ProsusAI/finbert`` — the published
# checkpoint publishes ``id2label`` so this is only used if the config
# is missing (e.g. a local checkpoint with a stripped config).
_FALLBACK_ID2LABEL: dict[int, str] = {0: "positive", 1: "negative", 2: "neutral"}


class FinBertScorer:
    """Transformer-based sentiment scorer with a VADER-compatible output.

    Construction is expensive — it downloads/loads a ~440 MB model and
    tokenizer. Inference is one forward pass per text (batched internally
    to ``batch_size``). For that reason the scorer is designed to be
    constructed once per CLI invocation and reused across every post.

    The class exposes both the Protocol-compatible single-text method
    (``polarity_scores``) and a bulk method (``polarity_scores_batch``)
    so callers that have a list of texts on hand can amortize tokenizer +
    model overhead.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        max_length: int = 256,
        batch_size: int = 16,
    ) -> None:
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as e:  # pragma: no cover - exercised via stubbed test
            raise ImportError(_INSTALL_HINT) from e

        if max_length < 1:
            raise ValueError("max_length must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        self._torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.model_name = model_name

        log.info("Loading finBERT model %s (device=%s)", model_name, device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = (
            AutoModelForSequenceClassification.from_pretrained(model_name)
            .to(device)
            .eval()
        )
        self._label_to_idx = self._build_label_index(self._model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def polarity_scores(self, text: str) -> dict[str, float]:
        """Score a single string. Matches VADER's return shape."""
        return self.polarity_scores_batch([text])[0]

    def polarity_scores_batch(self, texts: Iterable[str]) -> list[dict[str, float]]:
        """Score an iterable of strings in mini-batches."""
        clean = [t if isinstance(t, str) else "" for t in texts]
        if not clean:
            return []

        results: list[dict[str, float]] = []
        for start in range(0, len(clean), self.batch_size):
            chunk = clean[start : start + self.batch_size]
            probs = self._forward_softmax(chunk)
            for row in probs:
                results.append(self._row_to_vader_dict(row))
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_label_index(model: Any) -> dict[str, int]:
        """Return a ``{label_lower: id}`` map, robust to checkpoint drift.

        Raises if either of the critical labels (positive / negative) is
        missing — without both, ``compound`` is ill-defined.
        """
        cfg = getattr(model, "config", None)
        raw = getattr(cfg, "id2label", None) if cfg is not None else None
        if not raw:
            raw = _FALLBACK_ID2LABEL
        mapping: dict[str, int] = {}
        for k, v in raw.items():
            mapping[str(v).strip().lower()] = int(k)

        missing = {"positive", "negative"} - set(mapping)
        if missing:
            raise ValueError(
                f"finBERT label map is missing {sorted(missing)}. Got: {mapping}"
            )
        return mapping

    def _forward_softmax(self, texts: list[str]) -> np.ndarray:
        """Tokenize → forward pass → softmax. Returns ``(n, n_labels)``."""
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Respect device placement if the stubbed tokenizer omits ``.to``.
        to_fn = getattr(encoded, "to", None)
        if callable(to_fn):
            encoded = to_fn(self.device)

        with self._torch.no_grad():
            output = self._model(**encoded) if isinstance(encoded, dict) else self._model(encoded)

        logits = getattr(output, "logits", None)
        if logits is None:
            # Model returned a tuple-like object.
            logits = output[0]
        probs = self._torch.softmax(logits, dim=-1)
        # ``.cpu().numpy()`` on real torch; stubs may shortcut ``.numpy()``.
        cpu_fn = getattr(probs, "cpu", None)
        probs_cpu = cpu_fn() if callable(cpu_fn) else probs
        numpy_fn = getattr(probs_cpu, "numpy", None)
        arr = numpy_fn() if callable(numpy_fn) else np.asarray(probs_cpu)
        return np.asarray(arr, dtype=float)

    def _row_to_vader_dict(self, probs: np.ndarray) -> dict[str, float]:
        idx = self._label_to_idx
        pos = float(probs[idx["positive"]])
        neg = float(probs[idx["negative"]])
        neu = float(probs[idx["neutral"]]) if "neutral" in idx else float(1.0 - pos - neg)
        # Tiny negative values can show up from floating-point rounding.
        neu = max(0.0, neu)
        return {
            "compound": pos - neg,
            "pos": pos,
            "neg": neg,
            "neu": neu,
        }


def get_scorer(name: str = "vader", **kwargs: Any) -> Any:
    """Factory: return a Scorer-compatible object by name.

    ``vader`` defers to the existing VADER constructor in
    :mod:`sentinel.features.sentiment`. ``finbert`` constructs a
    :class:`FinBertScorer` (and lazy-imports torch + transformers).
    """
    name = name.lower()
    if name == "vader":
        # Local import avoids a circular dependency at module load.
        from sentinel.features.sentiment import _vader_scorer

        return _vader_scorer()
    if name == "finbert":
        return FinBertScorer(**kwargs)
    raise ValueError(f"Unknown scorer {name!r}. Use 'vader' or 'finbert'.")


__all__ = ["FinBertScorer", "get_scorer", "DEFAULT_MODEL"]
