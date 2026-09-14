"""Evaluation harness: score extracted metadata against labelled ground truth.

Without this, "the output looks good" is the only quality signal, which means
prompt changes get accepted or rejected on vibes. This turns that into a number
that moves when the pipeline gets better or worse.

Three metric families, because metadata fields fail differently:

  exact       title, author, document_type — either right or not
  set overlap keywords — partial credit via Jaccard
  semantic    summary — cosine similarity of embeddings, since two correct
              summaries can share almost no vocabulary
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@dataclass
class FieldScore:
    field_name: str
    predicted: str | list | None
    expected: str | list | None
    score: float
    method: str


@dataclass
class DocumentScore:
    document: str
    fields: list[FieldScore] = field(default_factory=list)
    error: str | None = None

    @property
    def mean_score(self) -> float:
        if not self.fields:
            return 0.0
        return float(np.mean([f.score for f in self.fields]))


@dataclass
class EvalReport:
    documents: list[DocumentScore]

    @property
    def overall(self) -> float:
        scored = [d for d in self.documents if not d.error]
        return float(np.mean([d.mean_score for d in scored])) if scored else 0.0

    def by_field(self) -> dict[str, float]:
        """Per-field means. This is the useful view — an overall number hides
        that title extraction is fine and author extraction is broken."""
        buckets: dict[str, list[float]] = {}
        for doc in self.documents:
            for f in doc.fields:
                buckets.setdefault(f.field_name, []).append(f.score)
        return {k: float(np.mean(v)) for k, v in sorted(buckets.items())}

    def failures(self, threshold: float = 0.5) -> list[FieldScore]:
        return [
            f for doc in self.documents for f in doc.fields if f.score < threshold
        ]

    def __str__(self) -> str:
        lines = [
            f"Documents evaluated  {len(self.documents)}",
            f"Overall score        {self.overall:.3f}",
            "",
            "By field:",
        ]
        for name, score in self.by_field().items():
            bar = "█" * int(score * 20)
            lines.append(f"  {name:<16} {score:.3f}  {bar}")
        errors = [d for d in self.documents if d.error]
        if errors:
            lines += ["", f"Errors ({len(errors)}):"]
            lines += [f"  {d.document}: {d.error}" for d in errors]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "overall": round(self.overall, 4),
            "by_field": {k: round(v, 4) for k, v in self.by_field().items()},
            "documents": [asdict(d) for d in self.documents],
        }


def _normalise(text: str) -> str:
    return " ".join(str(text).lower().split())


def score_exact(predicted, expected) -> float:
    """Case- and whitespace-insensitive match, with credit for containment.

    Containment matters: a model returning "Coordination Compounds — Chapter 9"
    against an expected "Coordination Compounds" is not wrong in a way anyone
    cares about.
    """
    if expected is None:
        return 1.0 if predicted is None else 0.0
    if predicted is None:
        return 0.0
    p, e = _normalise(predicted), _normalise(expected)
    if p == e:
        return 1.0
    if e in p or p in e:
        return 0.8
    return 0.0


def score_jaccard(predicted, expected) -> float:
    """Set overlap for keyword lists."""
    if not expected:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    p = {_normalise(x) for x in predicted}
    e = {_normalise(x) for x in expected}
    return len(p & e) / len(p | e)


class SemanticScorer:
    """Cosine similarity between embeddings. Loaded lazily so the harness can
    run exact-match-only evaluations without pulling a model."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("loading %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def __call__(self, predicted, expected) -> float:
        if expected is None:
            return 1.0 if predicted is None else 0.0
        if predicted is None:
            return 0.0
        model = self._load()
        vecs = model.encode([str(predicted), str(expected)])
        sim = float(
            np.dot(vecs[0], vecs[1])
            / (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1]) + 1e-9)
        )
        return max(0.0, min(1.0, sim))


# Which scorer applies to which field.
RUBRIC = {
    "title": ("exact", score_exact),
    "author": ("exact", score_exact),
    "document_type": ("exact", score_exact),
    "subject_domain": ("exact", score_exact),
    "keywords": ("jaccard", score_jaccard),
}


def load_ground_truth(path: str | Path) -> list[dict]:
    """Load labelled cases. Format: a JSON list of
    {"document": "examples/x.pdf", "expected": {...}}."""
    with open(path) as fh:
        return json.load(fh)


def score_document(document: str, predicted: dict, expected: dict,
                   semantic: SemanticScorer | None = None) -> DocumentScore:
    result = DocumentScore(document=document)

    for field_name, (method, scorer) in RUBRIC.items():
        if field_name not in expected:
            continue
        result.fields.append(FieldScore(
            field_name=field_name,
            predicted=predicted.get(field_name),
            expected=expected[field_name],
            score=scorer(predicted.get(field_name), expected[field_name]),
            method=method,
        ))

    if "summary" in expected and semantic is not None:
        result.fields.append(FieldScore(
            field_name="summary",
            predicted=predicted.get("summary"),
            expected=expected["summary"],
            score=semantic(predicted.get("summary"), expected["summary"]),
            method="semantic",
        ))

    return result


async def evaluate(
    ground_truth_path: str | Path,
    generate_fn=None,
    use_semantic: bool = True,
) -> EvalReport:
    """Run the pipeline over every labelled document and score the output.

    `generate_fn` is injectable so the harness can be tested against a stub
    without spending API calls.
    """
    if generate_fn is None:
        from .pipeline import generate_metadata

        async def generate_fn(path):  # noqa: E306
            return (await generate_metadata(path)).to_dict()

    cases = load_ground_truth(ground_truth_path)
    semantic = SemanticScorer() if use_semantic else None
    scores: list[DocumentScore] = []

    for case in cases:
        doc = case["document"]
        logger.info("evaluating %s", doc)
        try:
            predicted = await generate_fn(doc)
        except Exception as exc:
            logger.error("%s failed: %s", doc, exc)
            scores.append(DocumentScore(document=doc, error=f"{type(exc).__name__}: {exc}"))
            continue
        scores.append(score_document(doc, predicted, case["expected"], semantic))

    return EvalReport(documents=scores)


def evaluate_sync(ground_truth_path, generate_fn=None, use_semantic: bool = True) -> EvalReport:
    return asyncio.run(evaluate(ground_truth_path, generate_fn, use_semantic))
