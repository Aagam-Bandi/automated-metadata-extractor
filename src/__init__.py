"""Automated metadata extraction.

Imports are lazy: the evaluation harness and scoring functions must be usable
without the Gemini SDK or a network connection installed, so `src.evaluate`
does not drag in `src.pipeline` at import time.
"""

__all__ = [
    "generate_metadata", "generate_metadata_sync", "Metadata",
    "extract", "ExtractionResult",
    "EvalReport", "DocumentScore", "FieldScore",
    "evaluate", "evaluate_sync", "score_document", "score_exact", "score_jaccard",
]

_LAZY = {
    "generate_metadata": "pipeline",
    "generate_metadata_sync": "pipeline",
    "Metadata": "pipeline",
    "extract": "extract",
    "ExtractionResult": "extract",
    "EvalReport": "evaluate",
    "DocumentScore": "evaluate",
    "FieldScore": "evaluate",
    "evaluate": "evaluate",
    "evaluate_sync": "evaluate",
    "score_document": "evaluate",
    "score_exact": "evaluate",
    "score_jaccard": "evaluate",
}


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    module = importlib.import_module(f".{_LAZY[name]}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
