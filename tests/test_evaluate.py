"""Tests for the evaluation harness. No API key required — the pipeline is stubbed."""

import asyncio
import json

import pytest

from src.evaluate import (
    EvalReport,
    evaluate,
    score_document,
    score_exact,
    score_jaccard,
)


class TestExactScoring:
    def test_identical(self):
        assert score_exact("Coordination Compounds", "Coordination Compounds") == 1.0

    def test_case_and_whitespace_insensitive(self):
        assert score_exact("  COORDINATION   compounds ", "Coordination Compounds") == 1.0

    def test_containment_gets_partial_credit(self):
        # A returned title with a chapter suffix is not wrong in a way anyone cares about.
        assert score_exact("Coordination Compounds — Chapter 9", "Coordination Compounds") == 0.8

    def test_unrelated(self):
        assert score_exact("Thermodynamics", "Coordination Compounds") == 0.0

    def test_correctly_predicted_null(self):
        assert score_exact(None, None) == 1.0

    def test_hallucinated_value_where_none_expected(self):
        assert score_exact("A. Sharma", None) == 0.0

    def test_missing_value_where_one_expected(self):
        assert score_exact(None, "A. Sharma") == 0.0


class TestJaccardScoring:
    def test_identical_sets(self):
        assert score_jaccard(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_disjoint_sets(self):
        assert score_jaccard(["a", "b"], ["c", "d"]) == 0.0

    def test_partial_overlap(self):
        assert score_jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)

    def test_order_independent(self):
        assert score_jaccard(["b", "a"], ["a", "b"]) == 1.0

    def test_case_insensitive(self):
        assert score_jaccard(["Ligands"], ["ligands"]) == 1.0

    def test_extra_predictions_are_penalised(self):
        # Returning twenty keywords to guarantee hitting the six real ones
        # should not score as well as returning six correct ones.
        precise = score_jaccard(["a", "b"], ["a", "b"])
        padded = score_jaccard(["a", "b", "x", "y", "z"], ["a", "b"])
        assert padded < precise


class TestDocumentScoring:
    def test_perfect_document_scores_one(self):
        expected = {"title": "T", "document_type": "report", "keywords": ["x", "y"]}
        result = score_document("doc.pdf", dict(expected), expected)
        assert result.mean_score == 1.0

    def test_fields_absent_from_ground_truth_are_skipped(self):
        result = score_document("doc.pdf", {"title": "T", "author": "A"}, {"title": "T"})
        assert [f.field_name for f in result.fields] == ["title"]

    def test_partial_failure_lowers_mean(self):
        expected = {"title": "T", "document_type": "report"}
        result = score_document("doc.pdf", {"title": "T", "document_type": "memo"}, expected)
        assert 0 < result.mean_score < 1


class TestEvalReport:
    def _stub(self, tmp_path, returns):
        gt = [{"document": "a.pdf", "expected": {"title": "Alpha"}},
              {"document": "b.pdf", "expected": {"title": "Beta"}}]
        path = tmp_path / "gt.json"
        path.write_text(json.dumps(gt))

        async def generate(doc):
            value = returns[doc]
            if isinstance(value, Exception):
                raise value
            return value

        return path, generate

    def test_overall_and_by_field(self, tmp_path):
        path, generate = self._stub(tmp_path, {
            "a.pdf": {"title": "Alpha"},
            "b.pdf": {"title": "Beta"},
        })
        report = asyncio.run(evaluate(path, generate, use_semantic=False))
        assert report.overall == 1.0
        assert report.by_field()["title"] == 1.0

    def test_pipeline_errors_are_recorded_not_raised(self, tmp_path):
        path, generate = self._stub(tmp_path, {
            "a.pdf": {"title": "Alpha"},
            "b.pdf": RuntimeError("quota exceeded"),
        })
        report = asyncio.run(evaluate(path, generate, use_semantic=False))
        errored = [d for d in report.documents if d.error]
        assert len(errored) == 1
        assert "quota exceeded" in errored[0].error

    def test_errors_excluded_from_overall(self, tmp_path):
        # A crashed document must not be scored as zero — that conflates
        # "the pipeline broke" with "the pipeline was wrong".
        path, generate = self._stub(tmp_path, {
            "a.pdf": {"title": "Alpha"},
            "b.pdf": RuntimeError("boom"),
        })
        report = asyncio.run(evaluate(path, generate, use_semantic=False))
        assert report.overall == 1.0

    def test_failures_lists_low_scoring_fields(self, tmp_path):
        path, generate = self._stub(tmp_path, {
            "a.pdf": {"title": "Wrong"},
            "b.pdf": {"title": "Beta"},
        })
        report = asyncio.run(evaluate(path, generate, use_semantic=False))
        assert len(report.failures()) == 1

    def test_empty_report_does_not_divide_by_zero(self):
        assert EvalReport(documents=[]).overall == 0.0
