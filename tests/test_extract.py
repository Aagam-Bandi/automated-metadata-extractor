import pytest

from src.extract import SUPPORTED, OCR_FALLBACK_THRESHOLD, extract


class TestExtract:
    def test_rejects_unsupported_extension(self, tmp_path):
        path = tmp_path / "data.xlsx"
        path.write_text("x")
        with pytest.raises(ValueError, match="unsupported file type"):
            extract(str(path))

    def test_reads_plain_text(self, tmp_path):
        path = tmp_path / "doc.txt"
        path.write_text("The quick brown fox jumps over the lazy dog.")
        result = extract(str(path))
        assert result.word_count == 9
        assert "quick brown fox" in result.documents[0].page_content

    def test_word_count_ignores_whitespace_runs(self, tmp_path):
        path = tmp_path / "doc.txt"
        path.write_text("one   two \n\n three\t\tfour")
        assert extract(str(path)).word_count == 4

    def test_supported_set_is_what_the_readme_claims(self):
        assert SUPPORTED == {".pdf", ".docx", ".doc", ".txt"}

    def test_threshold_is_positive(self):
        assert OCR_FALLBACK_THRESHOLD > 0
