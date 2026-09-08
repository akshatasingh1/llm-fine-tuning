import pytest

from evaluation.llm_judge import _parse, _load_dotenv, _VERDICT_SCORE


class TestParse:
    def test_plain_json(self):
        assert _parse('{"verdict": "correct", "reason": "ok"}')["verdict"] == "correct"

    def test_json_with_surrounding_text(self):
        assert _parse('here you go: {"verdict": "partial"} done')["verdict"] == "partial"

    def test_verdict_is_lowercased(self):
        assert _parse('{"verdict": "WRONG"}')["verdict"] == "wrong"

    def test_no_json_is_error_not_wrong(self):
        assert _parse("the model refused")["verdict"] == "error"

    def test_invalid_json_is_error(self):
        assert _parse("{verdict: correct}")["verdict"] == "error"

    def test_unknown_verdict_is_error(self):
        assert _parse('{"verdict": "maybe"}')["verdict"] == "error"

    def test_empty_is_error(self):
        assert _parse("")["verdict"] == "error"


def test_verdict_score_table():
    assert _VERDICT_SCORE == {"correct": 1.0, "partial": 0.5, "wrong": 0.0}


class TestLoadDotenv:
    def test_missing_file_is_noop(self, tmp_path):
        _load_dotenv(str(tmp_path / "nope.env"))  # must not raise

    def test_loads_keys_without_overriding_existing(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("FOO_JUDGE_TEST=from_file\nBAR_JUDGE_TEST='quoted'\n# comment\n")
        monkeypatch.setenv("FOO_JUDGE_TEST", "from_env")
        monkeypatch.delenv("BAR_JUDGE_TEST", raising=False)
        _load_dotenv(str(env))
        import os
        assert os.environ["FOO_JUDGE_TEST"] == "from_env"   # existing wins
        assert os.environ["BAR_JUDGE_TEST"] == "quoted"     # quotes stripped
