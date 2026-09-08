from evaluation.sql_metrics import (
    extract_sql, normalize_sql, exact_match, execution_scores, evaluate_sql,
)


class TestExtractSql:
    def test_strips_markdown_fence(self):
        assert extract_sql("```sql\nSELECT a FROM t\n```") == "SELECT a FROM t"

    def test_cuts_at_next_section_header(self):
        assert extract_sql("SELECT a FROM t\n### Question: next") == "SELECT a FROM t"

    def test_takes_first_statement(self):
        assert extract_sql("SELECT a FROM t; DROP TABLE t") == "SELECT a FROM t"

    def test_collapses_whitespace(self):
        assert extract_sql("SELECT   a ,  b   FROM  t") == "SELECT a , b FROM t"


class TestNormalizeSql:
    def test_case_and_whitespace_insensitive(self):
        assert normalize_sql("SELECT  A  FROM T ;") == normalize_sql("select a from t")

    def test_operator_spacing(self):
        assert normalize_sql("age > 56") == normalize_sql("age>56")


class TestExactMatch:
    def test_all_match(self):
        assert exact_match(["SELECT a FROM t"], ["select a from t"]) == 1.0

    def test_none_match(self):
        assert exact_match(["SELECT a FROM t"], ["SELECT b FROM t"]) == 0.0

    def test_empty_is_zero_not_crash(self):
        assert exact_match([], []) == 0.0


class TestExecutionScores:
    schema = "CREATE TABLE head (name VARCHAR, age INTEGER)"

    def test_identical_query_matches(self):
        s = execution_scores(["SELECT name FROM head"], ["SELECT name FROM head"], [self.schema])
        assert s == {"execution_match": 1.0, "valid_sql_rate": 1.0}

    def test_hallucinated_column_is_invalid(self):
        s = execution_scores(["SELECT bogus FROM head"], ["SELECT name FROM head"], [self.schema])
        assert s["valid_sql_rate"] == 0.0
        assert s["execution_match"] == 0.0

    def test_equivalent_but_reordered_columns_still_match_on_empty_db(self):
        s = execution_scores(["SELECT age, name FROM head"], ["SELECT age, name FROM head"], [self.schema])
        assert s["execution_match"] == 1.0


def test_evaluate_sql_end_to_end():
    r = evaluate_sql(
        raw_predictions=["```sql\nSELECT COUNT(*) FROM head WHERE age > 56\n```"],
        references=["SELECT COUNT(*) FROM head WHERE age > 56"],
        schemas=["CREATE TABLE head (age INTEGER)"],
    )
    assert r["n"] == 1
    assert r["exact_match"] == 1.0
    assert r["execution_match"] == 1.0
    assert r["valid_sql_rate"] == 1.0
