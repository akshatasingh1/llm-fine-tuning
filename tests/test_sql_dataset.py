from data.sql_dataset import build_prompt, _to_example, PROMPT_TEMPLATE


def test_build_prompt_contains_schema_and_question_and_ends_open():
    p = build_prompt("How many rows?", "CREATE TABLE t (id INT)")
    assert "CREATE TABLE t (id INT)" in p
    assert "How many rows?" in p
    assert p.rstrip().endswith("### SQL:")  # prompt stops right before the answer


def test_build_prompt_strips_whitespace():
    p = build_prompt("  q  ", "  schema  ")
    assert "  q  " not in p and "q" in p


def test_to_example_shape():
    row = {"question": "q?", "context": "CREATE TABLE t (a INT)", "answer": "SELECT a FROM t  "}
    ex = _to_example(row)
    assert set(ex) == {"question", "schema", "answer", "prompt", "completion"}
    assert ex["completion"] == "SELECT a FROM t"      # stripped
    assert ex["schema"] == "CREATE TABLE t (a INT)"
    assert ex["prompt"] == build_prompt("q?", "CREATE TABLE t (a INT)")


def test_prompt_template_has_both_placeholders():
    assert "{schema}" in PROMPT_TEMPLATE and "{question}" in PROMPT_TEMPLATE
