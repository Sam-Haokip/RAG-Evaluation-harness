import json
import re

with open("eval_set.json") as f:
    EVAL_QUESTIONS = json.load(f)


def test_eval_set_has_expected_count():
    assert len(EVAL_QUESTIONS) == 36, f"expected 36 questions, found {len(EVAL_QUESTIONS)}"


def test_every_question_has_required_fields():
    required = {"id", "type", "question", "answer", "source"}
    for q in EVAL_QUESTIONS:
        missing = required - q.keys()
        assert not missing, f"{q.get('id', '?')} missing fields: {missing}"


def test_ids_are_unique():
    ids = [q["id"] for q in EVAL_QUESTIONS]
    assert len(ids) == len(set(ids)), "duplicate question IDs found"


def test_types_are_valid_for_answerable_questions():
    valid_types = {"factual", "multi_hop"}
    for q in EVAL_QUESTIONS:
        if "N/A" not in q["source"]:
            assert q["type"] in valid_types, f"{q['id']} has unexpected type: {q['type']}"


def test_unanswerable_questions_are_marked_correctly():
    unanswerable = [q for q in EVAL_QUESTIONS if "N/A" in q["source"]]
    assert len(unanswerable) == 6, f"expected 6 unanswerable questions, found {len(unanswerable)}"


def test_multi_hop_sources_name_multiple_files():
    multi_hop = [q for q in EVAL_QUESTIONS if q.get("type") == "multi_hop"]
    assert len(multi_hop) > 0, "expected at least one multi_hop question"
    for q in multi_hop:
        files = re.findall(r'\b\w+\.py\b', q["source"])
        assert len(files) >= 2, f"{q['id']} is tagged multi_hop but source names fewer than 2 files"


def test_answerable_questions_name_at_least_one_file():
    for q in EVAL_QUESTIONS:
        if "N/A" not in q["source"]:
            files = re.findall(r'\b\w+\.py\b', q["source"])
            assert len(files) >= 1, f"{q['id']} is answerable but source names no .py file"


def test_no_empty_questions_or_answers():
    for q in EVAL_QUESTIONS:
        assert q["question"].strip(), f"{q['id']} has an empty question"
        assert q["answer"].strip(), f"{q['id']} has an empty answer"
