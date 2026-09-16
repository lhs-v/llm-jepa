import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.prepare_jepa_data import convert, load_records, prepare_pairs


def record(key, rationale=None, decision=False):
    answer = {"is_proactive_action_required": decision}
    if decision:
        answer["intents"] = [{"intent_type": "urgent_alert", "intent": "Check immediately"}]
    if rationale is not None:
        answer["reasoning"] = rationale
    messages = [
        {"role": "system", "content": "With explanation" if rationale else "Only decision"},
        {"role": "user", "content": f"Situation {key}"},
        {"role": "assistant", "content": json.dumps(answer)},
    ]
    text = "<bos>" + "".join(f"<|turn>{'model' if m['role'] == 'assistant' else m['role']}\n{m['content']}<turn|>\n" for m in messages)
    return {"id": key, "messages": messages, "text": text, "language": "ko", "category": "home"}


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_id_join_keeps_each_order_and_own_context_without_answer_leakage():
    reason = {"a": record("a", "Evidence A"), "b": record("b", "Evidence B", True)}
    no_reason = {"b": record("b", decision=True), "a": record("a")}
    original = copy.deepcopy((reason, no_reason))
    left, right = prepare_pairs(reason, no_reason)
    assert list(left) == ["a", "b"]
    assert list(right) == ["b", "a"]
    assert right["b"]["jepa_rationale"] == "Evidence B"
    assert right["a"]["jepa_context"] == "Policy:\nOnly decision\n\nSituation:\nSituation a"
    assert left["a"]["jepa_context"] == "Policy:\nWith explanation\n\nSituation:\nSituation a"
    for key, row in no_reason.items():
        assert {name: right[key][name] for name in row} == row
    assert (reason, no_reason) == original


@pytest.mark.parametrize("mutation", ["decision", "intents", "situation", "empty_rationale", "already_augmented", "wrong_boolean", "nr_has_reason"])
def test_bad_pairs_are_rejected(mutation):
    r, n = record("a", "Evidence", True), record("a", decision=True)
    answer = json.loads(n["messages"][2]["content"])
    if mutation == "decision":
        answer["is_proactive_action_required"] = False
    elif mutation == "intents":
        answer["intents"][0]["intent"] = "Different action"
    elif mutation == "situation":
        n["messages"][1]["content"] = "Different situation"
    elif mutation == "empty_rationale":
        r["messages"][2]["content"] = json.dumps({"is_proactive_action_required": True, "reasoning": " "})
    elif mutation == "already_augmented":
        n["jepa_context"] = "old"
    elif mutation == "wrong_boolean":
        answer["is_proactive_action_required"] = 1
    elif mutation == "nr_has_reason":
        answer["reasoning"] = "Unexpected explanation"
    n["messages"][2]["content"] = json.dumps(answer)
    with pytest.raises(ValueError):
        prepare_pairs({"a": r}, {"a": n})


def test_nonintervention_empty_intents_and_omission_can_match():
    r, n = record("a", "Evidence"), record("a")
    answer = json.loads(n["messages"][2]["content"])
    answer["intents"] = []
    n["messages"][2]["content"] = json.dumps(answer)
    left, right = prepare_pairs({"a": r}, {"a": n})
    assert right["a"]["jepa_rationale"] == "Evidence"


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_assistant_json_is_rejected(constant):
    r, n = record("a", "Evidence"), record("a")
    for row in (r, n):
        row["messages"][2]["content"] = (
            row["messages"][2]["content"][:-1] + ', "confidence": ' + constant + "}"
        )
    with pytest.raises(ValueError):
        prepare_pairs({"a": r}, {"a": n})


def test_nested_boolean_and_number_are_different_answers():
    r, n = record("a", "Evidence"), record("a")
    for row, value in [(r, True), (n, 1)]:
        answer = json.loads(row["messages"][2]["content"])
        answer["metadata"] = {"urgent": value}
        row["messages"][2]["content"] = json.dumps(answer)
    with pytest.raises(ValueError, match="non-reasoning answers differ"):
        prepare_pairs({"a": r}, {"a": n})


def test_missing_or_different_type_ids_are_not_paired():
    for right in [{}, {"1": record("1")}]:
        with pytest.raises(ValueError):
            prepare_pairs({1: record(1, "Evidence")}, right)


@pytest.mark.parametrize("ids", [["a", "a"], [None], [True], [""]])
def test_loading_rejects_duplicate_or_invalid_ids(tmp_path, ids):
    path = tmp_path / "rows.jsonl"
    write_rows(path, [record(key) for key in ids])
    with pytest.raises(ValueError):
        load_records(path)


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_records(path)


def test_conversion_preserves_sources_and_refuses_existing_output(tmp_path):
    r, n = tmp_path / "r.jsonl", tmp_path / "nr.jsonl"
    write_rows(r, [record("a", "Evidence")])
    write_rows(n, [record("a")])
    source = r.read_bytes(), n.read_bytes()
    output = tmp_path / "prepared"
    assert convert(r, n, output) == 1
    assert (output / "reason.jsonl").exists()
    assert (output / "no_reason.jsonl").exists()
    converted = json.loads((output / "no_reason.jsonl").read_text())
    assert converted["jepa_rationale"] == "Evidence"
    assert source == (r.read_bytes(), n.read_bytes())
    with pytest.raises(FileExistsError):
        convert(r, n, output)


def test_invalid_pair_creates_no_output(tmp_path):
    r, n = tmp_path / "r.jsonl", tmp_path / "nr.jsonl"
    write_rows(r, [record("a", "Evidence")])
    write_rows(n, [record("b")])
    output = tmp_path / "prepared"
    with pytest.raises(ValueError):
        convert(r, n, output)
    assert not output.exists()


def test_failed_second_output_leaves_no_partial_pair(tmp_path):
    r, n = tmp_path / "r.jsonl", tmp_path / "nr.jsonl"
    write_rows(r, [record("a", "Evidence")])
    invalid = record("a")
    invalid["category"] = float("nan")
    write_rows(n, [invalid])
    output = tmp_path / "prepared"
    with pytest.raises(ValueError):
        convert(r, n, output)
    assert not output.exists()


def test_module_cli_preserves_both_variants_and_pairs_reordered_unicode_data(tmp_path):
    r, n = tmp_path / "reason_input.jsonl", tmp_path / "no_reason_input.jsonl"
    reason = [record("첫 번째", "정상 출입입니다."), record(2, "이상 신호입니다.", True)]
    no_reason = [record(2, decision=True), record("첫 번째")]
    write_rows(r, reason)
    write_rows(n, no_reason)
    output = tmp_path / "prepared"
    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.prepare_jepa_data",
            "--reason", str(r), "--no-reason", str(n),
            "--output-dir", str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert b"Prepared 2 examples per file" in result.stdout
    rationales = {"첫 번째": "정상 출입입니다.", 2: "이상 신호입니다."}
    for filename, original in [("reason.jsonl", reason), ("no_reason.jsonl", no_reason)]:
        rows = [
            json.loads(line)
            for line in (output / filename).read_text(encoding="utf-8").splitlines()
        ]
        assert [row["id"] for row in rows] == [row["id"] for row in original]
        for actual, source in zip(rows, original):
            assert {key: actual[key] for key in source} == source
            assert actual["jepa_rationale"] == rationales[source["id"]]
            assert set(actual) == set(source) | {"jepa_context", "jepa_rationale"}
