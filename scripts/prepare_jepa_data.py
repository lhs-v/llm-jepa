"""Add independent JEPA views to paired reason/no-reason SFT JSONL files."""

import argparse
import json
from pathlib import Path

from src.data.jepa_data import parse_jepa_record


def load_records(path):
    """Load rows in file order, rejecting duplicate or invalid IDs."""
    records = {}
    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("id")
            if type(key) not in (str, int) or (isinstance(key, str) and not key.strip()):
                raise ValueError(f"{path}:{line_number}: invalid id")
            if key in records:
                raise ValueError(f"{path}:{line_number}: duplicate id {key!r}")
            records[key] = row
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def prepare_pairs(reason, no_reason):
    """Validate paired targets and add views without changing original fields."""
    if not reason or reason.keys() != no_reason.keys():
        raise ValueError("The two files must contain the same nonempty ID set.")

    rationales = {}
    parsed = ({}, {})
    for key in reason:
        r = parse_jepa_record(reason[key])
        n = parse_jepa_record(no_reason[key])
        if r["user"].strip() != n["user"].strip():
            raise ValueError(f"id={key!r}: user situations differ")
        rationale = r["answer"].get("reasoning")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"id={key!r}: missing or empty reasoning")
        if n["answer"].get("reasoning") not in (None, ""):
            raise ValueError(f"id={key!r}: no_reason answer contains reasoning")

        targets = []
        for item in (r, n):
            target = dict(item["answer"])
            target.pop("reasoning", None)
            decision = target.get("is_proactive_action_required")
            if type(decision) is not bool:
                raise ValueError(f"id={key!r}: decision must be a boolean")
            if not decision:
                target.setdefault("intents", [])
            # Compare JSON values without treating True and 1 as equal.
            targets.append(json.dumps(target, sort_keys=True, allow_nan=False))
        if targets[0] != targets[1]:
            raise ValueError(f"id={key!r}: non-reasoning answers differ")
        rationales[key] = rationale
        parsed[0][key], parsed[1][key] = r, n

    outputs = []
    for records, views in zip((reason, no_reason), parsed):
        output = {}
        for key, row in records.items():
            if "jepa_context" in row or "jepa_rationale" in row:
                raise ValueError(f"id={key!r}: JEPA fields already exist")
            view = views[key]
            output[key] = {
                **row,
                "jepa_context": f"Policy:\n{view['system']}\n\nSituation:\n{view['user']}",
                "jepa_rationale": rationales[key],
            }
        outputs.append(output)
    return outputs


def convert(reason_file, no_reason_file, output_dir):
    """Write both validated variants into a new output directory."""
    outputs = prepare_pairs(load_records(reason_file), load_records(no_reason_file))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = [output_dir / "reason.jsonl", output_dir / "no_reason.jsonl"]
    try:
        for path, rows in zip(paths, outputs):
            with path.open("x", encoding="utf-8") as file:
                for row in rows.values():
                    file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    except Exception:
        for path in paths:
            path.unlink(missing_ok=True)
        output_dir.rmdir()
        raise
    return len(outputs[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reason", required=True)
    parser.add_argument("--no-reason", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    count = convert(args.reason, args.no_reason, args.output_dir)
    print(f"Prepared {count} examples per file in {args.output_dir}")
