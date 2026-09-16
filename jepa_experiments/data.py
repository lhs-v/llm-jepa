"""Strict, reproducible loading of intervention JSONL datasets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class Example:
    id: str
    situation: str
    policy: str
    rationale: str | None
    target: dict

    @property
    def target_text(self) -> str:
        return json.dumps(self.target, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _reject_constant(value: str):
    raise ValueError(f"Non-finite JSON number is not allowed: {value}")


def _get(record: dict, path: str | None):
    if path is None:
        return None
    value = record
    for segment in path.split("."):
        if not isinstance(value, dict) or segment not in value:
            return None
        value = value[segment]
    return value


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _target(value) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value, parse_constant=_reject_constant)
        except ValueError as exc:
            raise ValueError(f"target must contain a valid JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("target must be a JSON object or a string containing a JSON object")
    try:
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"target must be a finite JSON object: {exc}") from exc
    return value


def _record_example(record: dict, config: dict, line: int, require_rationale: bool) -> Example:
    fields = config["fields"]
    identifier = _get(record, fields["id"])
    if identifier is None:
        identifier = str(line)
    identifier = _text(identifier, "id")
    if config["format"] == "messages":
        messages = record.get("messages")
        if not isinstance(messages, list) or not all(isinstance(message, dict) for message in messages):
            raise ValueError("messages must be an array of message objects")
        roles = [message.get("role") for message in messages]
        if roles not in (["user", "assistant"], ["system", "user", "assistant"]):
            raise ValueError("messages must contain a single-turn conversation: optional system, user, assistant")
        policy = messages[0].get("content") if roles[0] == "system" else config.get("default_policy")
        situation = messages[-2].get("content")
        assistant = messages[-1]
        rationale = assistant.get("reasoning")
        if rationale is None:
            rationale = assistant.get("reasoning_content")
        target = assistant.get("content")
    else:
        situation = _get(record, fields["situation"])
        policy = _get(record, fields["policy"])
        if policy is None:
            policy = config.get("default_policy")
        rationale = _get(record, fields["rationale"])
        target = _get(record, fields["target"])
    situation = _text(situation, "situation")
    policy = _text(policy, "policy")
    if rationale is not None or require_rationale:
        rationale = _text(rationale, "rationale")
    return Example(identifier, situation, policy, rationale, _target(target))


def _schema_validator(path: str | None):
    if path is None:
        return None, None
    try:
        import jsonschema
    except ImportError as exc:
        raise ValueError("data.target_schema requires jsonschema; install requirements-gemma4.txt") from exc
    schema_path = Path(path)
    try:
        content = schema_path.read_bytes()
        schema = json.loads(content.decode("utf-8-sig"), parse_constant=_reject_constant)
        validator_type = jsonschema.validators.validator_for(schema)
        validator_type.check_schema(schema)
        validator = validator_type(schema)
    except (OSError, ValueError, jsonschema.exceptions.SchemaError) as exc:
        raise ValueError(f"Invalid target schema {schema_path}: {exc}") from exc
    return validator, hashlib.sha256(content).hexdigest()


def load_examples(data_config: dict, require_rationale: bool = False, split: str = "train") -> tuple[list[Example], dict]:
    """Validate all rows, then select the first max_samples in file order.

    Token length is handled by the tokenizer layer. This loader never drops
    malformed rows and does not truncate situation, rationale, or JSON text.
    """
    if split not in ("train", "eval"):
        raise ValueError("split must be train or eval")
    filename = data_config.get(f"{split}_file")
    if not filename:
        raise ValueError(f"data.{split}_file is required")
    path = Path(filename)
    try:
        content = path.read_bytes()
        lines = content.decode("utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read dataset {path}: {exc}") from exc
    validator, schema_hash = _schema_validator(data_config.get("target_schema"))
    examples = []
    seen = set()
    total = 0
    limit = data_config.get("max_samples")
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        total += 1
        try:
            record = json.loads(line, parse_constant=_reject_constant)
            if not isinstance(record, dict):
                raise ValueError("record must be a JSON object")
            example = _record_example(record, data_config, line_number, require_rationale)
            if example.id in seen:
                raise ValueError(f"duplicate id {example.id!r}")
            seen.add(example.id)
            if validator is not None:
                error = next(validator.iter_errors(example.target), None)
                if error is not None:
                    location = ".".join(str(part) for part in error.absolute_path) or "<root>"
                    raise ValueError(f"target schema violation at {location}: {error.message}")
            if limit is None or len(examples) < limit:
                examples.append(example)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not total:
        raise ValueError(f"{path}: dataset contains no records")
    stats = {"file": str(path.resolve()), "file_hash": hashlib.sha256(content).hexdigest(), "total": total, "read": total, "selected": len(examples)}
    if schema_hash is not None:
        stats["schema_hash"] = schema_hash
    return examples, stats
