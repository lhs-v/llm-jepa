"""Deterministic JSON evaluation with explicit invalid-prediction denominators."""

from __future__ import annotations

import json
import math
import time
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


def _ratio(numerator: int | float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"JSON number exceeds the supported finite range: {value}")
    return number


def _structural_equal(left: Any, right: Any) -> bool:
    # Python's True == 1 must not make a numeric decision match a Boolean target.
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_structural_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_structural_equal(a, b) for a, b in zip(left, right))
    return left == right


def _decision_value(value: Any, field: str) -> bool | None:
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value if type(value) is bool else None


def _input_device(model: Any) -> torch.device:
    # The embedding device also works when a model was dispatched across devices.
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None and embeddings.weight.device.type != "meta":
            return embeddings.weight.device
    device = getattr(model, "device", None)
    if device is not None and torch.device(device).type != "meta":
        return torch.device(device)
    return next(model.parameters()).device


def _make_validator(target_schema: Mapping[str, Any] | bool | str | Path | None) -> Any:
    from jsonschema.validators import validator_for

    if isinstance(target_schema, (str, Path)):
        with Path(target_schema).open(encoding="utf-8-sig") as handle:
            target_schema = json.load(handle)
    schema = {"type": "object"} if target_schema is None else target_schema
    validator_class = validator_for(schema)
    validator_class.check_schema(schema)
    return validator_class(schema)


def evaluate_examples(
    model: Any,
    tokenizer: Any,
    formatter: Any,
    examples: Iterable[Any],
    eval_config: Mapping[str, Any],
    target_schema: Mapping[str, Any] | bool | str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate and score each example without using its reference rationale.

    JSON/schema/exact-match fractions use all evaluated rows. Boolean confusion
    and its rates use rows with both a Boolean reference and a Boolean prediction;
    strict_accuracy additionally includes invalid predictions in its denominator.
    Undefined rates are null, rather than silently reported as zero. Model/device
    errors propagate; malformed model answers are recorded and evaluation continues.
    """
    thinking = eval_config.get("thinking", "off")
    if thinking not in {"off", "on"}:
        raise ValueError("evaluation.thinking must be 'off' or 'on'")
    max_new_tokens = eval_config.get("max_new_tokens", 256)
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("evaluation.max_new_tokens must be a positive integer")
    max_samples = eval_config.get("max_samples")
    if max_samples is not None and (type(max_samples) is not int or max_samples <= 0):
        raise ValueError("evaluation.max_samples must be a positive integer or null")
    max_length = eval_config.get("max_length")
    if max_length is not None and (type(max_length) is not int or max_length <= 0):
        raise ValueError("max_length must be a positive integer or null")
    overlength = eval_config.get("overlength", "error")
    if overlength not in {"error", "skip"}:
        raise ValueError("overlength must be 'error' or 'skip'")
    field = eval_config.get("decision_field", "intervene")
    if not isinstance(field, str) or not field or any(not part for part in field.split(".")):
        raise ValueError("evaluation.decision_field must be a dotted field path")
    validator = _make_validator(target_schema)
    model.eval()
    device = _input_device(model)
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    started = time.perf_counter()

    for example in islice(examples, max_samples):
        prompt = formatter.render_prompt(example, thinking=thinking == "on", rationale_task=False)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_tokens = inputs["input_ids"].shape[-1]
        if max_length is not None and input_tokens > max_length:
            if overlength == "error":
                raise ValueError(f"Example {example.id!r}: prompt has {input_tokens} tokens, exceeding max_length={max_length}")
            skipped.append({"id": example.id, "input_tokens": input_tokens, "max_length": max_length})
            continue
        inputs = {name: value.to(device) for name, value in inputs.items()}
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens, "do_sample": False, "use_cache": True,
        }
        if tokenizer.pad_token_id is not None:
            generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
        # Hugging Face's stop_strings processor needs the tokenizer. Retain the
        # loaded generation config, including every EOS ID and stopping option.
        generation_config = getattr(model, "generation_config", None)
        if getattr(generation_config, "stop_strings", None):
            generate_kwargs["tokenizer"] = tokenizer
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        generation_started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, **generate_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency = time.perf_counter() - generation_started
        sequences = output.sequences if hasattr(output, "sequences") else output
        new_ids = sequences[0, input_tokens:]
        raw = tokenizer.decode(new_ids, skip_special_tokens=False)
        formatted = formatter.parse_generation(raw)
        answer = formatted["answer"]
        parsed = None
        json_error = None
        try:
            if not isinstance(answer, str):
                raise ValueError("Generation formatter did not return a text answer")
            parsed = json.loads(answer, parse_constant=_reject_constant, parse_float=_finite_float)
        except (ValueError, RecursionError) as error:
            json_error = str(error)
        json_valid = json_error is None
        schema_error = None
        if json_valid:
            error = next(validator.iter_errors(parsed), None)
            if error is not None:
                schema_error = error.message
        else:
            schema_error = "Answer is not valid JSON"
        format_error = formatted.get("format_error")
        predicted_decision = _decision_value(parsed, field) if json_valid and not format_error else None
        reference_decision = _decision_value(example.target, field)
        results.append({
            "id": example.id,
            "reference": example.target,
            "generation_raw": raw,
            "answer": answer,
            "parsed_answer": parsed,
            "rationale": formatted.get("rationale"),
            "thought_generated": bool(formatted.get("thought_generated", False)),
            "format_valid": format_error is None,
            "format_error": format_error,
            "json_valid": json_valid,
            "json_error": json_error,
            "schema_valid": json_valid and schema_error is None,
            "schema_error": schema_error,
            "structural_exact_match": json_valid and not bool(format_error) and _structural_equal(parsed, example.target),
            "decision": predicted_decision,
            "decision_valid": predicted_decision is not None,
            "reference_decision": reference_decision,
            "reference_decision_valid": reference_decision is not None,
            "input_tokens": input_tokens,
            "output_tokens": len(new_ids),
            "latency_seconds": latency,
        })

    count = len(results)
    eligible = [row for row in results if row["reference_decision_valid"]]
    valid = [row for row in eligible if row["decision_valid"]]
    tp = sum(row["decision"] is True and row["reference_decision"] is True for row in valid)
    tn = sum(row["decision"] is False and row["reference_decision"] is False for row in valid)
    fp = sum(row["decision"] is True and row["reference_decision"] is False for row in valid)
    fn = sum(row["decision"] is False and row["reference_decision"] is True for row in valid)
    decision_metrics = {
        "field": field,
        "eligible_reference_count": len(eligible),
        "invalid_reference_count": count - len(eligible),
        "invalid_decision_count": sum(not row["decision_valid"] for row in results),
        "invalid_decision_with_valid_reference_count": len(eligible) - len(valid),
        "valid_pair_count": len(valid),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "correct_count": tp + tn,
        "accuracy": _ratio(tp + tn, len(valid)),
        "accuracy_denominator": len(valid),
        "strict_accuracy": _ratio(tp + tn, len(eligible)),
        "strict_accuracy_denominator": len(eligible),
        "prediction_coverage": _ratio(len(valid), len(eligible)),
        "precision": _ratio(tp, tp + fp),
        "precision_denominator": tp + fp,
        "recall": _ratio(tp, tp + fn),
        "recall_denominator": tp + fn,
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "f1_denominator": 2 * tp + fp + fn,
        "false_positive_rate": _ratio(fp, fp + tn),
        "false_positive_rate_denominator": fp + tn,
        "false_negative_rate": _ratio(fn, fn + tp),
        "false_negative_rate_denominator": fn + tp,
    }
    metrics = {
        "examples": count,
        "considered_examples": count + len(skipped),
        "skipped_overlength_count": len(skipped),
        "skipped_examples": skipped,
        "fraction_denominator": count,
        "json_valid_count": sum(row["json_valid"] for row in results),
        "schema_valid_count": sum(row["schema_valid"] for row in results),
        "format_valid_count": sum(row["format_valid"] for row in results),
        "structural_exact_match_count": sum(row["structural_exact_match"] for row in results),
        "generated_thought_count": sum(row["thought_generated"] for row in results),
        "decision_metrics": decision_metrics,
        "average_output_tokens": _ratio(sum(row["output_tokens"] for row in results), count),
        "average_latency_seconds": _ratio(sum(row["latency_seconds"] for row in results), count),
        "elapsed_seconds": time.perf_counter() - started,
        "latency_scope": "model.generate only; CUDA synchronized",
        "prompt_mode": f"thinking_{thinking}",
    }
    for name in ("json_valid", "schema_valid", "format_valid", "generated_thought"):
        metrics[f"{name}_fraction"] = _ratio(metrics[f"{name}_count"], count)
    metrics["structural_exact_match"] = _ratio(metrics["structural_exact_match_count"], count)
    return results, metrics
