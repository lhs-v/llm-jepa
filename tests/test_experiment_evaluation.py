import json
from types import SimpleNamespace

import pytest
import torch


class TextTokenizer:
    pad_token_id = 0

    def __call__(self, text, *, return_tensors, add_special_tokens):
        assert return_tensors == "pt"
        assert add_special_tokens is False
        return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones((1, 2), dtype=torch.long)}

    def decode(self, token_ids, *, skip_special_tokens):
        # Stripping channel markers before parsing would mix rationale into JSON.
        assert skip_special_tokens is False
        return "".join(chr(value) for value in token_ids.tolist())


class CompletionModel:
    device = torch.device("cpu")

    def __init__(self, completions):
        self.completions = iter(completions)
        self.generation_config = SimpleNamespace(eos_token_id=[10, 11], stop_strings=["<stop>"])
        self.config = SimpleNamespace(use_cache=False)
        self.calls = []

    def eval(self):
        return self

    def generate(self, input_ids, attention_mask, **kwargs):
        self.calls.append(kwargs)
        assert attention_mask.tolist() == [[1, 1]]
        new_ids = torch.tensor([[ord(char) for char in next(self.completions)]], dtype=torch.long)
        return torch.cat((input_ids, new_ids), dim=1)


class PlainFormatter:
    def __init__(self):
        self.prompts = []

    def render_prompt(self, example, *, thinking, rationale_task):
        assert rationale_task is False
        self.prompts.append((example.situation, example.policy, thinking))
        return example.situation + "\n" + example.policy

    def parse_generation(self, raw):
        return {"answer": raw, "rationale": None, "thought_generated": False, "format_error": None}


def example(target, identifier="row"):
    return SimpleNamespace(
        id=identifier,
        situation="Observation only",
        policy="Use evidence",
        rationale="SECRET GOLD RATIONALE",
        target=target,
        target_text=json.dumps(target),
    )


def run_evaluation(completions, references, **kwargs):
    from jepa_experiments.evaluation import evaluate_examples

    model = CompletionModel(completions)
    formatter = PlainFormatter()
    rows, metrics = evaluate_examples(
        model,
        TextTokenizer(),
        formatter,
        [example(target, str(index)) for index, target in enumerate(references)],
        {"thinking": "off", "max_new_tokens": 80, "decision_field": "intervene"},
        **kwargs,
    )
    return rows, metrics, model, formatter


def test_invalid_decisions_are_excluded_from_boolean_confusion_and_counted():
    rows, metrics, _, _ = run_evaluation(
        ['{"intervene":true}', '{"intervene":false}', '{"intervene":true}',
         '{"intervene":false}', 'not json', '{"reason":"missing"}', '{"intervene":"true"}'],
        [{"intervene": value} for value in [True, False, False, True, True, False, True]],
    )
    assert len(rows) == 7
    assert metrics["json_valid_count"] == 6
    assert metrics["schema_valid_count"] == 6
    assert metrics["structural_exact_match_count"] == 2
    decision = metrics["decision_metrics"]
    assert {key: decision[key] for key in ("tp", "tn", "fp", "fn")} == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}
    assert decision["invalid_decision_count"] == 3
    assert decision["eligible_reference_count"] == 7
    assert decision["accuracy_denominator"] == 4
    assert decision["accuracy"] == 0.5
    assert decision["strict_accuracy"] == pytest.approx(2 / 7)
    for name in ("precision", "recall", "f1", "false_positive_rate", "false_negative_rate"):
        assert decision[name] == 0.5
    assert rows[4]["json_error"]
    assert rows[4]["parsed_answer"] is None
    assert rows[5]["decision_valid"] is False


def test_structural_match_ignores_object_order_but_preserves_boolean_type():
    rows, metrics, _, _ = run_evaluation(
        ['{"details":{"b":2,"a":1},"intervene":true}', '{"intervene":1}'],
        [{"intervene": True, "details": {"a": 1, "b": 2}}, {"intervene": True}],
    )
    assert [row["structural_exact_match"] for row in rows] == [True, False]
    assert metrics["structural_exact_match"] == 0.5
    assert metrics["decision_metrics"]["invalid_decision_count"] == 1


@pytest.mark.parametrize("completion", ['```json\n{"intervene":true}\n```', 'prefix {"intervene":true}', '{"intervene":true} trailing', '{"score":NaN}', '{"score":Infinity}', '{"intervene":true,"score":1e9999}', '{"intervene":true,"score":-1e9999}'])
def test_json_must_be_a_complete_standard_json_value(completion):
    rows, metrics, _, _ = run_evaluation([completion], [{"intervene": True}])
    assert rows[0]["json_valid"] is False
    assert rows[0]["schema_valid"] is False
    assert metrics["json_valid_count"] == 0
    assert metrics["decision_metrics"]["accuracy"] is None
    assert rows[0]["parsed_answer"] is None
    json.dumps(rows, allow_nan=False)  # Invalid answers must remain loggable.


def test_schema_validation_is_separate_from_json_syntax():
    schema = {
        "type": "object", "required": ["intervene"], "additionalProperties": False,
        "properties": {"intervene": {"type": "boolean"}},
    }
    rows, metrics, _, _ = run_evaluation(
        ['{"intervene":true}', '{"intervene":"true"}', '[true]'],
        [{"intervene": True}] * 3,
        target_schema=schema,
    )
    assert [row["json_valid"] for row in rows] == [True, True, True]
    assert [row["schema_valid"] for row in rows] == [True, False, False]
    assert rows[1]["schema_error"]
    assert metrics["schema_valid_count"] == 1


def test_schema_file_with_utf8_bom_matches_dataset_loader_behavior(tmp_path):
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({
        "type": "object", "required": ["intervene"],
        "properties": {"intervene": {"type": "boolean"}},
    }), encoding="utf-8-sig")
    rows, metrics, _, _ = run_evaluation(
        ['{"intervene":true}', '{"intervene":"true"}'],
        [{"intervene": True}, {"intervene": True}], target_schema=schema,
    )
    assert [row["schema_valid"] for row in rows] == [True, False]
    assert metrics["schema_valid_count"] == 1


def test_generation_preserves_model_stop_configuration_and_enables_cache():
    rows, metrics, model, formatter = run_evaluation(['{"intervene":true}'], [{"intervene": True}])
    kwargs = model.calls[0]
    assert kwargs["max_new_tokens"] == 80
    assert kwargs["do_sample"] is False
    assert kwargs["use_cache"] is True
    assert "eos_token_id" not in kwargs
    assert "stop_strings" not in kwargs
    assert kwargs["tokenizer"].pad_token_id == 0
    assert model.generation_config.eos_token_id == [10, 11]
    assert model.generation_config.stop_strings == ["<stop>"]
    assert formatter.prompts == [("Observation only", "Use evidence", False)]
    assert rows[0]["generation_raw"] == '{"intervene":true}'
    assert rows[0]["output_tokens"] == 18
    assert rows[0]["latency_seconds"] >= 0
    assert metrics["average_output_tokens"] == 18
    assert metrics["elapsed_seconds"] >= 0


def test_thought_channel_is_parsed_before_strict_json_and_is_reported():
    from jepa_experiments.evaluation import evaluate_examples

    class ChannelFormatter(PlainFormatter):
        def parse_generation(self, raw):
            assert raw == '<channel>analysis<sep>reason<channel>final<sep>{"intervene":true}<stop>'
            return {"answer": '{"intervene":true}', "rationale": "reason", "thought_generated": True, "format_error": None}

    formatter = ChannelFormatter()
    rows, metrics = evaluate_examples(
        CompletionModel(['<channel>analysis<sep>reason<channel>final<sep>{"intervene":true}<stop>']),
        TextTokenizer(), formatter, [example({"intervene": True})],
        {"thinking": "on", "max_new_tokens": 80, "decision_field": "intervene"},
    )
    assert formatter.prompts[0][-1] is True
    assert rows[0]["parsed_answer"] == {"intervene": True}
    assert rows[0]["rationale"] == "reason"
    assert rows[0]["thought_generated"] is True
    assert metrics["generated_thought_fraction"] == 1.0
    assert metrics["prompt_mode"] == "thinking_on"


def test_nested_decision_field_and_non_boolean_references_have_explicit_denominators():
    from jepa_experiments.evaluation import evaluate_examples

    rows, metrics = evaluate_examples(
        CompletionModel(['{"choice":{"act":true}}', '{"choice":{"act":false}}']),
        TextTokenizer(), PlainFormatter(),
        [example({"choice": {"act": True}}), example({"choice": {"act": "unknown"}})],
        {"thinking": "off", "max_new_tokens": 80, "decision_field": "choice.act"},
    )
    decision = metrics["decision_metrics"]
    assert decision["accuracy"] == 1.0
    assert decision["accuracy_denominator"] == 1
    assert decision["invalid_reference_count"] == 1
    assert rows[1]["reference_decision_valid"] is False


def test_max_samples_limits_generation_and_empty_input_reports_null_rates():
    from jepa_experiments.evaluation import evaluate_examples

    rows, metrics = evaluate_examples(
        CompletionModel(['{"intervene":true}']), TextTokenizer(), PlainFormatter(),
        [example({"intervene": True})] * 3,
        {"thinking": "off", "max_samples": 1, "max_new_tokens": 80},
    )
    assert len(rows) == metrics["examples"] == 1
    rows, metrics = evaluate_examples(CompletionModel([]), TextTokenizer(), PlainFormatter(), [], {"thinking": "off"})
    assert rows == []
    assert metrics["json_valid_fraction"] is None
    assert metrics["decision_metrics"]["accuracy_denominator"] == 0


def test_generation_runtime_failures_are_not_converted_to_invalid_predictions():
    from jepa_experiments.evaluation import evaluate_examples

    class FailedModel(CompletionModel):
        def generate(self, *args, **kwargs):
            raise RuntimeError("device failure")

    with pytest.raises(RuntimeError, match="device failure"):
        evaluate_examples(FailedModel([]), TextTokenizer(), PlainFormatter(), [example({"intervene": True})], {"thinking": "off"})


def test_overlength_prompts_are_never_silently_truncated():
    from jepa_experiments.evaluation import evaluate_examples

    with pytest.raises(ValueError, match="row.*max_length"):
        evaluate_examples(
            CompletionModel([]), TextTokenizer(), PlainFormatter(), [example({"intervene": True})],
            {"thinking": "off", "max_length": 1, "overlength": "error"},
        )
    rows, metrics = evaluate_examples(
        CompletionModel([]), TextTokenizer(), PlainFormatter(), [example({"intervene": True})],
        {"thinking": "off", "max_length": 1, "overlength": "skip"},
    )
    assert rows == []
    assert metrics["considered_examples"] == 1
    assert metrics["skipped_overlength_count"] == 1
    assert metrics["skipped_examples"] == [{"id": "row", "input_tokens": 2, "max_length": 1}]


def saved_run(tmp_path, *, revision="a" * 40, model_name="organization/model", source_hashes=None):
    run = tmp_path / "training-run"
    (run / "adapter").mkdir(parents=True)
    (run / "resolved_config.json").write_text(json.dumps({
        "model": {"model_name_or_path": model_name, "revision": "moving-tag"},
        "data": {"eval_file": "eval.jsonl"},
    }), encoding="utf-8")
    metadata = {
        "original_model_name_or_path": model_name, "resolved_revision": revision,
    }
    if source_hashes is not None:
        metadata["local_source_sha256"] = source_hashes
    (run / "model_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return run


def test_saved_run_pins_resolved_base_revision_and_allows_evaluation_overrides(tmp_path):
    from evaluate_experiment import resolve_evaluation_config

    run = saved_run(tmp_path)
    config, adapter_dir, provenance = resolve_evaluation_config(
        run_dir=run,
        overrides=['model.device=cpu', 'model.cache_dir=cache', 'data.eval_file=other.jsonl', 'evaluation.thinking=on'],
    )
    assert config["model"]["revision"] == "a" * 40
    assert config["model"]["model_name_or_path"] == "organization/model"
    assert config["model"]["device"] == "cpu"
    assert config["model"]["cache_dir"] == str((run / "cache").resolve())
    assert config["data"]["eval_file"] == str((run / "other.jsonl").resolve())
    assert config["evaluation"]["thinking"] == "on"
    assert adapter_dir == run / "adapter"
    assert provenance["source_config"] == str(run / "resolved_config.json")
    assert len(provenance["source_config_sha256"]) == 64


@pytest.mark.parametrize("override", [
    'model.model_name_or_path=other/model', 'model.revision=other-revision',
    'model.quantization=4bit', 'model.dtype=float32', 'model.backend=auto_causal_lm',
    'model.chat_format=standard', 'model.target_modules=["q_proj"]',
    'model={"model_name_or_path":"other/model"}', 'prompts.system=Changed prompt',
])
def test_saved_run_rejects_changes_to_model_or_prompt_identity(tmp_path, override):
    from evaluate_experiment import resolve_evaluation_config

    with pytest.raises(ValueError, match="saved run|--config"):
        resolve_evaluation_config(run_dir=saved_run(tmp_path), overrides=[override])


@pytest.mark.parametrize("revision", [None, "main", ""])
def test_remote_saved_run_requires_a_recorded_commit_revision(tmp_path, revision):
    from evaluate_experiment import resolve_evaluation_config

    with pytest.raises(ValueError, match="resolved_revision"):
        resolve_evaluation_config(run_dir=saved_run(tmp_path, revision=revision))


def test_local_saved_base_allows_null_hub_revision(tmp_path):
    from evaluate_experiment import resolve_evaluation_config
    from jepa_experiments.models import local_source_fingerprints

    local_model = tmp_path / "local-model"
    local_model.mkdir()
    (local_model / "config.json").write_text("{}", encoding="utf-8")
    config, _, _ = resolve_evaluation_config(run_dir=saved_run(
        tmp_path, revision=None, model_name=str(local_model), source_hashes=local_source_fingerprints(local_model),
    ))
    assert config["model"]["revision"] is None


def test_saved_local_run_requires_snapshot_fingerprints(tmp_path):
    from evaluate_experiment import resolve_evaluation_config

    local_model = tmp_path / "local-model"
    local_model.mkdir()
    (local_model / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="local_source_sha256.*immutable|immutable.*local_source_sha256"):
        resolve_evaluation_config(run_dir=saved_run(tmp_path, revision=None, model_name=str(local_model)))


@pytest.mark.parametrize("change", ["add", "remove", "directory_missing"])
def test_saved_local_snapshot_rejects_changed_file_inventory(tmp_path, change):
    from evaluate_experiment import resolve_evaluation_config
    from jepa_experiments.models import local_source_fingerprints

    local_model = tmp_path / "local-model"
    local_model.mkdir()
    artifact = local_model / "config.json"
    artifact.write_text("{}", encoding="utf-8")
    run = saved_run(tmp_path, revision=None, model_name=str(local_model), source_hashes=local_source_fingerprints(local_model))
    if change == "add":
        (local_model / "extra.json").write_text("{}", encoding="utf-8")
    else:
        artifact.unlink()
        if change == "directory_missing":
            local_model.rmdir()
    with pytest.raises(ValueError, match="local.*snapshot|snapshot.*changed"):
        resolve_evaluation_config(run_dir=run)


def test_base_config_can_be_overridden_without_an_adapter(tmp_path):
    from evaluate_experiment import resolve_evaluation_config

    config_path = tmp_path / "eval.json"
    config_path.write_text(json.dumps({"data": {"eval_file": "eval.jsonl"}}), encoding="utf-8")
    config, adapter_dir, _ = resolve_evaluation_config(config_path=config_path, overrides=['model.model_name_or_path=other/model', 'model.revision=pinned'])
    assert config["model"]["model_name_or_path"] == "other/model"
    assert config["model"]["revision"] == "pinned"
    assert config["data"]["train_file"] is None
    assert adapter_dir is None


def test_evaluation_refuses_existing_output_directory_before_loading_model(tmp_path):
    from evaluate_experiment import main

    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        main(['--config', str(tmp_path / "unused.json"), '--output-dir', str(existing)])
    assert marker.read_text(encoding="utf-8") == "keep"


def test_cli_requires_one_source_and_an_output_directory():
    from evaluate_experiment import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['--config', 'config.json'])
    with pytest.raises(SystemExit):
        parser.parse_args(['--config', 'config.json', '--run-dir', 'run', '--output-dir', 'new'])
    args = parser.parse_args(['--run-dir', 'run', '--output-dir', 'new', '--thinking', 'off', '--set', 'model.device=cpu'])
    assert args.thinking == "off"
    assert args.overrides == ['model.device=cpu']


def test_cli_writes_predictions_metrics_and_reproducibility_manifest(tmp_path, monkeypatch):
    import hashlib
    from evaluate_experiment import main
    import jepa_experiments.models as models

    class NativeTextTokenizer(TextTokenizer):
        chat_template = "native template"
        eos_token = "</s>"

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
            assert tokenize is False
            assert add_generation_prompt is True
            assert enable_thinking is False
            assert [message["role"] for message in messages] == ["system", "user"]
            prompt = "\n".join(message["content"] for message in messages)
            assert "SECRET GOLD RATIONALE" not in prompt
            return prompt

    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text("\n".join(json.dumps({
        "id": str(index), "situation": "Observation", "policy": "Policy",
        "rationale": "SECRET GOLD RATIONALE", "target": {"intervene": True},
    }) for index in range(2)), encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "model": {"model_name_or_path": "organization/model", "revision": "main", "backend": "auto_causal_lm", "chat_format": "standard", "pooling": "last_nonpad", "device": "cpu"},
        "data": {"eval_file": "eval.jsonl"},
    }), encoding="utf-8")

    def load_model(model_config, *, training, adapter_dir):
        assert training is False
        assert adapter_dir is None
        assert model_config["model_name_or_path"] == "organization/model"
        return CompletionModel(['{"intervene":true}</s>', 'invalid</s>']), NativeTextTokenizer(), {"resolved_revision": "b" * 40}

    monkeypatch.setattr(models, "load_model", load_model)
    output = tmp_path / "evaluation"
    assert main(['--config', str(config_path), '--output-dir', str(output)]) == 0
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert metrics["examples"] == len(rows) == 2
    assert metrics["resolved_revision"] == "b" * 40
    assert metrics["prompt_mode"] == "thinking_off"
    assert metrics["json_valid_count"] == 1
    assert manifest["status"] == "completed"
    assert manifest["source_config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert manifest["data_stats"]["file_hash"] == hashlib.sha256(eval_file.read_bytes()).hexdigest()
    assert manifest["resolved_config_sha256"] == hashlib.sha256((output / "resolved_config.json").read_bytes()).hexdigest()
    assert json.loads((output / "model_metadata.json").read_text(encoding="utf-8"))["resolved_revision"] == "b" * 40
    assert rows[0]["generation_raw"] == '{"intervene":true}</s>'
    assert rows[0]["parsed_answer"] == {"intervene": True}
    assert rows[1]["json_error"]
