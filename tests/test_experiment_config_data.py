"""Contract tests for experiment configuration and input datasets."""

import copy
import hashlib
import json

import pytest


def config_api():
    from jepa_experiments.config import apply_overrides, load_config, validate_config

    return apply_overrides, load_config, validate_config


def write_jsonl(tmp_path, rows, name="records.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return path


def data_config(tmp_path, rows, **updates):
    _, _, validate = config_api()
    config = {"train_file": str(write_jsonl(tmp_path, rows)), **updates}
    return validate({"data": config}, base_dir=tmp_path)["data"]


def test_load_resolves_paths_and_preserves_hub_model_ids(tmp_path):
    _, load, _ = config_api()
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"data": {"train_file": "data/train.jsonl"}, "model": {"model_name_or_path": "org/custom"}}))
    config = load(path, {"training": {"output_dir": "runs/trial"}})
    assert config["model"]["model_name_or_path"] == "org/custom"
    assert config["model"]["revision"] is None
    assert config["data"]["train_file"] == str(tmp_path / "data" / "train.jsonl")
    assert config["model"]["cache_dir"] == str(tmp_path / ".cache" / "huggingface")
    assert config["training"]["output_dir"] == str(tmp_path / "runs" / "trial")


def test_default_model_is_pinned_and_explicit_local_model_paths_resolve(tmp_path):
    _, _, validate = config_api()
    assert validate({}, tmp_path)["model"]["revision"] == "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
    config = validate({"model": {"model_name_or_path": "./models/tiny", "decoder_path": "model.layers"}}, tmp_path)
    assert config["model"]["model_name_or_path"] == str(tmp_path / "models" / "tiny")
    assert config["model"]["revision"] is None
    assert config["model"]["decoder_path"] == "model.layers"


def test_overrides_deep_merge_parse_values_and_do_not_mutate():
    apply, _, validate = config_api()
    original = {"training": {"seed": 13}}
    snapshot = copy.deepcopy(original)
    result = apply(original, ["training.batch_size=2", "model.gradient_checkpointing=false", "experiment.name=trial=1"])
    assert original == snapshot
    assert result["training"] == {"seed": 13, "batch_size": 2}
    assert result["model"]["gradient_checkpointing"] is False
    assert result["experiment"]["name"] == "trial=1"
    assert validate(result)["training"]["seed"] == 13


def test_model_override_does_not_leak_default_revision_into_custom_model():
    apply, _, validate = config_api()
    config = validate({})
    changed = validate(apply(config, ["model.model_name_or_path=org/custom"]))
    assert changed["model"]["revision"] is None
    pinned = validate(apply(config, ["model.revision=custom-commit", "model.model_name_or_path=org/custom"]))
    assert pinned["model"]["revision"] == "custom-commit"


def test_loading_saved_config_with_custom_model_override_resets_revision(tmp_path):
    _, load, validate = config_api()
    path = tmp_path / "saved.json"
    path.write_text(json.dumps(validate({})))
    config = load(path, {"model": {"model_name_or_path": "org/custom"}})
    assert config["model"]["revision"] is None


@pytest.mark.parametrize("config, match", [
    ({"experiment": {"recpie": "sft"}}, "recpie"),
    ({"data": {"fields": {"extra": "x"}}}, "extra"),
    ({"training": {"batch_size": True}}, "batch_size"),
    ({"training": {"learning_rate": float("nan")}}, "learning_rate"),
    ({"training": {"num_epochs": 0}}, "num_epochs"),
    ({"training": {"num_epochs": 1.5}}, "num_epochs"),
    ({"model": {"gradient_checkpointing": "false"}}, "gradient_checkpointing"),
    ({"experiment": {"recipe": "unknown"}}, "recipe"),
    ({"model": {"chat_format": "standard"}, "experiment": {"recipe": "cot_jepa"}}, "adapter"),
    ({"model": {"chat_format": "standard", "pooling": "turn_end"}}, "last_nonpad"),
    ({"prompts": {"input_template": "{missing}"}}, "input_template"),
])
def test_config_rejects_unknown_fields_bad_types_and_unsupported_templates(config, match):
    _, _, validate = config_api()
    with pytest.raises(ValueError, match=match):
        validate(config)


@pytest.mark.parametrize("item", ["training.seed", "training..seed=2", "unknown=2", "training.seed.value=2"])
def test_malformed_or_unknown_overrides_are_rejected(item):
    apply, _, _ = config_api()
    with pytest.raises(ValueError):
        apply({}, [item])


def test_records_support_nested_mapping_and_canonical_target(tmp_path):
    from jepa_experiments.data import load_examples

    rows = [{"key": "case-a", "input": {"s": "상황", "p": "policy"}, "answer": {"reason": "because", "json": '{"z": 2, "a": "한글"}'}}]
    config = data_config(tmp_path, rows, fields={"id": "key", "situation": "input.s", "policy": "input.p", "rationale": "answer.reason", "target": "answer.json"})
    examples, stats = load_examples(config, require_rationale=True)
    assert (examples[0].id, examples[0].situation, examples[0].policy, examples[0].rationale) == ("case-a", "상황", "policy", "because")
    assert examples[0].target_text == '{"a":"한글","z":2}'
    assert stats["total"] == stats["read"] == stats["selected"] == 1
    assert stats["file_hash"] == hashlib.sha256((tmp_path / "records.jsonl").read_bytes()).hexdigest()


def test_optional_id_and_rationale_fallback_and_first_n_selection(tmp_path):
    from jepa_experiments.data import load_examples

    rows = [{"situation": f"s{i}", "target": {"action": "wait"}} for i in range(3)]
    config = data_config(tmp_path, rows, fields={"id": None, "rationale": None}, default_policy="p", max_samples=2)
    examples, stats = load_examples(config)
    assert [example.id for example in examples] == ["1", "2"]
    assert all(example.rationale is None and example.policy == "p" for example in examples)
    assert stats["total"] == stats["read"] == 3
    assert stats["selected"] == 2


@pytest.mark.parametrize("row, match", [
    ({"situation": "s", "policy": "p", "target": {}}, "rationale"),
    ({"situation": 3, "policy": "p", "target": {}, "rationale": "r"}, "situation"),
    ({"situation": "s", "policy": "", "target": {}, "rationale": "r"}, "policy"),
    ({"situation": "s", "policy": "p", "target": "[]", "rationale": "r"}, "target"),
    ({"situation": "s", "policy": "p", "target": '{"value":NaN}', "rationale": "r"}, "target"),
    ({"situation": "s", "policy": "p", "target": "{bad}", "rationale": "r"}, "target"),
])
def test_invalid_records_fail_with_file_line_context(tmp_path, row, match):
    from jepa_experiments.data import load_examples

    with pytest.raises(ValueError, match=rf"records.jsonl:1:.*{match}"):
        load_examples(data_config(tmp_path, [row]), require_rationale=True)


def test_duplicate_ids_and_bad_records_after_limit_are_not_silently_dropped(tmp_path):
    from jepa_experiments.data import load_examples

    row = {"id": "same", "situation": "s", "policy": "p", "target": {}}
    with pytest.raises(ValueError, match=r"records.jsonl:2:.*duplicate"):
        load_examples(data_config(tmp_path, [row, row], max_samples=1))


def test_legacy_single_turn_messages_support_reasoning_and_default_policy(tmp_path):
    from jepa_experiments.data import load_examples

    rows = [
        {"id": "a", "messages": [{"role": "system", "content": "p"}, {"role": "user", "content": "s"}, {"role": "assistant", "content": '{"intervene":true}', "reasoning": "r"}]},
        {"messages": [{"role": "user", "content": "s2"}, {"role": "assistant", "content": '{"intervene":false}', "reasoning_content": "r2"}]},
    ]
    examples, _ = load_examples(data_config(tmp_path, rows, format="messages", default_policy="fallback"), require_rationale=True)
    assert examples[0].policy == "p"
    assert examples[1].policy == "fallback"
    assert examples[1].rationale == "r2"
    assert examples[1].target == {"intervene": False}


def test_messages_reject_multiple_user_turns(tmp_path):
    from jepa_experiments.data import load_examples

    messages = [{"role": "user", "content": "s"}, {"role": "assistant", "content": "{}"}] * 2
    with pytest.raises(ValueError, match=r"records.jsonl:1:.*single.turn"):
        load_examples(data_config(tmp_path, [{"messages": messages}], format="messages", default_policy="p"))


def test_eval_file_can_be_loaded_without_train_file(tmp_path):
    from jepa_experiments.data import load_examples

    _, _, validate = config_api()
    path = write_jsonl(tmp_path, [{"situation": "s", "policy": "p", "target": {"intervene": False}}])
    config = validate({"data": {"eval_file": str(path)}})
    examples, stats = load_examples(config["data"], split="eval")
    assert len(examples) == stats["selected"] == 1
    with pytest.raises(ValueError, match="train_file"):
        load_examples(config["data"])


def test_json_schema_validates_target_and_records_schema_hash(tmp_path):
    pytest.importorskip("jsonschema")
    from jepa_experiments.data import load_examples

    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"type": "object", "required": ["action"], "properties": {"action": {"enum": ["wait"]}}, "additionalProperties": False}))
    config = data_config(tmp_path, [{"situation": "s", "target": {"action": "wait"}}], default_policy="p", target_schema=str(schema))
    _, stats = load_examples(config)
    assert stats["schema_hash"] == hashlib.sha256(schema.read_bytes()).hexdigest()
    write_jsonl(tmp_path, [{"situation": "s", "target": {"action": "go"}}])
    with pytest.raises(ValueError, match=r"records.jsonl:1:.*schema"):
        load_examples(config)
