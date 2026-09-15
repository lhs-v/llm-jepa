import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Gemma4ForCausalLM, Gemma4TextConfig, PreTrainedTokenizerFast


@pytest.fixture(scope="module")
def tiny_experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("training")
    base = root / "base"
    words = ["<unk>", "<bos>", "<eos>", "<pad>", "<turn|>", "<|turn>", "<|think|>",
             "<|channel>", "<channel|>", "system", "user", "model", "thought", "Policy",
             "Situation", "help", "requested", "quiet", "intervene", "true", "false"]
    core = Tokenizer(WordLevel(dict(zip(words, range(len(words)))), unk_token="<unk>"))
    core.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="<unk>", bos_token="<bos>",
                                 eos_token="<eos>", pad_token="<pad>",
                                 additional_special_tokens=words[4:9])
    tok.chat_template = (
        "{{ bos_token }}{% for m in messages %}{{ '<|turn>' + ('model' if m['role'] == 'assistant' else m['role']) + '\n' }}"
        "{% if m['role'] == 'system' and enable_thinking %}{{ '<|think|>' }}{% endif %}"
        "{% if m.get('reasoning') %}{{ '<|channel>thought\n' + m['reasoning'] + '\n<channel|>' }}{% endif %}"
        "{{ m['content'] + '<turn|>\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|turn>model\n' }}{% endif %}"
    )
    tok.save_pretrained(base)
    model = Gemma4ForCausalLM(Gemma4TextConfig(
        vocab_size=64, vocab_size_per_layer_input=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        hidden_size_per_layer_input=4, num_kv_shared_layers=0,
        layer_types=["sliding_attention", "full_attention"], per_layer_config={},
        sliding_window=8, max_position_embeddings=512, bos_token_id=1, eos_token_id=2, pad_token_id=3,
    ))
    model.save_pretrained(base)
    rows = [{"id": str(i), "situation": "help requested" if i % 2 else "quiet",
             "policy": "help requested", "rationale": "requested" if i % 2 else "quiet",
             "target": {"intervene": bool(i % 2)}} for i in range(5)]
    data = root / "train.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return root, base, data


def config_for(tiny_experiment, recipe, output):
    from jepa_experiments.config import validate_config
    _, base, data = tiny_experiment
    return validate_config({
        "experiment": {"recipe": recipe},
        "model": {"model_name_or_path": str(base), "device": "cpu", "dtype": "float32",
                  "lora_rank": 2, "lora_alpha": 4, "lora_dropout": 0.0},
        "data": {"train_file": str(data), "eval_file": str(data), "max_length": 256},
        "training": {"output_dir": str(output), "num_epochs": 1, "batch_size": 2,
                     "gradient_accumulation_steps": 2, "max_steps": 2, "save_every_steps": 1},
    })


@pytest.mark.parametrize("recipe", ["sft", "json_jepa", "rationale_jepa", "multitask", "multitask_jepa", "cot", "cot_jepa", "mixed_jepa"])
def test_complete_training_saves_and_reloads_all_recipes(tiny_experiment, tmp_path, recipe):
    from jepa_experiments.training import train
    from jepa_experiments.models import load_model
    torch.set_num_threads(2)
    output = tmp_path / recipe
    config = config_for(tiny_experiment, recipe, output)
    result = train(config)
    assert result["optimizer_steps"] == 2
    assert result["examples_seen"] == 5  # Partial accumulation group must run.
    assert result["first_gradient_norm"] > 0
    assert (output / "adapter/adapter_model.safetensors").exists()
    assert (output / "checkpoints/step-000002/adapter/adapter_config.json").exists()
    saved = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert saved["experiment"]["recipe"] == recipe
    assert manifest["features"]["selected_ids"] == [str(i) for i in range(5)]
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert records[-1]["examples"] == 1
    assert all(torch.isfinite(torch.tensor(record["loss"])) for record in records)
    loaded, _, _ = load_model(saved["model"], training=False, adapter_dir=output / "adapter")
    assert all(not p.requires_grad for p in loaded.parameters())
    assert any(p.abs().sum() > 0 for n, p in loaded.named_parameters() if "lora_B" in n)
    from evaluate_experiment import main as evaluate_main
    evaluation = tmp_path / f"{recipe}-eval"
    assert evaluate_main(["--run-dir", str(output), "--output-dir", str(evaluation),
                          "--thinking", "on" if recipe in {"cot", "cot_jepa", "mixed_jepa"} else "off",
                          "--set", "evaluation.max_new_tokens=2", "--set", "evaluation.max_samples=1"]) == 0
    metrics = json.loads((evaluation / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["examples"] == 1
    assert metrics["resolved_revision"] is None


def test_prepare_does_not_create_output_and_refuses_overwrite(tiny_experiment, tmp_path):
    from jepa_experiments.training import prepare, train
    config = config_for(tiny_experiment, "rationale_jepa", tmp_path / "output")
    report = prepare(config)
    assert report["features"]["selected_examples"] == 5
    assert not Path(config["training"]["output_dir"]).exists()
    output = Path(config["training"]["output_dir"])
    output.mkdir()
    (output / "keep.txt").write_text("existing run")
    with pytest.raises(FileExistsError):
        train(config)
    assert (output / "keep.txt").read_text() == "existing run"


def test_independent_rationale_never_changes_direct_path(tiny_experiment):
    from dataclasses import replace
    from jepa_experiments.data import Example
    from jepa_experiments.models import load_tokenizer
    from jepa_experiments.formatting import Formatter
    from jepa_experiments.objectives import build_features, resolve_recipe
    config = config_for(tiny_experiment, "rationale_jepa", tiny_experiment[0] / "unused")
    fmt = Formatter(load_tokenizer(config["model"]), prompts=config["prompts"])
    row = Example("a", "help", "requested", "quiet", {"intervene": True})
    first, _ = build_features([row], fmt, resolve_recipe(config["experiment"]), 256)
    changed, _ = build_features([replace(row, rationale="help requested")], fmt, resolve_recipe(config["experiment"]), 256)
    assert first[0]["direct"] == changed[0]["direct"]
    assert first[0]["view_x"] == changed[0]["view_x"]
    assert first[0]["view_target"] != changed[0]["view_target"]
