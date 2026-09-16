import json
from pathlib import Path

import pytest
import torch

import evaluate_gemma4 as evaluation


class FakeTokenizer:
    pad_token_id = 0

    def __init__(self, decoded: str = "answer") -> None:
        self.decoded = decoded
        self.template_call = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_call = (messages, kwargs)
        return "rendered prompt"

    def __call__(self, text, **kwargs):
        assert text == "rendered prompt"
        assert kwargs["add_special_tokens"] is False
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }

    def decode(self, token_ids, **kwargs):
        assert token_ids.tolist() == [7, 8]
        assert kwargs["skip_special_tokens"] is True
        return f" {self.decoded} "


class FakeModel:
    device = torch.device("cpu")

    def generate(self, input_ids, attention_mask, **kwargs):
        assert attention_mask.tolist() == [[1, 1]]
        assert kwargs == {"max_new_tokens": 16, "do_sample": False, "pad_token_id": 0}
        return torch.cat((input_ids, torch.tensor([[7, 8]])), dim=1)


def test_render_prompt_uses_gemma4_template_without_thinking():
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Question"},
    ]

    assert evaluation.render_prompt(tokenizer, messages) == "rendered prompt"
    assert tokenizer.template_call == (
        messages,
        {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        },
    )


def test_evaluate_examples_reports_only_exact_match():
    results, metrics = evaluation.evaluate_examples(
        FakeModel(),
        FakeTokenizer(),
        [{"messages": [{"role": "user", "content": "Question"}], "reference": "answer"}],
        max_new_tokens=16,
    )

    assert metrics == {
        "examples": 1,
        "scored_examples": 1,
        "exact_matches": 1,
        "exact_match": 1.0,
    }
    assert results[0]["generation"] == "answer"
    assert results[0]["exact_match"] is True


def test_adapter_settings_come_from_training_run_config(tmp_path: Path):
    run_dir = tmp_path / "run"
    adapter_dir = run_dir / "adapter"
    adapter_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "model_name_or_path": "google/gemma-4-E2B-it",
                "revision": "abc123",
                "quantization": "none",
            }
        ),
        encoding="utf-8",
    )

    settings = evaluation.resolve_model_settings(
        adapter_dir=adapter_dir,
        model_name_or_path=None,
        revision=None,
        cache_dir="cache",
        quantization=None,
    )

    assert settings.model_name_or_path == "google/gemma-4-E2B-it"
    assert settings.revision == "abc123"
    assert settings.quantization == "none"
    assert settings.run_config_path == run_dir / "run_config.json"


def test_write_results_refuses_existing_output_directory(tmp_path: Path):
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    settings = evaluation.ModelSettings("model", None, None, "none")

    with pytest.raises(FileExistsError):
        evaluation.write_results(output_dir, [], {}, settings, None)

    assert marker.read_text(encoding="utf-8") == "keep"


def test_peft_adapter_can_reload_on_tiny_gemma4(tmp_path: Path):
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    config = transformers.Gemma4TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"],
        vocab_size_per_layer_input=32,
        hidden_size_per_layer_input=4,
        num_kv_shared_layers=0,
    )
    trained = peft.get_peft_model(
        transformers.Gemma4ForCausalLM(config),
        peft.LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]),
    )
    adapter_dir = tmp_path / "adapter"
    trained.save_pretrained(adapter_dir)

    reloaded = peft.PeftModel.from_pretrained(
        transformers.Gemma4ForCausalLM(config), adapter_dir, is_trainable=False
    )

    assert isinstance(reloaded, peft.PeftModel)
    assert any("lora_A" in name for name, _ in reloaded.named_parameters())
