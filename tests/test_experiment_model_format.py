"""Real tokenizers and small local causal models; never download model weights."""
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoTokenizer, Gemma4ForCausalLM, Gemma4TextConfig, LlamaConfig,
    LlamaForCausalLM, PreTrainedTokenizerFast,
)


@pytest.fixture(scope="module")
def official_tokenizer():
    try:
        return AutoTokenizer.from_pretrained(
            "google/gemma-4-E2B-it", revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
            cache_dir=str(Path(__file__).resolve().parents[1] / ".cache/huggingface"),
            local_files_only=True,
        )
    except OSError:
        pytest.skip("Official Gemma 4 tokenizer is not cached")


def native_tokenizer():
    tokenizer = Tokenizer(WordLevel(
        {"<unk>": 0, "<bos>": 1, "<eos>": 2, "user": 3, "assistant": 4,
         "system": 5, "Policy": 6, "Situation": 7, "hello": 8, "yes": 9}, unk_token="<unk>",
    ))
    tokenizer.pre_tokenizer = Whitespace()
    result = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>",
                                     bos_token="<bos>", eos_token="<eos>")
    result.chat_template = (
        "{{ bos_token }}{% for m in messages %}{{ m['role'] + '\\n' + m['content'] }}"
        "{% if m['role'] == 'assistant' %}{{ eos_token }}{% endif %}{{ '\\n' }}"
        "{% endfor %}{% if add_generation_prompt %}{{ 'assistant\\n' }}{% endif %}"
    )
    return result


def example(**changes):
    values = dict(id="e1", situation="SITUATION", policy="POLICY", rationale="REASON",
                  target={"allowed": True}, target_text='{"allowed":true}')
    values.update(changes)
    return SimpleNamespace(**values)


def tiny_gemma():
    return Gemma4ForCausalLM(Gemma4TextConfig(
        vocab_size=32, vocab_size_per_layer_input=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, hidden_size_per_layer_input=4,
        num_kv_shared_layers=0, layer_types=["sliding_attention", "full_attention"],
        per_layer_config={}, sliding_window=8, max_position_embeddings=64,
        attention_dropout=0.0,
    ))


def model_config(path):
    return dict(model_name_or_path=str(path), backend="auto_causal_lm", chat_format="standard",
                pooling="last_nonpad", dtype="float32", device="cpu", quantization="none",
                lora_rank=2, lora_alpha=4, lora_dropout=0.0,
                target_modules=["q_proj", "v_proj"], gradient_checkpointing=True)


def test_gemma_supervision_masks_prompt_and_separates_three_branches(official_tokenizer):
    from jepa_experiments.formatting import Formatter
    formatter = Formatter(official_tokenizer, prompts={"system": "JSON only", "rationale_system": "Explain"})
    expected = {"direct": '{"allowed":true}<turn|>', "rationale": "REASON<turn|>",
                "sequential": '<|channel>thought\nREASON\n<channel|>{"allowed":true}<turn|>'}
    for branch, answer in expected.items():
        encoded = formatter.encode_supervision(example(), branch)
        labels = encoded["labels"]
        assert official_tokenizer.decode([x for x in labels if x != -100]) == answer
        assert encoded["input_ids"].count(official_tokenizer.bos_token_id) == 1
        prefix = encoded["input_ids"][:labels.index(next(x for x in labels if x != -100))]
        assert "REASON" not in official_tokenizer.decode(prefix)
        assert '"allowed"' not in official_tokenizer.decode(prefix)
    assert "<|think|>" in formatter.render_prompt(example(), thinking=True)
    assert "<|think|>" not in formatter.render_prompt(example())
    assert formatter.messages(example(), "rationale")[0]["content"] == "Explain"


def test_gemma_independent_view_ends_at_turn_and_never_leaks_example(official_tokenizer):
    from jepa_experiments.formatting import Formatter
    formatter = Formatter(official_tokenizer)
    assert official_tokenizer.decode(formatter.encode_view("REASON")) == "<bos><|turn>user\nREASON<turn|>"
    assert formatter.input_text(example()) == "Policy:\nPOLICY\n\nSituation:\nSITUATION"
    with pytest.raises(ValueError, match="situation|policy"):
        Formatter(official_tokenizer, prompts={"input_template": "{rationale}"})
    with pytest.raises(ValueError, match="nonempty"):
        formatter.encode_supervision(example(target_text=""), "direct")
    with pytest.raises(ValueError, match="nonempty"):
        formatter.encode_supervision(example(rationale=""), "sequential")


@pytest.mark.parametrize("raw,answer,rationale,thought,error", [
    ('{"allowed":true}<turn|>\n', '{"allowed":true}', None, False, False),
    ('<|channel>thought\nreason\n<channel|>{"allowed":true}<turn|><eos>', '{"allowed":true}', "reason", True, False),
    ('<|channel>thought\nunfinished', "", "unfinished", True, True),
    ('<channel|>{"allowed":true}<turn|>', '{"allowed":true}', None, False, True),
    ('```json\n{"allowed":true}\n```<turn|>', '```json\n{"allowed":true}\n```', None, False, False),
])
def test_generation_parses_channels_without_repairing_json(official_tokenizer, raw, answer, rationale, thought, error):
    from jepa_experiments.formatting import Formatter
    parsed = Formatter(official_tokenizer).parse_generation(raw)
    assert parsed["answer"] == answer
    assert parsed["rationale"] == rationale
    assert parsed["thought_generated"] is thought
    assert bool(parsed["format_error"]) is error


def test_standard_format_uses_native_template_and_rejects_gemma_sequential():
    from jepa_experiments.formatting import Formatter
    tokenizer = native_tokenizer()
    formatter = Formatter(tokenizer, chat_format="standard", pooling="last_nonpad")
    encoded = formatter.encode_supervision(example(target_text="yes"), "direct")
    assert tokenizer.decode([x for x in encoded["labels"] if x != -100]) == "yes <eos>"
    assert formatter.parse_generation("<|channel>thought literal<eos>")["answer"] == "<|channel>thought literal"
    assert formatter.encode_view("hello") == tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}], tokenize=True, add_generation_prompt=False, return_dict=False,
    )
    with pytest.raises(ValueError, match="Gemma|gemma"):
        formatter.encode_supervision(example(), "sequential")
    with pytest.raises(ValueError, match="thinking|Thinking"):
        formatter.render_prompt(example(), thinking=True)


@pytest.mark.parametrize("kind", ["gemma", "llama"])
def test_decoder_keeps_adapters_and_frozen_embeddings_and_backpropagates(kind):
    from jepa_experiments.models import attach_lora, get_decoder
    base = tiny_gemma() if kind == "gemma" else LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
    ))
    base.to(dtype=torch.bfloat16)
    model = attach_lora(base, model_config("unused"))
    model.train()
    assert get_decoder(model) is get_decoder(model, "model")
    ids = torch.tensor([[1, 8, 9, 2]])
    hidden = get_decoder(model)(input_ids=ids, use_cache=False).last_hidden_state
    hidden.float().square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if "lora_" in n)
    assert all(not p.requires_grad and p.grad is None for n, p in model.named_parameters() if "lora_" not in n)
    assert all(p.dtype == torch.bfloat16 for n, p in model.named_parameters() if "embed_tokens" in n)
    with pytest.raises(ValueError, match="decoder_path"):
        get_decoder(model, "missing.module")
    with pytest.raises(ValueError, match="decoder_path"):
        get_decoder(model, "lm_head")


def test_local_generic_loader_tokenizer_padding_and_adapter_roundtrip(tmp_path):
    from jepa_experiments.models import load_model, load_tokenizer
    base_dir, adapter_dir = tmp_path / "base", tmp_path / "adapter"
    native_tokenizer().save_pretrained(base_dir)
    base = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
    ))
    base.save_pretrained(base_dir)
    config = model_config(base_dir)
    tokenizer = load_tokenizer(config)
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
    assert tokenizer.padding_side == "right"
    model, _, metadata = load_model(config)
    assert metadata["resolved_revision"] is None
    assert metadata["model_name_or_path"] == str(base_dir)
    assert metadata["pad_token_fallback"] == "eos"
    assert {"config.json", "model.safetensors", "tokenizer.json"} <= metadata["local_source_sha256"].keys()
    ids = torch.tensor([[1, 8, 9, 2]])
    model.eval()
    with torch.no_grad():
        expected = model(ids).logits
    model.save_pretrained(adapter_dir)
    loaded, _, loaded_metadata = load_model(config, training=False, adapter_dir=adapter_dir)
    assert all(not p.requires_grad for p in loaded.parameters())
    assert len(loaded.peft_config) == 1
    assert loaded_metadata["adapter_dir"] == str(adapter_dir)
    assert loaded_metadata["local_source_sha256"] == metadata["local_source_sha256"]
    with torch.no_grad():
        torch.testing.assert_close(expected, loaded(ids).logits)


def test_invalid_cpu_quantization_fails_before_reading_weights():
    from jepa_experiments.models import load_model
    config = model_config("does-not-exist")
    config["quantization"] = "4bit"
    with pytest.raises(ValueError, match="CUDA|cuda"):
        load_model(config)


def test_gemma_backend_restores_local_multimodal_text_weights(tmp_path, official_tokenizer):
    from safetensors.torch import save_file
    from transformers import Gemma4Config
    from jepa_experiments.models import load_model
    original = tiny_gemma().eval()
    Gemma4Config(text_config=original.config.to_dict()).save_pretrained(tmp_path)
    weights = {name.replace("model.", "model.language_model.", 1): value.clone()
               for name, value in original.state_dict().items() if name != "lm_head.weight"}
    weights["model.vision_tower.unused.weight"] = torch.ones(2, 2)
    save_file(weights, str(tmp_path / "model.safetensors"))
    official_tokenizer.save_pretrained(tmp_path)
    config = {**model_config(tmp_path), "backend": "gemma4", "chat_format": "gemma4", "pooling": "turn_end"}
    loaded, _, metadata = load_model(config, training=False)
    assert isinstance(loaded, Gemma4ForCausalLM)
    assert not hasattr(loaded.model, "vision_tower")
    assert metadata["resolved_revision"] is None
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)


def test_tokenizer_prepare_requires_native_template_without_loading_weights(tmp_path):
    from jepa_experiments.models import load_tokenizer
    tokenizer = native_tokenizer()
    tokenizer.chat_template = None
    tokenizer.save_pretrained(tmp_path)
    LlamaConfig().save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="chat template"):
        load_tokenizer(model_config(tmp_path))


def test_generic_loader_rejects_bidirectional_encoder_config_before_weights(tmp_path):
    from transformers import BertConfig
    from jepa_experiments.models import load_model
    BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
               num_attention_heads=2, intermediate_size=32, is_decoder=False).save_pretrained(tmp_path)
    native_tokenizer().save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="causal|decoder"):
        load_model(model_config(tmp_path))


def test_local_source_fingerprints_include_nested_artifacts_and_skip_hidden_directories(tmp_path):
    from jepa_experiments.models import local_source_fingerprints

    (tmp_path / "model.safetensors").write_bytes(b"abc")
    (tmp_path / "tokenizer").mkdir()
    (tmp_path / "tokenizer" / "vocab.txt").write_bytes(b"")
    for hidden in (tmp_path / ".git", tmp_path / ".cache", tmp_path / "tokenizer" / ".cache"):
        hidden.mkdir()
        (hidden / "ignored.txt").write_text("mutable cache", encoding="utf-8")
    expected = {
        "model.safetensors": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "tokenizer/vocab.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    }
    assert local_source_fingerprints(tmp_path) == expected
    (tmp_path / ".cache" / "ignored.txt").write_text("changed cache", encoding="utf-8")
    assert local_source_fingerprints(tmp_path) == expected
    assert local_source_fingerprints("organization/remote-model-not-present") is None


def test_local_source_fingerprints_do_not_silently_skip_unreadable_subdirectories(tmp_path, monkeypatch):
    import jepa_experiments.models as models

    nested = tmp_path / "artifacts"
    nested.mkdir()
    scan = models.os.scandir

    def guarded_scan(path):
        if Path(path) == nested:
            raise PermissionError("Unreadable snapshot directory")
        return scan(path)

    monkeypatch.setattr(models.os, "scandir", guarded_scan)
    with pytest.raises(PermissionError, match="Unreadable snapshot"):
        models.local_source_fingerprints(tmp_path)


@pytest.mark.parametrize("changed_artifact", ["weights", "tokenizer"])
def test_saved_local_adapter_rejects_changed_base_artifacts_before_reload(tmp_path, changed_artifact):
    from evaluate_experiment import resolve_evaluation_config
    from jepa_experiments.config import validate_config
    from jepa_experiments.models import load_model
    from jepa_experiments.training import _save_adapter

    base_dir, run_dir = tmp_path / "base", tmp_path / "run"
    tokenizer = native_tokenizer()
    tokenizer.save_pretrained(base_dir)
    base = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
    ))
    base.save_pretrained(base_dir)
    config = validate_config({"experiment": {"recipe": "sft"}, "model": model_config(base_dir)})
    model, loaded_tokenizer, metadata = load_model(config["model"])
    _save_adapter(run_dir, model, loaded_tokenizer, config, metadata)

    resolved, adapter_dir, _ = resolve_evaluation_config(run_dir=run_dir)
    restored, _, restored_metadata = load_model(resolved["model"], training=False, adapter_dir=adapter_dir)
    assert restored_metadata["local_source_sha256"] == metadata["local_source_sha256"]
    assert all(not parameter.requires_grad for parameter in restored.parameters())

    if changed_artifact == "weights":
        with torch.no_grad():
            base.model.embed_tokens.weight[1, 0].add_(0.5)
        base.save_pretrained(base_dir)
    else:
        tokenizer.chat_template += "\nchanged template"
        tokenizer.save_pretrained(base_dir)
    with pytest.raises(ValueError, match="local.*snapshot|snapshot.*changed"):
        resolve_evaluation_config(run_dir=run_dir)
