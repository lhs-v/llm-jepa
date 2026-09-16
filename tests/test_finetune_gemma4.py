"""Small real Gemma 4 tests; cached official tokenizer tests need no network."""
import copy
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, Gemma4ForCausalLM, Gemma4TextConfig

from finetune_gemma4 import (
    DEFAULT_MODEL, DEFAULT_REVISION, add_lora, collate_examples,
    encode_example, jepa_loss, lm_loss,
)


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return AutoTokenizer.from_pretrained(
            DEFAULT_MODEL, revision=DEFAULT_REVISION,
            cache_dir=str(Path(__file__).resolve().parents[1] / ".cache/huggingface"),
            local_files_only=True,
        )
    except OSError:
        pytest.skip("Run --prepare_only once to cache the official Gemma 4 tokenizer")


def messages(answer="[0-9]+"):
    return [
        {"role": "system", "content": "Convert natural language to regex."},
        {"role": "user", "content": "all digits"},
        {"role": "assistant", "content": answer},
    ]


def test_official_chat_masks_prompt_and_keeps_two_independent_views(tokenizer):
    example = encode_example(tokenizer, messages(), max_length=128)
    supervised = [x for x in example["labels"] if x != -100]
    assert tokenizer.decode(supervised) == "[0-9]+<turn|>"
    assert example["input_ids"].count(tokenizer.bos_token_id) == 1
    assert tokenizer.decode(example["text_ids"]) == "<bos><|turn>user\nall digits<turn|>"
    assert tokenizer.decode(example["code_ids"]) == "<bos><|turn>user\n[0-9]+<turn|>"
    changed = encode_example(tokenizer, messages("abc"), max_length=128)
    assert changed["text_ids"] == example["text_ids"]
    assert changed["code_ids"] != example["code_ids"]
    batch = collate_examples([example, changed], tokenizer.pad_token_id)
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100)


def test_overlength_and_empty_answers_are_rejected(tokenizer):
    with pytest.raises(ValueError, match="max_length"):
        encode_example(tokenizer, messages("long " * 200), max_length=32)
    with pytest.raises(ValueError, match="nonempty"):
        encode_example(tokenizer, messages(""), max_length=128)


def tiny_model(checkpointing=False):
    config = Gemma4TextConfig(
        vocab_size=128, vocab_size_per_layer_input=128,
        hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        hidden_size_per_layer_input=8, num_kv_shared_layers=2,
        layer_types=["sliding_attention", "full_attention"] * 2,
        per_layer_config={}, sliding_window=8, max_position_embeddings=64,
        attention_dropout=0.0,
    )
    torch.manual_seed(7)
    return add_lora(Gemma4ForCausalLM(config), rank=2, dropout=0.0,
                    gradient_checkpointing=checkpointing)


def tiny_batch():
    examples = [
        {"input_ids": [2, 7, 8, 9, 10, 1], "labels": [-100]*3 + [9, 10, 1],
         "text_ids": [2, 7, 8, 1], "code_ids": [2, 9, 10, 1]},
        {"input_ids": [2, 4, 5, 1], "labels": [-100]*2 + [5, 1],
         "text_ids": [2, 4, 1], "code_ids": [2, 5, 1]},
    ]
    return collate_examples(examples, pad_token_id=0)


def test_jepa_backpropagates_through_both_views_and_only_adapters():
    model = tiny_model()
    states = []
    def capture(_module, _args, output):
        output.last_hidden_state.retain_grad()
        states.append(output.last_hidden_state)
    handle = model.get_base_model().model.register_forward_hook(capture)
    loss = jepa_loss(model, tiny_batch())
    loss.backward()
    handle.remove()
    assert len(states) == 2
    assert all(s.grad is not None and s.grad.abs().sum() > 0 for s in states)
    assert torch.isfinite(loss) and 0 <= loss <= 2
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for n, p in model.named_parameters() if "lora_" in n)
    assert all(not p.requires_grad and p.grad is None
               for n, p in model.named_parameters() if "lora_" not in n)


def test_checkpointing_with_ple_and_shared_kv_preserves_gradients():
    plain = tiny_model()
    checkpointed = tiny_model(checkpointing=True)
    checkpointed.load_state_dict(plain.state_dict())
    batch = tiny_batch()
    losses = []
    for model in (plain, checkpointed):
        model.train()
        ce = lm_loss(model, batch)
        ce.backward()
        distance = jepa_loss(model, batch)
        (0.1 * distance).backward()
        losses.append((ce.detach(), distance.detach()))
    torch.testing.assert_close(torch.stack(losses[0]), torch.stack(losses[1]))
    for (_, a), (_, b) in zip(plain.named_parameters(), checkpointed.named_parameters()):
        if a.requires_grad:
            assert a.grad is not None and torch.isfinite(a.grad).all()
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-4)


def test_jepa_ignores_right_padding():
    model = tiny_model().eval()
    batch = tiny_batch()
    padded = copy.deepcopy(batch)
    for key, value in list(padded.items()):
        pad = -100 if key == "labels" else 0
        padded[key] = torch.nn.functional.pad(value, (0, 4), value=pad)
    with torch.no_grad():
        torch.testing.assert_close(jepa_loss(model, batch), jepa_loss(model, padded),
                                   atol=2e-6, rtol=2e-4)


def test_accumulation_matches_a_single_batch_for_different_answer_lengths():
    from finetune_gemma4 import accumulation_weights
    batch = tiny_batch()
    microbatches = [{key: value[i:i+1] for key, value in batch.items()} for i in range(2)]
    whole, accumulated = tiny_model(), tiny_model()
    (lm_loss(whole, batch) + 0.1*jepa_loss(whole, batch)).backward()
    for microbatch, (ce_weight, jepa_weight) in zip(microbatches, accumulation_weights(microbatches)):
        (ce_weight*lm_loss(accumulated, microbatch) +
         0.1*jepa_weight*jepa_loss(accumulated, microbatch)).backward()
    for (_, a), (_, b) in zip(whole.named_parameters(), accumulated.named_parameters()):
        if a.requires_grad:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-4)


def test_text_loader_reads_multimodal_checkpoint_without_random_text_weights(tmp_path):
    from safetensors.torch import save_file
    from transformers import Gemma4Config
    from finetune_gemma4 import load_text_decoder
    original = tiny_model().get_base_model()
    # Use a fresh decoder without LoRA as a checkpoint with the official prefix.
    original = Gemma4ForCausalLM(original.config).eval()
    Gemma4Config(text_config=original.config.to_dict()).save_pretrained(tmp_path)
    weights = {name.replace("model.", "model.language_model.", 1): value.clone()
               for name, value in original.state_dict().items() if name != "lm_head.weight"}
    weights["model.vision_tower.unused.weight"] = torch.ones(2, 2)
    save_file(weights, str(tmp_path / "model.safetensors"))
    loaded = load_text_decoder(str(tmp_path), dtype=torch.float32).eval()
    assert isinstance(loaded, Gemma4ForCausalLM)
    assert not hasattr(loaded.model, "vision_tower")
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)
    ids = torch.tensor([[2, 3, 4]])
    with torch.no_grad():
        torch.testing.assert_close(original(ids).logits, loaded(ids).logits)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_bf16_lora_gpu_optimizer_step_keeps_frozen_embeddings_bf16():
    base = Gemma4ForCausalLM(tiny_model().config).to("cuda", dtype=torch.bfloat16)
    model = add_lora(base, rank=2, dropout=0.0)
    batch = {name: value.cuda() for name, value in tiny_batch().items()}
    parameters = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in parameters]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ce = lm_loss(model, batch)
    ce.backward()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        distance = jepa_loss(model, batch)
    (0.1*distance).backward()
    assert torch.isfinite(ce) and torch.isfinite(distance)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, parameters))
    assert all(p.dtype == torch.bfloat16 and not p.requires_grad
               for n, p in model.named_parameters() if "embed_tokens" in n)
