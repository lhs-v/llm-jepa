"""Portable helper behavior with real tiny Gemma 4 models and LoRA."""

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    Gemma4Config,
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
    Gemma4TextConfig,
)

from src.trainers.jepa_loss import encode_jepa_view, jepa_loss_per_example


@pytest.fixture(scope="module", autouse=True)
def small_model_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model(conditional=False, checkpointing=False):
    torch.manual_seed(7)
    config = Gemma4TextConfig(
        vocab_size=64, vocab_size_per_layer_input=64, hidden_size=16,
        intermediate_size=32, num_hidden_layers=4, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, hidden_size_per_layer_input=4,
        num_kv_shared_layers=2, layer_types=["sliding_attention", "full_attention"] * 2,
        per_layer_config={}, sliding_window=8, max_position_embeddings=64,
        attention_dropout=0.0, use_cache=False,
    )
    base = (
        Gemma4ForConditionalGeneration(Gemma4Config(text_config=config))
        if conditional else Gemma4ForCausalLM(config)
    )
    model = get_peft_model(base, LoraConfig(
        task_type="CAUSAL_LM", r=2, lora_alpha=4, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def test_cosine_geometry_and_fp32_loss_from_bfloat16():
    context = torch.tensor([[1, 0], [1, 0], [1, 0]], dtype=torch.bfloat16)
    rationale = torch.tensor([[1, 0], [0, 1], [-1, 0]], dtype=torch.bfloat16)
    loss = jepa_loss_per_example(context, rationale)
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, torch.tensor([0.0, 1.0, 2.0]))


@pytest.mark.parametrize("left,right", [
    (torch.zeros(3), torch.zeros(3)),
    (torch.zeros(2, 3), torch.zeros(1, 3)),
    (torch.zeros(2, 3), torch.zeros(2, 4)),
])
def test_invalid_representation_shapes_are_rejected(left, right):
    with pytest.raises(ValueError, match="shape"):
        jepa_loss_per_example(left, right)


@pytest.mark.parametrize("conditional", [False, True])
def test_representation_uses_last_real_token_for_both_padding_sides(conditional):
    model = tiny_model(conditional=conditional).eval()
    expected = model(
        input_ids=torch.tensor([[2, 7, 8]]), output_hidden_states=True,
    ).hidden_states[-1][:, -1]
    ids = torch.tensor([[2, 7, 8, 0, 0], [0, 0, 2, 7, 8]])
    mask = torch.tensor([[1, 1, 1, 0, 0], [0, 0, 1, 1, 1]])
    original_ids, original_mask = ids.clone(), mask.clone()
    actual = encode_jepa_view(model, ids, mask)
    torch.testing.assert_close(actual, expected.expand(2, -1))
    assert torch.equal(ids, original_ids) and torch.equal(mask, original_mask)


def test_all_padding_row_is_rejected():
    with pytest.raises(ValueError, match="valid token"):
        encode_jepa_view(tiny_model(), torch.tensor([[0, 0]]), torch.tensor([[0, 0]]))


@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_both_views_receive_gradients_and_only_lora_parameters_train(conditional, checkpointing):
    model = tiny_model(conditional=conditional, checkpointing=checkpointing).train()
    context = encode_jepa_view(
        model, torch.tensor([[2, 7, 8], [2, 4, 0]]), torch.tensor([[1, 1, 1], [1, 1, 0]]),
    )
    rationale = encode_jepa_view(
        model, torch.tensor([[2, 9, 10, 11], [2, 5, 6, 0]]),
        torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
    )
    context.retain_grad()
    rationale.retain_grad()
    loss = jepa_loss_per_example(context, rationale).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert context.grad is not None and context.grad.abs().sum() > 0
    assert rationale.grad is not None and rationale.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for name, p in model.named_parameters() if "lora_" in name)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
