"""Loss routing and gradient correctness on an actual small Gemma 4 decoder."""
import copy

import pytest
import torch
from transformers import Gemma4ForCausalLM, Gemma4TextConfig

from finetune_gemma4 import add_lora
from jepa_experiments.objectives import (
    resolve_recipe, collate_features, group_normalizers, iter_losses,
)


def recipe(name):
    return resolve_recipe({"recipe": name, "jepa_weight": 0.2,
                           "rationale_weight": 0.3, "sequential_weight": 0.4})


def model():
    torch.manual_seed(17)
    config = Gemma4TextConfig(
        vocab_size=64, vocab_size_per_layer_input=64, hidden_size=16,
        hidden_size_per_layer_input=4, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, num_kv_shared_layers=2, sliding_window=8,
        layer_types=["sliding_attention", "full_attention"]*2,
        per_layer_config={}, max_position_embeddings=64,
    )
    return add_lora(Gemma4ForCausalLM(config), rank=2, dropout=0,
                    gradient_checkpointing=False)


def features(spec):
    rows = []
    for index, answer in enumerate(([9, 10, 1], [11, 1], [12, 13, 14, 1])):
        row = {"id": str(index)}
        for branch in spec.ce_weights:
            prompt = [2, 4, 5] if branch != "rationale" else [2, 6]
            target = answer if branch == "direct" else [7] + list(answer)
            row[branch] = {"input_ids": prompt + list(target),
                           "labels": [-100]*len(prompt) + list(target)}
        if spec.jepa_weight:
            row["view_x"] = [2, 4, index+15, 1]
            row["view_target"] = [2, 6] + list(answer)
        rows.append(row)
    return rows


@pytest.mark.parametrize("name,expected", [
    ("sft", {"direct"}), ("json_jepa", {"direct", "jepa"}),
    ("rationale_jepa", {"direct", "jepa"}),
    ("multitask", {"direct", "rationale"}),
    ("multitask_jepa", {"direct", "rationale", "jepa"}),
    ("cot", {"sequential"}), ("cot_jepa", {"sequential", "jepa"}),
    ("mixed_jepa", {"direct", "sequential", "jepa"}),
])
def test_all_eight_recipes_route_real_differentiable_losses(name, expected):
    spec = recipe(name)
    batch = collate_features(features(spec), pad_token_id=0)
    net = model()
    seen = set()
    for name, normalized, weighted in iter_losses(net, batch, spec):
        seen.add(name)
        assert torch.isfinite(normalized)
        weighted.backward()
    assert seen == expected
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())
    assert all(p.grad is None for n, p in net.named_parameters() if "lora_" not in n)


@pytest.mark.parametrize("name", ["multitask_jepa", "mixed_jepa"])
def test_accumulation_matches_full_batch_with_unequal_token_and_example_counts(name):
    spec = recipe(name)
    rows = features(spec)
    whole = collate_features(rows, 0)
    parts = [collate_features(rows[:1], 0), collate_features(rows[1:], 0)]
    left, right = model(), model()
    for _, _, loss in iter_losses(left, whole, spec):
        loss.backward()
    totals = group_normalizers(parts, spec)
    for batch in parts:
        for _, _, loss in iter_losses(right, batch, spec, normalizers=totals):
            loss.backward()
    for (name, a), (_, b) in zip(left.named_parameters(), right.named_parameters()):
        if a.requires_grad:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-4, msg=name)


def test_rationale_jepa_has_no_rationale_generation_and_both_views_receive_gradients():
    spec = recipe("rationale_jepa")
    net = model()
    batch = collate_features(features(spec), 0)
    assert "rationale" not in batch and "sequential" not in batch
    losses = iter_losses(net, batch, spec)
    name, _, ce = next(losses)
    assert name == "direct"
    ce.backward()
    net.zero_grad(set_to_none=True)
    states = []
    def capture(_module, _args, output):
        output.last_hidden_state.retain_grad()
        states.append(output.last_hidden_state)
    handle = net.get_base_model().model.register_forward_hook(capture)
    name, _, alignment = next(losses)
    assert name == "jepa"
    alignment.backward()
    handle.remove()
    assert len(states) == 2
    assert all(s.grad is not None and s.grad.abs().sum() > 0 for s in states)


def test_zero_jepa_coefficient_does_not_require_rationale_or_extra_views():
    spec = resolve_recipe({"recipe": "rationale_jepa", "jepa_weight": 0})
    assert not spec.requires_rationale
    batch = collate_features(features(spec), 0)
    assert set(batch) == {"direct"}
    assert [name for name, _, _ in iter_losses(model(), batch, spec)] == ["direct"]
