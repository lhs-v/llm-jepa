"""Differentiable JEPA helpers for Gemma 4 SFT with independent text views.

Copy this file into the target project's src/trainers/ directory. Pass the
wrapped model received by Trainer.compute_loss to encode_jepa_view. Both views
keep their gradients; loss weighting and gradient accumulation belong to the
calling trainer. The model must support output_hidden_states and logits_to_keep.
"""

import torch
import torch.nn.functional as F


def jepa_loss_per_example(situation_repr, rationale_repr):
    """Return FP32 cosine distances [batch] for matching [batch, hidden] views."""
    if situation_repr.ndim != 2 or rationale_repr.ndim != 2:
        raise ValueError("JEPA representations must have shape [batch, hidden].")
    if situation_repr.shape != rationale_repr.shape:
        raise ValueError("JEPA representations must have matching shapes.")
    return 1.0 - F.cosine_similarity(
        situation_repr.float(), rationale_repr.float(), dim=-1
    )


def encode_jepa_view(model, input_ids, attention_mask):
    """Encode [batch, length] text and select its last nonpadding hidden state.

    attention_mask must contain 1 for real tokens and 0 for padding. Explicit
    positions support either padding side. The result has shape [batch, hidden]
    and retains the model's output dtype and autograd graph.
    """
    valid = attention_mask.bool()
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    last_index = (
        positions.unsqueeze(0)
        .expand_as(input_ids)
        .masked_fill(~valid, -1)
        .amax(dim=1)
    )
    if (last_index < 0).any():
        raise ValueError("JEPA input must contain at least one valid token")

    position_ids = attention_mask.long().cumsum(dim=1) - 1
    position_ids = position_ids.masked_fill(~valid, 0)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
        logits_to_keep=1,
    )
    hidden = outputs.hidden_states[-1]
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, last_index]
