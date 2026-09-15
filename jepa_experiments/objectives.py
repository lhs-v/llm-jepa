"""Eight recipe definitions and loss computation, independent of chat/model layout."""
from dataclasses import dataclass
from collections.abc import Mapping

import torch
import torch.nn.functional as F


RECIPES = {
    "sft": (("direct",), None),
    "json_jepa": (("direct",), "json"),
    "rationale_jepa": (("direct",), "rationale"),
    "multitask": (("direct", "rationale"), None),
    "multitask_jepa": (("direct", "rationale"), "rationale"),
    "cot": (("sequential",), None),
    "cot_jepa": (("sequential",), "rationale"),
    "mixed_jepa": (("direct", "sequential"), "rationale"),
}


@dataclass(frozen=True)
class Recipe:
    name: str
    ce_weights: Mapping[str, float]
    jepa_target: str | None
    jepa_weight: float

    @property
    def requires_rationale(self):
        return ("rationale" in self.ce_weights or "sequential" in self.ce_weights
                or (self.jepa_weight > 0 and self.jepa_target == "rationale"))


def resolve_recipe(experiment):
    name = experiment["recipe"]
    if name not in RECIPES:
        raise ValueError(f"Unknown recipe {name!r}; choose from {', '.join(RECIPES)}")
    branches, target = RECIPES[name]
    weights = {"direct": 1.0, "rationale": experiment.get("rationale_weight", 0.5),
               "sequential": experiment.get("sequential_weight", 1.0)}
    active = {key: float(weights[key]) for key in branches if weights[key] > 0}
    if not active:
        raise ValueError("At least one generation loss must have a positive weight")
    return Recipe(name, active, target, float(experiment.get("jepa_weight", 0.1)) if target else 0.0)


def build_features(examples, formatter, recipe, max_length, overlength="error"):
    """Build only active views; reject or explicitly report whole overlength rows."""
    rows, skipped, max_lengths = [], [], {}
    for example in examples:
        row = {"id": example.id}
        for branch in recipe.ce_weights:
            row[branch] = formatter.encode_supervision(example, branch)
        if recipe.jepa_weight:
            row["view_x"] = formatter.encode_view(formatter.input_text(example))
            target = example.target_text if recipe.jepa_target == "json" else example.rationale
            if not target:
                raise ValueError(f"Example {example.id!r} requires a nonempty rationale")
            row["view_target"] = formatter.encode_view(target)
        lengths = {key: len(value["input_ids"] if isinstance(value, dict) else value)
                   for key, value in row.items() if key != "id"}
        for key, length in lengths.items():
            max_lengths[key] = max(max_lengths.get(key, 0), length)
        if max(lengths.values()) > max_length:
            if overlength == "skip":
                skipped.append({"id": example.id, "lengths": lengths})
                continue
            raise ValueError(f"Example {example.id!r} exceeds max_length={max_length}: {lengths}")
        rows.append(row)
    if not rows:
        raise ValueError("No usable examples remain after length validation")
    return rows, {"selected_ids": [row["id"] for row in rows], "skipped_overlength": skipped,
                  "max_observed_lengths": max_lengths, "selected_examples": len(rows)}


def collate_features(rows, pad_token_id):
    result = {}
    keys = [key for key in rows[0] if key != "id"]
    if any(set(row) - {"id"} != set(keys) for row in rows):
        raise ValueError("A batch must contain the same active recipe branches")
    for key in keys:
        supervised = isinstance(rows[0][key], dict)
        sequences = [row[key]["input_ids"] if supervised else row[key] for row in rows]
        width = max(map(len, sequences))
        if any(not ids for ids in sequences):
            raise ValueError("Empty model inputs are not supported")
        result[key] = {
            "input_ids": torch.tensor([ids + [pad_token_id]*(width-len(ids)) for ids in sequences]),
            "attention_mask": torch.tensor([[1]*len(ids)+[0]*(width-len(ids)) for ids in sequences]),
        }
        if supervised:
            labels = [row[key]["labels"] for row in rows]
            result[key]["labels"] = torch.tensor([ids+[-100]*(width-len(ids)) for ids in labels])
    return result


def group_normalizers(batches, recipe):
    totals = {branch: sum(int((batch[branch]["labels"][:, 1:] != -100).sum())
                          for batch in batches) for branch in recipe.ce_weights}
    if recipe.jepa_weight:
        totals["jepa"] = sum(batch["view_x"]["input_ids"].shape[0] for batch in batches)
    if any(value <= 0 for value in totals.values()):
        raise ValueError("Every active loss needs at least one supervised token/example")
    return totals


def iter_losses(model, batch, recipe, normalizers=None, decoder_path=None):
    """Yield (name, normalized component, weighted loss) for sequential backward.

    Calling backward before requesting the next item releases each CE graph before
    the two JEPA views are built. No optimizer step may occur between components.
    CE is normalized by shifted target tokens and JEPA by examples in the complete
    accumulation group, so uneven sequence lengths and partial batches are correct.
    """
    totals = normalizers if normalizers is not None else group_normalizers([batch], recipe)
    for branch, weight in recipe.ce_weights.items():
        inputs = batch[branch]
        count = int((inputs["labels"][:, 1:] != -100).sum())
        loss = model(**inputs, use_cache=False).loss
        normalized = loss * (count / totals[branch])
        yield branch, normalized, weight*normalized
    if recipe.jepa_weight:
        from .models import get_decoder
        decoder = get_decoder(model, decoder_path)
        embeddings = []
        for branch in ("view_x", "view_target"):
            inputs = batch[branch]
            hidden = decoder(**inputs, use_cache=False).last_hidden_state
            index = inputs["attention_mask"].sum(-1)-1
            embeddings.append(hidden[torch.arange(hidden.shape[0], device=hidden.device), index].float())
        distance = 1-F.cosine_similarity(*embeddings, dim=-1).mean()
        normalized = distance * (embeddings[0].shape[0] / totals["jepa"])
        yield "jepa", normalized, recipe.jepa_weight*normalized
