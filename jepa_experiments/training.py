"""Single-device LoRA training with explicit, independently normalized objectives."""
import copy
from contextlib import nullcontext
from functools import partial
import hashlib
from importlib.metadata import PackageNotFoundError, version
from itertools import islice
import json
import math
from pathlib import Path
import platform
import time

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, set_seed

from .data import load_examples
from .formatting import Formatter
from .models import load_model, load_tokenizer
from .objectives import build_features, collate_features, group_normalizers, iter_losses, resolve_recipe


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _features(config, examples, tokenizer, recipe):
    formatter = Formatter(tokenizer, chat_format=config["model"]["chat_format"],
                          prompts=config["prompts"], pooling=config["model"]["pooling"])
    return build_features(examples, formatter, recipe, config["data"]["max_length"], config["data"]["overlength"])


def prepare(config):
    """Validate data and tokenize all active branches without loading model weights."""
    recipe = resolve_recipe(config["experiment"])
    examples, source = load_examples(config["data"], require_rationale=recipe.requires_rationale)
    tokenizer = load_tokenizer(config["model"])
    _, features = _features(config, examples, tokenizer, recipe)
    return {"recipe": recipe.name, "ce_weights": dict(recipe.ce_weights),
            "jepa_target": recipe.jepa_target, "jepa_weight": recipe.jepa_weight,
            "source": source, "features": features}


def runtime_metadata():
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "jsonschema"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "jepa_experiments").glob("*.py")) + [
        root / "train_experiment.py", root / "evaluate_experiment.py", root / "finetune_gemma4.py"]
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "cuda_runtime": torch.version.cuda,
            "source_hashes": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in files if path.exists()}}


def _save_adapter(directory, model, tokenizer, config, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    adapter = directory / "adapter"
    model.save_pretrained(adapter, save_embedding_layers=False)
    tokenizer.save_pretrained(adapter)
    write_json(directory / "resolved_config.json", config)
    write_json(directory / "model_metadata.json", metadata)


def train(config):
    """Train a fresh adapter. Checkpoints are inference snapshots, not resume states."""
    config = copy.deepcopy(config)
    options = config["training"]
    output = Path(options["output_dir"])
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output must be a new or empty directory: {output}")
    recipe = resolve_recipe(config["experiment"])
    examples, source = load_examples(config["data"], require_rationale=recipe.requires_rationale)
    set_seed(options["seed"])
    model, tokenizer, metadata = load_model(config["model"], training=True)
    # Tokenize with the exact tokenizer loaded alongside this base revision.
    features, feature_stats = _features(config, examples, tokenizer, recipe)
    if metadata["resolved_revision"]:
        config["model"]["revision"] = metadata["resolved_revision"]
    elif not Path(config["model"]["model_name_or_path"]).is_dir():
        raise ValueError("Hub model must resolve to an immutable revision before training")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "resolved_config.json", config)
    write_json(output / "model_metadata.json", metadata)
    write_json(output / "dataset_manifest.json", {"source": source, "features": feature_stats})
    runtime = runtime_metadata()
    runtime["resolved_config_sha256"] = hashlib.sha256((output / "resolved_config.json").read_bytes()).hexdigest()
    runtime["optimizer"] = {"name": "AdamW", "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8}
    device = torch.device(config["model"]["device"])
    if device.type == "cuda":
        runtime["gpu"] = torch.cuda.get_device_name(device)
        torch.cuda.reset_peak_memory_stats(device)
    write_json(output / "runtime.json", runtime)
    loader = DataLoader(features, batch_size=options["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(options["seed"]),
                        collate_fn=partial(collate_features, pad_token_id=tokenizer.pad_token_id))
    accumulation = options["gradient_accumulation_steps"]
    total_steps = math.ceil(len(loader) / accumulation) * options["num_epochs"]
    if options["max_steps"] is not None:
        total_steps = min(total_steps, options["max_steps"])
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=options["learning_rate"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * options["warmup_ratio"]), num_training_steps=total_steps)
    use_autocast = device.type == "cuda" and config["model"]["dtype"] == "bfloat16"
    steps = examples_seen = 0
    first_gradient = None
    started = time.perf_counter()
    model.train()
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(options["num_epochs"]):
            iterator = iter(loader)
            while steps < total_steps:
                group = list(islice(iterator, accumulation))
                if not group:
                    break
                normalizers = group_normalizers(group, recipe)
                group_examples = sum(next(iter(batch.values()))["input_ids"].shape[0] for batch in group)
                optimizer.zero_grad(set_to_none=True)
                components, total_loss = {}, 0.0
                for cpu_batch in group:
                    batch = {name: {key: value.to(device) for key, value in inputs.items()}
                             for name, inputs in cpu_batch.items()}
                    losses = iter_losses(model, batch, recipe, normalizers, config["model"]["decoder_path"])
                    while True:
                        with torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast else nullcontext():
                            item = next(losses, None)
                        if item is None:
                            break
                        name, component, weighted = item
                        if not torch.isfinite(weighted):
                            raise FloatingPointError(f"Non-finite {name} loss before optimizer step {steps + 1}")
                        components[name] = components.get(name, 0.0) + float(component.detach())
                        total_loss += float(weighted.detach())
                        weighted.backward()
                        del item, component, weighted
                norm = float(torch.nn.utils.clip_grad_norm_(parameters, options["max_grad_norm"], error_if_nonfinite=True))
                if first_gradient is None:
                    first_gradient = norm
                    if norm == 0:
                        raise RuntimeError("No adapter gradient in the first optimizer step")
                lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                steps += 1
                examples_seen += group_examples
                record = {"step": steps, "epoch": epoch + 1, "examples": group_examples,
                          "examples_seen": examples_seen, "normalizers": normalizers,
                          "components": components, "loss": total_loss, "gradient_norm": norm,
                          "learning_rate": lr, "elapsed_seconds": time.perf_counter() - started}
                if device.type == "cuda":
                    record["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                if options["save_every_steps"] and steps % options["save_every_steps"] == 0:
                    _save_adapter(output / "checkpoints" / f"step-{steps:06d}", model, tokenizer, config, metadata)
            if steps >= total_steps:
                break
    _save_adapter(output, model, tokenizer, config, metadata)
    result = {"status": "complete", "optimizer_steps": steps, "examples_seen": examples_seen,
              "first_gradient_norm": first_gradient, "elapsed_seconds": time.perf_counter() - started,
              "recipe": recipe.name, "adapter_dir": str(output / "adapter")}
    if device.type == "cuda":
        result["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
    write_json(output / "completed.json", result)
    return result
