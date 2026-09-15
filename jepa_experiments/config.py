"""Validated JSON configuration shared by training and evaluation."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from string import Formatter


DEFAULT_MODEL = "google/gemma-4-E2B-it"
DEFAULT_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
RECIPES = ("sft", "json_jepa", "rationale_jepa", "multitask", "multitask_jepa", "cot", "cot_jepa", "mixed_jepa")

DEFAULT_CONFIG = {
    "schema_version": 1,
    "experiment": {"name": "intervention", "recipe": "rationale_jepa", "jepa_weight": 0.1, "rationale_weight": 0.5, "sequential_weight": 1.0},
    "model": {
        "model_name_or_path": DEFAULT_MODEL, "revision": None,
        "backend": "gemma4", "chat_format": "gemma4", "pooling": "turn_end",
        "quantization": "none", "dtype": "bfloat16", "device": "cuda:0",
        "attn_implementation": "sdpa", "cache_dir": ".cache/huggingface",
        "lora_rank": 64, "lora_alpha": 128, "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "gradient_checkpointing": True, "decoder_path": None,
    },
    "data": {
        "train_file": None, "eval_file": None, "format": "records",
        "fields": {"id": "id", "situation": "situation", "policy": "policy", "rationale": "rationale", "target": "target"},
        "default_policy": None, "target_schema": None,
        "max_length": 512, "max_samples": None, "overlength": "error",
    },
    "prompts": {
        "system": "Decide whether AI should intervene. Follow the supplied policy. Return only the requested JSON object.",
        "rationale_system": "Explain the facts and policy rules relevant to the intervention decision. Return only a concise rationale.",
        "input_template": "Policy:\n{policy}\n\nSituation:\n{situation}",
    },
    "training": {
        "output_dir": "outputs/intervention", "seed": 42, "batch_size": 4,
        "gradient_accumulation_steps": 4, "learning_rate": 2e-5, "num_epochs": 4,
        "max_steps": None, "warmup_ratio": 0.03, "max_grad_norm": 1.0, "save_every_steps": 0,
    },
    "evaluation": {"thinking": "off", "max_new_tokens": 256, "max_samples": None, "decision_field": "intervene"},
}


def _merge(base: dict, incoming: dict, path: str = "") -> dict:
    if not isinstance(incoming, dict):
        raise ValueError(f"{path or 'config'} must be an object")
    result = copy.deepcopy(base)
    for key, value in incoming.items():
        field = f"{path}.{key}" if path else str(key)
        if key not in base:
            raise ValueError(f"Unknown configuration field: {field}")
        result[key] = _merge(base[key], value, field) if isinstance(base[key], dict) else copy.deepcopy(value)
    return result


def _reject_constant(value: str):
    raise ValueError(f"Non-finite JSON number is not allowed: {value}")


def load_config(path, overrides: dict | None = None) -> dict:
    """Read a JSON config and resolve relative paths against its directory."""
    config_path = Path(path).expanduser().resolve()
    try:
        config = json.loads(config_path.read_text(encoding="utf-8-sig"), parse_constant=_reject_constant)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot load config {config_path}: {exc}") from exc
    config = _merge(DEFAULT_CONFIG, config)
    if overrides is not None:
        model_override = overrides.get("model", {}) if isinstance(overrides, dict) else {}
        if isinstance(model_override, dict) and "model_name_or_path" in model_override and "revision" not in model_override and model_override["model_name_or_path"] != config["model"]["model_name_or_path"]:
            config["model"]["revision"] = None
        config = _merge(config, overrides)
    return validate_config(config, base_dir=config_path.parent)


def apply_overrides(config: dict, items: list[str]) -> dict:
    """Apply dotted ``key=value`` overrides without modifying the input dict."""
    _merge(DEFAULT_CONFIG, config)
    result = copy.deepcopy(config)
    explicit_revision = False
    for item in items:
        if not isinstance(item, str) or "=" not in item:
            raise ValueError(f"Override must use key=value syntax: {item!r}")
        key, raw = item.split("=", 1)
        parts = key.split(".")
        schema = DEFAULT_CONFIG
        for part in parts:
            if not part or not isinstance(schema, dict) or part not in schema:
                raise ValueError(f"Unknown or invalid override field: {key}")
            schema = schema[part]
        try:
            value = json.loads(raw, parse_constant=_reject_constant)
        except json.JSONDecodeError:
            value = raw
        if isinstance(schema, dict):
            _merge(schema, value, key)
        if key == "model.revision" or (key == "model" and "revision" in value):
            explicit_revision = True
        destination = result
        for part in parts[:-1]:
            destination = destination.setdefault(part, {})
            if not isinstance(destination, dict):
                raise ValueError(f"Cannot override nested field: {key}")
        if isinstance(schema, dict):
            current = destination.get(parts[-1], {})
            # The schema validates names, while the merge below retains partial defaults.
            destination[parts[-1]] = _merge(_merge(schema, current), value, key)
        else:
            destination[parts[-1]] = value
    previous_model = config.get("model", {}).get("model_name_or_path", DEFAULT_MODEL)
    current_model = result.get("model", {}).get("model_name_or_path", DEFAULT_MODEL)
    if current_model != previous_model and not explicit_revision:
        result.setdefault("model", {})["revision"] = None
    return result


def _text(value, name: str, nullable: bool = False):
    if value is None and nullable:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string" + (" or null" if nullable else ""))


def _number(value, name: str, minimum=0, inclusive=False, integer=False, nullable=False):
    if value is None and nullable:
        return
    valid_type = type(value) is int if integer else type(value) in (int, float)
    if not valid_type or not math.isfinite(value) or (value < minimum if inclusive else value <= minimum):
        relation = ">=" if inclusive else ">"
        raise ValueError(f"{name} must be a finite {'integer' if integer else 'number'} {relation} {minimum}" + (" or null" if nullable else ""))


def _enum(value, name, choices):
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{name} must be one of: {', '.join(choices)}")


def _path(value: str, base_dir: Path) -> str:
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else base_dir / path).resolve())


def validate_config(config: dict, base_dir: Path | None = None) -> dict:
    """Deep-merge defaults, reject invalid fields, and return a fresh config."""
    result = _merge(DEFAULT_CONFIG, config)
    if type(result["schema_version"]) is not int or result["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    experiment, model, data = result["experiment"], result["model"], result["data"]
    training, evaluation = result["training"], result["evaluation"]
    _text(experiment["name"], "experiment.name")
    _enum(experiment["recipe"], "experiment.recipe", RECIPES)
    for field in ("jepa_weight", "rationale_weight", "sequential_weight"):
        _number(experiment[field], f"experiment.{field}", inclusive=True)
    for field in ("model_name_or_path", "device", "cache_dir"):
        _text(model[field], f"model.{field}")
    for field in ("revision", "decoder_path"):
        _text(model[field], f"model.{field}", nullable=True)
    for field, choices in {
        "backend": ("gemma4", "auto_causal_lm"), "chat_format": ("gemma4", "standard"),
        "pooling": ("turn_end", "last_nonpad"), "quantization": ("none", "4bit"),
        "dtype": ("bfloat16", "float32"), "attn_implementation": ("sdpa", "eager", "flash_attention_2", "flash_attention_3"),
    }.items():
        _enum(model[field], f"model.{field}", choices)
    _number(model["lora_rank"], "model.lora_rank", integer=True)
    _number(model["lora_alpha"], "model.lora_alpha")
    _number(model["lora_dropout"], "model.lora_dropout", inclusive=True)
    if model["lora_dropout"] >= 1:
        raise ValueError("model.lora_dropout must be less than 1")
    if type(model["gradient_checkpointing"]) is not bool:
        raise ValueError("model.gradient_checkpointing must be a boolean")
    modules = model["target_modules"]
    if not isinstance(modules, list) or not modules:
        raise ValueError("model.target_modules must be a nonempty array of module names")
    for module in modules:
        _text(module, "model.target_modules entry")
    if len(set(modules)) != len(modules):
        raise ValueError("model.target_modules must not contain duplicates")

    _enum(data["format"], "data.format", ("records", "messages"))
    _enum(data["overlength"], "data.overlength", ("error", "skip"))
    for field in ("train_file", "eval_file", "target_schema", "default_policy"):
        _text(data[field], f"data.{field}", nullable=True)
    for field, value in data["fields"].items():
        _text(value, f"data.fields.{field}", nullable=field in ("id", "rationale", "policy"))
        if value is not None and any(not part for part in value.split(".")):
            raise ValueError(f"data.fields.{field} must be a valid dotted path")
    _number(data["max_length"], "data.max_length", integer=True)
    _number(data["max_samples"], "data.max_samples", integer=True, nullable=True)
    for field, value in result["prompts"].items():
        _text(value, f"prompts.{field}")
    try:
        fields = {name for _, name, _, _ in Formatter().parse(result["prompts"]["input_template"]) if name is not None}
        if fields != {"policy", "situation"}:
            raise ValueError("must contain only {policy} and {situation}, and include both")
        result["prompts"]["input_template"].format(policy="policy", situation="situation")
    except (KeyError, ValueError, IndexError, AttributeError) as exc:
        raise ValueError(f"prompts.input_template is invalid: {exc}") from exc

    _text(training["output_dir"], "training.output_dir")
    for field in ("seed", "save_every_steps"):
        _number(training[field], f"training.{field}", inclusive=True, integer=True)
    for field in ("batch_size", "gradient_accumulation_steps", "num_epochs"):
        _number(training[field], f"training.{field}", integer=True)
    for field in ("learning_rate", "max_grad_norm"):
        _number(training[field], f"training.{field}")
    _number(training["max_steps"], "training.max_steps", integer=True, nullable=True)
    _number(training["warmup_ratio"], "training.warmup_ratio", inclusive=True)
    if training["warmup_ratio"] > 1:
        raise ValueError("training.warmup_ratio must be at most 1")
    _enum(evaluation["thinking"], "evaluation.thinking", ("off", "on"))
    _number(evaluation["max_new_tokens"], "evaluation.max_new_tokens", integer=True)
    _number(evaluation["max_samples"], "evaluation.max_samples", integer=True, nullable=True)
    _text(evaluation["decision_field"], "evaluation.decision_field")
    if model["chat_format"] == "standard" and (experiment["recipe"] in ("cot", "cot_jepa", "mixed_jepa") or evaluation["thinking"] == "on"):
        raise ValueError("Thinking/sequential modes with standard chat_format require a template adapter; use gemma4 or an OFF recipe")
    if model["chat_format"] == "standard" and model["pooling"] != "last_nonpad":
        raise ValueError("standard chat_format requires model.pooling=last_nonpad; turn_end pooling requires a template adapter")

    directory = Path(base_dir or Path.cwd()).expanduser().resolve()
    for field in ("train_file", "eval_file", "target_schema"):
        if data[field] is not None:
            data[field] = _path(data[field], directory)
    training["output_dir"] = _path(training["output_dir"], directory)
    model["cache_dir"] = _path(model["cache_dir"], directory)
    model_id = model["model_name_or_path"]
    if model_id == DEFAULT_MODEL and model["revision"] is None:
        model["revision"] = DEFAULT_REVISION
    if Path(model_id).is_absolute() or model_id.startswith(("./", "../", ".\\", "..\\")):
        model["model_name_or_path"] = _path(model_id, directory)
    return result
