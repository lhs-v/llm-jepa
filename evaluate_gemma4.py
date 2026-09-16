"""Evaluate a Gemma 4 base model or PEFT adapter with exact match scoring."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


DEFAULT_MODEL = "google/gemma-4-E2B-it"


@dataclass(frozen=True)
class ModelSettings:
    model_name_or_path: str
    revision: str | None
    cache_dir: str | None
    quantization: str
    run_config_path: Path | None = None


def render_prompt(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> str:
    """Render one optional system message followed by one user message."""
    roles = [message.get("role") for message in messages]
    if roles not in (["user"], ["system", "user"]):
        raise ValueError("prompt messages must be [user] or [system, user]")
    if any(not isinstance(message.get("content"), str) for message in messages):
        raise ValueError("every prompt message must have string content")

    return tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _find_run_config(adapter_dir: Path) -> Path:
    candidates = [
        adapter_dir / "run_config.json",
        adapter_dir.parent / "run_config.json",
        adapter_dir.parent.parent / "run_config.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No run_config.json found in {adapter_dir} or either parent directory"
    )


def resolve_model_settings(
    *,
    adapter_dir: Path | None,
    model_name_or_path: str | None,
    revision: str | None,
    cache_dir: str | None,
    quantization: str | None,
) -> ModelSettings:
    """Resolve the base model settings, preferring an adapter's training metadata."""
    run_config_path = None
    run_config: dict[str, Any] = {}
    if adapter_dir is not None:
        run_config_path = _find_run_config(adapter_dir)
        with run_config_path.open(encoding="utf-8") as handle:
            run_config = json.load(handle)

    saved_model = run_config.get("model_name_or_path")
    saved_revision = run_config.get("resolved_model_revision") or run_config.get("revision")
    saved_quantization = run_config.get("quantization")

    if model_name_or_path is not None and saved_model is not None and model_name_or_path != saved_model:
        raise ValueError(
            f"--model-name-or-path {model_name_or_path!r} does not match adapter base model {saved_model!r}"
        )
    if revision is not None and saved_revision is not None and revision != saved_revision:
        raise ValueError(f"--revision {revision!r} does not match adapter revision {saved_revision!r}")
    if quantization is not None and saved_quantization is not None and quantization != saved_quantization:
        raise ValueError(
            f"--quantization {quantization!r} does not match adapter quantization {saved_quantization!r}"
        )

    resolved_model = saved_model or model_name_or_path or DEFAULT_MODEL
    resolved_revision = saved_revision if saved_revision is not None else revision
    resolved_quantization = saved_quantization or quantization or "none"
    if resolved_quantization not in {"4bit", "none"}:
        raise ValueError(f"Unsupported quantization setting: {resolved_quantization!r}")

    return ModelSettings(
        model_name_or_path=resolved_model,
        revision=resolved_revision,
        cache_dir=cache_dir,
        quantization=resolved_quantization,
        run_config_path=run_config_path,
    )


def load_evaluation_components(
    settings: ModelSettings, adapter_dir: Path | None
) -> tuple[Any, Any]:
    """Load the shared Gemma base model/tokenizer and optionally attach a PEFT adapter."""
    from finetune_gemma4 import load_base_model, load_tokenizer

    model = load_base_model(
        settings.model_name_or_path,
        revision=settings.revision,
        cache_dir=settings.cache_dir,
        quantization=settings.quantization,
    )
    tokenizer = load_tokenizer(
        settings.model_name_or_path,
        revision=settings.revision,
        cache_dir=settings.cache_dir,
    )
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def _input_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None and torch.device(device).type != "meta":
        return torch.device(device)
    return next(model.parameters()).device


def generate_response(
    model: Any,
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    max_new_tokens: int,
) -> tuple[str, str]:
    """Generate a deterministic response while retaining the model's stop-token config."""
    prompt = render_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    device = _input_device(model)
    inputs = {name: value.to(device) for name, value in inputs.items()}

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    if tokenizer.pad_token_id is not None:
        generate_kwargs["pad_token_id"] = tokenizer.pad_token_id

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)
    new_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    return prompt, tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def _parse_record(record: Mapping[str, Any], line_number: int) -> tuple[list[dict[str, str]], str | None]:
    raw_messages = record.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError(f"line {line_number}: expected a messages list")

    messages: list[dict[str, str]] = []
    for message in raw_messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ValueError(f"line {line_number}: invalid message")
        if not isinstance(message.get("content"), str):
            raise ValueError(f"line {line_number}: message content must be a string")
        messages.append({"role": message["role"], "content": message["content"]})

    assistant_indices = [index for index, message in enumerate(messages) if message["role"] == "assistant"]
    if len(assistant_indices) > 1:
        raise ValueError(f"line {line_number}: only one assistant reference is supported")
    if assistant_indices and assistant_indices[0] != len(messages) - 1:
        raise ValueError(f"line {line_number}: assistant reference must be the final message")

    reference = messages[-1]["content"].strip() if assistant_indices else None
    prompt_messages = messages[:-1] if assistant_indices else messages
    # Validate the exact prompt shape before loading a large model.
    roles = [message["role"] for message in prompt_messages]
    if roles not in (["user"], ["system", "user"]):
        raise ValueError(f"line {line_number}: prompt messages must be [user] or [system, user]")
    return prompt_messages, reference


def read_jsonl_examples(path: Path, max_examples: int | None = None) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number}: expected a JSON object")
            messages, reference = _parse_record(record, line_number)
            examples.append({"messages": messages, "reference": reference})
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples


def evaluate_examples(
    model: Any,
    tokenizer: Any,
    examples: Iterable[Mapping[str, Any]],
    *,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    exact_matches = 0
    scored_examples = 0

    for index, example in enumerate(examples):
        messages = example["messages"]
        reference = example.get("reference")
        prompt, generation = generate_response(
            model, tokenizer, messages, max_new_tokens=max_new_tokens
        )
        is_match = None
        if reference is not None:
            scored_examples += 1
            is_match = generation == reference.strip()
            exact_matches += int(is_match)
        results.append(
            {
                "index": index,
                "messages": messages,
                "prompt": prompt,
                "reference": reference,
                "generation": generation,
                "exact_match": is_match,
            }
        )

    metrics = {
        "examples": len(results),
        "scored_examples": scored_examples,
        "exact_matches": exact_matches,
        "exact_match": exact_matches / scored_examples if scored_examples else None,
    }
    return results, metrics


def write_results(
    output_dir: Path,
    results: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    settings: ModelSettings,
    adapter_dir: Path | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "generations.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    metadata = dict(metrics)
    metadata.update(
        {
            "model_name_or_path": settings.model_name_or_path,
            "revision": settings.revision,
            "quantization": settings.quantization,
            "adapter_dir": str(adapter_dir) if adapter_dir is not None else None,
            "training_run_config": (
                str(settings.run_config_path) if settings.run_config_path is not None else None
            ),
        }
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-file", type=Path, help="JSONL with system/user/assistant messages")
    source.add_argument("--prompt", help="Generate one response to a user prompt")
    parser.add_argument("--system-prompt", help="Optional system prompt used with --prompt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument("--quantization", choices=("4bit", "none"))
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    if args.adapter_dir is not None and not args.adapter_dir.is_dir():
        parser.error(f"adapter directory does not exist: {args.adapter_dir}")
    if args.input_file is not None and not args.input_file.is_file():
        parser.error(f"input file does not exist: {args.input_file}")
    if args.system_prompt is not None and args.prompt is None:
        parser.error("--system-prompt requires --prompt")
    if args.max_examples is not None and args.max_examples <= 0:
        parser.error("--max-examples must be positive")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    try:
        settings = resolve_model_settings(
            adapter_dir=args.adapter_dir,
            model_name_or_path=args.model_name_or_path,
            revision=args.revision,
            cache_dir=args.cache_dir,
            quantization=args.quantization,
        )
        if args.input_file is not None:
            examples = read_jsonl_examples(args.input_file, args.max_examples)
        else:
            messages = []
            if args.system_prompt is not None:
                messages.append({"role": "system", "content": args.system_prompt})
            messages.append({"role": "user", "content": args.prompt})
            examples = [{"messages": messages, "reference": None}]
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    model, tokenizer = load_evaluation_components(settings, args.adapter_dir)
    results, metrics = evaluate_examples(
        model,
        tokenizer,
        examples,
        max_new_tokens=args.max_new_tokens,
    )
    write_results(args.output_dir, results, metrics, settings, args.adapter_dir)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
