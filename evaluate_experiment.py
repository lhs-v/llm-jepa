"""Evaluate an intervention base model or the adapter saved by train_experiment.py."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic intervention JSON and report syntax, schema, decision, and thought metrics.",
        epilog=("Relative config and --set file paths resolve against the source config directory. "
                "A saved run keeps its model and prompts; override data.*, evaluation.*, "
                "model.device, or model.cache_dir. Output must be a new directory."),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="JSON experiment config; evaluate the base model without an adapter")
    source.add_argument("--run-dir", type=Path, help="Saved training run with resolved_config.json, model_metadata.json, and adapter/")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for predictions.jsonl, metrics.json, and provenance")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="Repeatable dotted configuration override; values accept JSON")
    parser.add_argument("--thinking", choices=("off", "on"), help="Override evaluation.thinking; ON requires a supported thought-channel formatter")
    return parser


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_evaluation_config(
    *,
    config_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    overrides: list[str] | None = None,
) -> tuple[dict[str, Any], Path | None, dict[str, Any]]:
    """Resolve a base config or freeze the exact model identity of a saved adapter."""
    from jepa_experiments.config import apply_overrides, load_config

    if (config_path is None) == (run_dir is None):
        raise ValueError("Specify exactly one of --config or --run-dir")
    items = list(overrides or [])
    adapter_dir = None
    metadata = None
    if run_dir is not None:
        run_dir = Path(run_dir).expanduser().resolve()
        config_path = run_dir / "resolved_config.json"
        adapter_dir = run_dir / "adapter"
        if not adapter_dir.is_dir():
            raise FileNotFoundError(f"Saved run adapter directory is missing: {adapter_dir}")
        metadata_path = run_dir / "model_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Saved run model metadata is missing: {metadata_path}")
        with metadata_path.open(encoding="utf-8-sig") as handle:
            metadata = json.load(handle)
        for item in items:
            key = item.split("=", 1)[0]
            if not (key in {"model.device", "model.cache_dir", "data", "evaluation"}
                    or key.startswith("data.") or key.startswith("evaluation.")):
                raise ValueError(
                    f"Cannot override {key!r} for a saved run. Only data.*, evaluation.*, "
                    "model.device, and model.cache_dir may change; use --config to evaluate another base model."
                )
        saved = load_config(config_path)
        if not isinstance(metadata, dict) or "resolved_revision" not in metadata:
            raise ValueError("Saved model_metadata.json must record resolved_revision")
        revision = metadata["resolved_revision"]
        model_path = saved["model"]["model_name_or_path"]
        local_source = Path(model_path).is_dir() or Path(model_path).is_absolute() or metadata.get("local_source_sha256") is not None
        if local_source:
            from jepa_experiments.models import local_source_fingerprints

            saved_hashes = metadata.get("local_source_sha256")
            if not isinstance(saved_hashes, dict) or not saved_hashes or any(
                not isinstance(name, str) or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                for name, digest in saved_hashes.items()
            ):
                raise ValueError(
                    "Saved local model requires local_source_sha256 in model_metadata.json as immutable "
                    "snapshot provenance; recreate the run from the original immutable local snapshot."
                )
            current_hashes = local_source_fingerprints(model_path)
            if current_hashes != saved_hashes:
                changed = sorted(name for name in saved_hashes.keys() | (current_hashes or {}).keys()
                                 if saved_hashes.get(name) != (current_hashes or {}).get(name))
                raise ValueError(f"Saved local model snapshot changed or is missing: {model_path}; differing artifacts: {changed}")
        elif not (
            isinstance(revision, str) and re.fullmatch(r"[0-9a-fA-F]{40}", revision)
        ):
            raise ValueError("Remote saved models require a commit hash in model_metadata.json resolved_revision")
        original_model = metadata.get("original_model_name_or_path")
        if original_model is not None and original_model != model_path:
            raise ValueError("Saved run model metadata disagrees with resolved_config.json model_name_or_path")
        # Append after allowed user overrides, so saved model identity cannot drift.
        items.append("model.revision=" + json.dumps(revision))
    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path, overrides=apply_overrides({}, items))
    provenance = {
        "source_config": str(config_path),
        "source_config_sha256": _file_hash(config_path),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "adapter_dir": str(adapter_dir) if adapter_dir is not None else None,
        "overrides": list(overrides or []),
    }
    if metadata is not None:
        provenance["saved_model_metadata"] = metadata
    return config, adapter_dir, provenance


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Evaluation output already exists; choose a new --output-dir: {output_dir}")
    overrides = list(args.overrides)
    if args.thinking is not None:
        overrides.append("evaluation.thinking=" + args.thinking)
    config, adapter_dir, provenance = resolve_evaluation_config(
        config_path=args.config, run_dir=args.run_dir, overrides=overrides,
    )

    from jepa_experiments.data import load_examples
    from jepa_experiments.evaluation import evaluate_examples
    from jepa_experiments.formatting import Formatter
    from jepa_experiments.models import load_model

    examples, data_stats = load_examples(config["data"], require_rationale=False, split="eval")
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        **provenance,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "data_stats": data_stats,
    }
    config_output = output_dir / "resolved_config.json"
    manifest_output = output_dir / "evaluation_manifest.json"
    _write_json(config_output, config)
    manifest["resolved_config_sha256"] = _file_hash(config_output)
    _write_json(manifest_output, manifest)
    started = time.perf_counter()
    try:
        model, tokenizer, metadata = load_model(config["model"], training=False, adapter_dir=adapter_dir)
        formatter = Formatter(
            tokenizer, chat_format=config["model"]["chat_format"],
            prompts=config["prompts"], pooling=config["model"]["pooling"],
        )
        eval_config = {**config["evaluation"], "max_length": config["data"]["max_length"], "overlength": config["data"]["overlength"]}
        results, metrics = evaluate_examples(
            model, tokenizer, formatter, examples, eval_config,
            target_schema=config["data"].get("target_schema"),
        )
        metrics.update({
            "model_name_or_path": config["model"]["model_name_or_path"],
            "resolved_revision": metadata.get("resolved_revision"),
            "adapter_dir": str(adapter_dir) if adapter_dir is not None else None,
            "total_elapsed_seconds": time.perf_counter() - started,
        })
        with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in results:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        _write_json(output_dir / "metrics.json", metrics)
        _write_json(output_dir / "model_metadata.json", metadata)
        manifest.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat())
        _write_json(manifest_output, manifest)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}", finished_at=datetime.now(timezone.utc).isoformat())
        _write_json(manifest_output, manifest)
        raise
    print(json.dumps({"output_dir": str(output_dir), "metrics": metrics}, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
