"""Configurable intervention experiments. See HANDOFF.md for recipes and handoff."""
import argparse
import json
from pathlib import Path

from jepa_experiments.config import apply_overrides, load_config
from jepa_experiments.objectives import RECIPES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON config; paths resolve relative to this file")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--recipe", choices=RECIPES)
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--prepare-only", action="store_true", help="Validate/tokenize without model weights or run output")
    parser.add_argument("--list-recipes", action="store_true")
    args = parser.parse_args(argv)
    if args.list_recipes:
        for name, (branches, target) in RECIPES.items():
            print(f"{name}: CE={'+'.join(branches)}, JEPA target={target or 'none'}")
        return
    if not args.config:
        parser.error("--config is required unless --list-recipes is used")
    overrides = list(args.set)
    output_dir = str(Path(args.output_dir).expanduser().resolve()) if args.output_dir else None
    for key, value in (("experiment.recipe", args.recipe), ("training.output_dir", output_dir),
                       ("training.max_steps", args.max_steps), ("data.max_samples", args.max_samples)):
        if value is not None:
            overrides.append(f"{key}={json.dumps(value)}")
    config = load_config(args.config, overrides=apply_overrides({}, overrides))
    from jepa_experiments.training import prepare, train
    result = prepare(config) if args.prepare_only else train(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
