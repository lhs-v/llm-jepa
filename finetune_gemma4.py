"""Single-GPU Gemma 4 LoRA with the original LLM-JEPA k=0 objective.

L = assistant-token CE + lambda * (1 - cosine(Enc(Text), Enc(Code))).
Both independently encoded views share the trainable decoder; neither is detached.
"""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
from itertools import islice
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig, AutoTokenizer, BitsAndBytesConfig, Gemma4ForCausalLM,
    get_linear_schedule_with_warmup, set_seed,
)

DEFAULT_MODEL = "google/gemma-4-E2B-it"
DEFAULT_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def resolved_revision(model_name_or_path, revision):
    return revision or (DEFAULT_REVISION if model_name_or_path == DEFAULT_MODEL else None)


def load_tokenizer(model_name_or_path=DEFAULT_MODEL, *, revision=None, cache_dir=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, revision=resolved_revision(model_name_or_path, revision),
        cache_dir=cache_dir,
    )
    tokenizer.padding_side = "right"
    if not tokenizer.chat_template or tokenizer.pad_token_id is None:
        raise ValueError("An instruction-tuned Gemma 4 tokenizer with a chat template/pad token is required")
    if "<turn|>" not in tokenizer.get_vocab():
        raise ValueError("The tokenizer must use the Gemma 4 <turn|> format")
    return tokenizer


def render_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def encode_example(tokenizer, messages, max_length):
    roles = [m.get("role") for m in messages]
    if roles not in (["system", "user", "assistant"], ["user", "assistant"]):
        raise ValueError("Expected an optional system message and one user/assistant pair")
    if any(not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages):
        raise ValueError("All messages must contain nonempty text")
    end_id = tokenizer.convert_tokens_to_ids("<turn|>")
    full = tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False,
        add_generation_prompt=False, enable_thinking=False,
    )
    # Select the final turn marker by ID, dropping only template suffix tokens.
    end = len(full) - 1 - full[::-1].index(end_id)
    full = full[:end + 1]
    prompt = tokenizer.encode(render_prompt(tokenizer, messages[:-1]), add_special_tokens=False)
    if full[:len(prompt)] != prompt or len(full) <= len(prompt) + 1:
        raise ValueError("Chat template prompt is not an exact prefix with a nonempty answer")
    views = []
    for message in messages[-2:]:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": message["content"]}],
            tokenize=True, return_dict=False,
            add_generation_prompt=False, enable_thinking=False,
        )
        end = len(ids) - 1 - ids[::-1].index(end_id)
        views.append(ids[:end + 1])
    if max(map(len, [full, *views])) > max_length:
        raise ValueError(f"Example exceeds max_length={max_length}; no partial-answer truncation")
    return {"input_ids": full, "labels": [-100]*len(prompt) + full[len(prompt):],
            "text_ids": views[0], "code_ids": views[1]}


def collate_examples(examples, pad_token_id):
    batch = {}
    for ids_key, mask_key in [("input_ids", "attention_mask"),
                              ("text_ids", "text_mask"), ("code_ids", "code_mask")]:
        width = max(len(e[ids_key]) for e in examples)
        batch[ids_key] = torch.tensor([
            e[ids_key] + [pad_token_id]*(width-len(e[ids_key])) for e in examples
        ], dtype=torch.long)
        batch[mask_key] = torch.tensor([
            [1]*len(e[ids_key]) + [0]*(width-len(e[ids_key])) for e in examples
        ], dtype=torch.long)
        if ids_key == "input_ids":
            batch["labels"] = torch.tensor([
                e["labels"] + [-100]*(width-len(e["labels"])) for e in examples
            ], dtype=torch.long)
    return batch


def load_text_decoder(model_name_or_path, *, revision=None, cache_dir=None, **kwargs):
    """Extract pretrained text weights, failing if any text parameters are missing."""
    revision = resolved_revision(model_name_or_path, revision)
    config = AutoConfig.from_pretrained(model_name_or_path, revision=revision, cache_dir=cache_dir)
    text_config = config.get_text_config()
    if text_config.model_type != "gemma4_text" or getattr(text_config, "enable_moe_block", False):
        raise ValueError("This runner supports dense Gemma 4 text decoders (E2B/E4B/31B), not MoE or Gemma4UV")
    # AutoModelForCausalLM also maps the outer Gemma4Config to the full multimodal
    # class in 5.17. Select the text class explicitly and preserve checkpoint keys.
    commit = getattr(config, "_commit_hash", None)
    model, loading = Gemma4ForCausalLM.from_pretrained(
        model_name_or_path, config=text_config, revision=commit or revision,
        cache_dir=cache_dir, key_mapping={r"^model\.language_model\.": "model."},
        output_loading_info=True, **kwargs,
    )
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise RuntimeError(f"Pretrained text weights were not fully restored: {loading}")
    model.config._commit_hash = commit
    return model


def load_base_model(model_name_or_path=DEFAULT_MODEL, *, revision=None,
                    cache_dir=None, quantization="none"):
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for pretrained training/inference; --prepare_only needs none")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configuration requires a GPU supporting BF16")
    kwargs = {}
    if quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization != "none":
        raise ValueError("quantization must be '4bit' or 'none'")
    model = load_text_decoder(
        model_name_or_path, revision=revision, cache_dir=cache_dir,
        dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa", **kwargs,
    )
    model.config.use_cache = False
    return model


def add_lora(model, rank=64, dropout=0.05, gradient_checkpointing=True):
    # PEFT's generic k-bit helper upcasts *all* frozen embeddings to FP32.
    # Gemma 4's large PLE table must stay BF16 on 12GB GPUs. LoRA freezes the
    # base here; non-reentrant checkpointing works without input-grad hooks.
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=rank*2,
        lora_dropout=dropout, target_modules=TARGET_MODULES, bias="none",
    ))
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("Expected only LoRA adapter parameters to be trainable")
    return model


def lm_loss(model, batch):
    return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                 labels=batch["labels"], use_cache=False).loss


def jepa_loss(model, batch):
    # PEFT injects LoRA in-place into this same decoder. Bypass only the LM head,
    # saving the two unnecessary [batch, sequence, vocabulary] logits tensors.
    decoder = model.get_base_model().model
    embeddings = []
    for ids_key, mask_key in [("text_ids", "text_mask"), ("code_ids", "code_mask")]:
        mask = batch[mask_key]
        hidden = decoder(input_ids=batch[ids_key], attention_mask=mask,
                         use_cache=False).last_hidden_state
        indices = mask.sum(-1) - 1
        embeddings.append(hidden[torch.arange(hidden.shape[0], device=hidden.device), indices].float())
    return 1 - F.cosine_similarity(embeddings[0], embeddings[1], dim=-1).mean()


def accumulation_weights(batches):
    """Match full-batch CE (per token) and JEPA (per example), including tails."""
    tokens = [int((batch["labels"][:, 1:] != -100).sum()) for batch in batches]
    examples = [batch["input_ids"].shape[0] for batch in batches]
    return [(n / sum(tokens), b / sum(examples)) for n, b in zip(tokens, examples)]


def prepare_dataset(path, tokenizer, max_length, max_samples=None, seed=42):
    records = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                records.append((line_number, json.loads(line)))
    random.Random(seed).shuffle(records)
    examples, skipped = [], 0
    for line_number, record in records:
        try:
            examples.append(encode_example(tokenizer, record["messages"], max_length))
        except ValueError as error:
            if "exceeds max_length" not in str(error):
                raise ValueError(f"{path}:{line_number}: {error}") from error
            skipped += 1
            continue
        if max_samples and len(examples) >= max_samples:
            break
    if not examples:
        raise ValueError("No usable examples; increase --max_length or check the dataset")
    return examples, {"total_records": len(records), "selected_examples": len(examples),
                      "skipped_overlength": skipped}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache_dir", default=".cache/huggingface")
    parser.add_argument("--train_file", default="datasets/synth_train.jsonl")
    parser.add_argument("--output_dir", default="outputs/gemma4-e2b-jepa")
    parser.add_argument("--quantization", choices=["4bit", "none"], default="none")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lbd", type=float, default=0.1)
    parser.add_argument("--regular", action="store_true", help="CE-only LoRA baseline (skips both JEPA forwards)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare_only", action="store_true", help="Validate tokenizer/dataset without downloading weights")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    for key in ("max_length", "batch_size", "gradient_accumulation_steps", "learning_rate",
                "num_epochs", "max_steps", "max_train_samples", "lora_rank"):
        value = getattr(args, key)
        if value is not None and value <= 0:
            parser.error(f"--{key} must be positive")
    if not math.isfinite(args.lbd) or args.lbd < 0 or not 0 <= args.lora_dropout < 1:
        parser.error("lbd must be finite/nonnegative and lora_dropout must be in [0, 1)")
    args.revision = resolved_revision(args.model_name_or_path, args.revision)
    set_seed(args.seed)
    output = Path(args.output_dir)
    if not args.prepare_only and output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}. Choose a new directory.")
    tokenizer = load_tokenizer(args.model_name_or_path, revision=args.revision, cache_dir=args.cache_dir)
    examples, dataset_stats = prepare_dataset(args.train_file, tokenizer, args.max_length,
                                              args.max_train_samples, args.seed)
    print(json.dumps({"dataset": dataset_stats, "sample": {
        "supervised": tokenizer.decode([v for v in examples[0]["labels"] if v != -100]),
        "text_view": tokenizer.decode(examples[0]["text_ids"]),
        "code_view": tokenizer.decode(examples[0]["code_ids"]),
    }}, ensure_ascii=True), flush=True)
    if args.prepare_only:
        return
    model = add_lora(load_base_model(args.model_name_or_path, revision=args.revision,
                                    cache_dir=args.cache_dir, quantization=args.quantization),
                     rank=args.lora_rank, dropout=args.lora_dropout)
    model.train()
    model.print_trainable_parameters()
    device = model.get_input_embeddings().weight.device
    output.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), method="LoRA" if args.regular else "LLM-JEPA",
                  predictors=0, pooling="end_of_turn", enable_thinking=False,
                  supervise_turn_end=True, dataset=dataset_stats,
                  dataset_sha256=hashlib.sha256(Path(args.train_file).read_bytes()).hexdigest(),
                  resolved_model_revision=getattr(model.config, "_commit_hash", None),
                  versions={name: importlib.metadata.version(name) for name in
                            ["torch", "transformers", "peft", "bitsandbytes", "accelerate"]})
    (output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed),
                        collate_fn=lambda rows: collate_examples(rows, tokenizer.pad_token_id))
    steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    total_steps = args.num_epochs * steps_per_epoch
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.03*total_steps), total_steps)
    optimizer.zero_grad(set_to_none=True)
    step = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(args.num_epochs):
            iterator = iter(loader)
            while group := list(islice(iterator, args.gradient_accumulation_steps)):
                sums = {"ce_loss": 0.0, "jepa_loss": 0.0}
                for batch, (ce_weight, jepa_weight) in zip(group, accumulation_weights(group)):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with autocast():
                        ce = lm_loss(model, batch)
                    if not torch.isfinite(ce):
                        raise RuntimeError("Non-finite language-model loss")
                    sums["ce_loss"] += ce.detach().item() * ce_weight
                    (ce * ce_weight).backward()
                    del ce
                    # Free CE logits/graph before the two independent JEPA views.
                    # Both backward passes precede the same optimizer step.
                    if not args.regular and args.lbd > 0:
                        with autocast():
                            distance = jepa_loss(model, batch)
                        if not torch.isfinite(distance):
                            raise RuntimeError("Non-finite JEPA loss")
                        sums["jepa_loss"] += distance.detach().item() * jepa_weight
                        (args.lbd * distance * jepa_weight).backward()
                        del distance
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                if step == 0 and grad_norm.item() == 0:
                    raise RuntimeError("No nonzero LoRA gradients reached the optimizer")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                metric = dict(sums, step=step, epoch=epoch+1, grad_norm=grad_norm.item(),
                              elapsed_seconds=round(time.perf_counter()-started, 3),
                              peak_allocated_gib=round(torch.cuda.max_memory_allocated()/2**30, 3))
                log.write(json.dumps(metric) + "\n")
                log.flush()
                print(json.dumps(metric), flush=True)
                if step >= total_steps:
                    break
            if step >= total_steps:
                break
    # Save an adapter, not a merged/dequantized full model.
    model.save_pretrained(output, save_embedding_layers=False)
    tokenizer.save_pretrained(output)
    (output / "completed.json").write_text(json.dumps({"optimizer_steps": step}), encoding="utf-8")
    print(f"Saved adapter and run metadata to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
