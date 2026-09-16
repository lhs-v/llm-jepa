"""Pinned pretrained causal models and LoRA without upcasting frozen embeddings."""
from pathlib import Path
import hashlib
import inspect
import os
import re

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from finetune_gemma4 import TARGET_MODULES, load_text_decoder


def local_source_fingerprints(path):
    """Hash local snapshot artifacts in bounded memory; Hub IDs have no local manifest.

    Relative POSIX paths make the inventory portable. Dot directories such as
    .git and .cache contain mutable bookkeeping rather than snapshot artifacts.
    """
    root = Path(path).expanduser()
    if not root.is_dir():
        return None
    root = root.resolve()
    fingerprints = {}

    def raise_scan_error(error):
        raise error

    for directory, directories, filenames in os.walk(root, onerror=raise_scan_error):
        directories[:] = sorted(name for name in directories if not name.startswith("."))
        for name in sorted(filenames):
            artifact = Path(directory) / name
            if not artifact.is_file():
                continue
            digest = hashlib.sha256()
            with artifact.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            fingerprints[artifact.relative_to(root).as_posix()] = digest.hexdigest()
    return dict(sorted(fingerprints.items()))


def _source(model_config):
    source = model_config["model_name_or_path"]
    kwargs = {"revision": model_config.get("revision"), "cache_dir": model_config.get("cache_dir")}
    config = AutoConfig.from_pretrained(source, **kwargs)
    commit = getattr(config, "_commit_hash", None)
    # A local directory has no Hub commit; do not mislabel a branch as immutable.
    revision = commit or kwargs["revision"]
    resolved = None if Path(source).is_dir() else (commit or (revision if revision and re.fullmatch(r"[0-9a-f]{40}", revision) else None))
    return config, revision, resolved


def _tokenizer(model_config, revision):
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_name_or_path"], revision=revision, cache_dir=model_config.get("cache_dir"),
    )
    if not tokenizer.chat_template:
        raise ValueError("An instruction tokenizer with its native chat template is required")
    tokenizer.padding_side = "right"
    fallback = None
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs a PAD token or an EOS token usable for right padding")
        tokenizer.pad_token = tokenizer.eos_token
        fallback = "eos"
    if model_config.get("chat_format", "gemma4") == "gemma4" and "<turn|>" not in tokenizer.get_vocab():
        raise ValueError("Gemma 4 formatting requires the <turn|> tokenizer")
    return tokenizer, fallback


def load_tokenizer(model_config):
    """Resolve config/tokenizer only; missing PAD falls back to EOS with right masks."""
    _, revision, _ = _source(model_config)
    return _tokenizer(model_config, revision)[0]


def get_decoder(model, decoder_path=None):
    """Return the same adapter-injected decoder, bypassing only its vocabulary head."""
    base = model.get_base_model() if isinstance(model, PeftModel) else model
    if decoder_path:
        try:
            decoder = base.get_submodule(decoder_path)
        except (AttributeError, KeyError) as error:
            raise ValueError(f"decoder_path={decoder_path!r} does not identify a module") from error
    elif callable(getattr(base, "get_decoder", None)):
        decoder = base.get_decoder()
    else:
        decoder = getattr(base, "model", None)
    if not isinstance(decoder, torch.nn.Module) or decoder is base:
        raise ValueError("Cannot identify a causal decoder; configure model.decoder_path explicitly")
    if "input_ids" not in inspect.signature(decoder.forward).parameters:
        raise ValueError("decoder_path must select a text decoder accepting input_ids")
    if callable(getattr(decoder, "get_output_embeddings", None)) and decoder.get_output_embeddings() is not None:
        raise ValueError("decoder_path must select the hidden-state decoder, excluding the LM head")
    return decoder


def attach_lora(model, model_config):
    if isinstance(model, PeftModel):
        raise ValueError("Model already has an adapter; do not attach another LoRA")
    rank = model_config.get("lora_rank", 64)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=rank,
        lora_alpha=model_config.get("lora_alpha", 128),
        lora_dropout=model_config.get("lora_dropout", 0.05),
        target_modules=model_config.get("target_modules", TARGET_MODULES), bias="none",
    ))
    model.config.use_cache = False
    # Non-reentrant checkpointing propagates adapter gradients with frozen inputs.
    # prepare_model_for_kbit_training would upcast Gemma's large PLE embeddings.
    if model_config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("Only LoRA adapter parameters may be trainable")
    return model


def load_model(model_config, training=True, adapter_dir=None):
    """Load one pinned base/tokenizer pair on one device, optionally restore an adapter."""
    quantization = model_config.get("quantization", "none")
    device = torch.device(model_config.get("device", "cuda:0"))
    dtype_name = model_config.get("dtype", "bfloat16")
    if dtype_name not in {"bfloat16", "float32"}:
        raise ValueError("dtype must be bfloat16 or float32")
    dtype = getattr(torch, dtype_name)
    if quantization not in {"none", "4bit"}:
        raise ValueError("quantization must be none or 4bit")
    if quantization == "4bit" and device.type != "cuda":
        raise ValueError("4bit quantization requires a CUDA device")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device is unavailable; preparation needs no GPU")
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Requested bfloat16 requires a BF16-capable CUDA GPU")
    backend = model_config.get("backend", "gemma4")
    if backend not in {"gemma4", "auto_causal_lm"}:
        raise ValueError("backend must be gemma4 or auto_causal_lm")
    source = model_config["model_name_or_path"]
    source_fingerprints = local_source_fingerprints(source)
    config, revision, resolved = _source(model_config)
    # In Transformers 5, dual encoder/decoder families (BERT, RoBERTa) declare
    # is_decoder; native causal families such as Llama/GPT-2 do not declare it.
    if getattr(config, "is_encoder_decoder", False) or getattr(config, "is_decoder", True) is False:
        raise ValueError("This runner requires a decoder-only causal language model")
    tokenizer, fallback = _tokenizer(model_config, revision)
    kwargs = {"revision": revision, "cache_dir": model_config.get("cache_dir"),
              "dtype": dtype, "device_map": {"": str(device)},
              "attn_implementation": model_config.get("attn_implementation", "sdpa")}
    if quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    if backend == "gemma4":
        model = load_text_decoder(source, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(source, config=config, **kwargs)
    model.config._commit_hash = resolved
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=training)
        if training:
            model.config.use_cache = False
            if model_config.get("gradient_checkpointing", True):
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    elif training:
        model = attach_lora(model, model_config)
    if training:
        if any(parameter.requires_grad and "lora_" not in name for name, parameter in model.named_parameters()):
            raise RuntimeError("Only LoRA adapter parameters may be trainable")
        model.train()
    else:
        model.requires_grad_(False)
        model.eval()
    decoder = get_decoder(model, model_config.get("decoder_path"))
    tokenizer_commit = tokenizer.init_kwargs.get("_commit_hash") or resolved
    metadata = {
        "model_name_or_path": source, "original_model_name_or_path": source,
        "requested_revision": model_config.get("revision"), "resolved_revision": resolved,
        "local_source_sha256": source_fingerprints,
        "base_commit": resolved, "tokenizer_commit": tokenizer_commit,
        "tokenizer_revision": tokenizer_commit, "backend": backend,
        "dtype": dtype_name, "device": str(device), "quantization": quantization,
        "decoder_class": type(decoder).__name__, "pad_token_fallback": fallback,
        "padding_side": tokenizer.padding_side,
        "adapter_dir": str(adapter_dir) if adapter_dir is not None else None,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "frozen_embedding_dtypes": sorted({str(p.dtype) for name, p in model.named_parameters()
                                            if "embed" in name and not p.requires_grad}),
    }
    return model, tokenizer, metadata
