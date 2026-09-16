# Intervention JEPA experiment design

## Authorized objective

Implement the eight training variants discussed with the user. The production
task is situation + policy -> intervention JSON. Rationale data and model layout
may change; preserve explicit data and model boundaries and document handoff.
The user authorized implementation after the experimental design discussion.

## Scientific contract

- Direct JSON prediction never receives a rationale prefix.
- Independent JEPA views encode X=(situation, policy) and either JSON or rationale.
- Shared LoRA parameters, bidirectional gradients, cosine loss, identity predictor k=0.
- Separate rationale supervision uses its own request, not a prefix of direct JSON.
- Sequential and mixed modes use explicit thinking-on/off prompts.
- No STP, teacher EMA, stop-gradient, new prediction tokens, or performance claims.
- Eight recipes: sft, json_jepa, rationale_jepa, multitask, multitask_jepa,
  cot, cot_jepa, mixed_jepa.

## Module boundaries

- `jepa_experiments/config.py`: strict versioned JSON configuration and path resolution.
- `data.py`: external fields/messages -> canonical Example(id,situation,policy,rationale,target).
- `models.py`: model/tokenizer loading, LoRA, decoder access, revision metadata.
- `formatting.py`: model-specific chat prompts, loss masks, pooling and output parsing.
- `objectives.py`: recipe selection, collation, properly normalized CE and JEPA gradients.
- `training.py`: single-device loop, snapshots, adapter saves, reproducible run metadata.
- `evaluation.py`: actual JSON/decision/schema metrics and explicit thought/token accounting.
- Root entrypoints `train_experiment.py` and `evaluate_experiment.py`.
- Configs, illustrative data, DATA_FORMAT.md and HANDOFF.md are portable handoff artifacts.

## Configuration and scope

H100 single-GPU BF16 LoRA rank 64 is the primary experiment configuration. Local
4-bit smoke tests establish code compatibility only. Support explicit Gemma4 text
loading and a generic causal-LM adapter for direct-output recipes; generic thinking
formats require a formatter extension and must fail clearly until implemented.
Use nested dot-path data field mappings and optional JSON Schema for varying targets.
Relative filesystem paths resolve against the configuration file directory.
Preserve the original scripts and their prior tests.

## Verification

Test all recipe routing and independent views using the cached official tokenizer.
Use tiny real Gemma4 and generic causal decoders to test gradients, checkpointing,
accumulation, frozen weights, save/reload, and a complete train/eval cycle.
Run a bounded real E2B smoke test of direct JSON plus rationale JEPA. No full H100
run or quality benchmark is implied. Record exact commands, results, limitations,
and integration entry points in HANDOFF.md.
