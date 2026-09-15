# Intervention JEPA Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development to execute
> the independent modules and perform a final integration review.

**Goal:** Eight selectable intervention learning recipes with independent rationale
JEPA, portable configuration, model/data extension boundaries, and a reliable handoff.

**Architecture:** Convert external records into a canonical Example. Keep prompt
formatting and model loading separate from the recipe loss and execution loop.
Preserve the existing Gemma4 loader's verified text checkpoint conversion.

**Tech Stack:** Python 3.12, PyTorch 2.11 CUDA 12.8, Transformers 5.17, PEFT 0.20,
bitsandbytes 0.50.2, JSON configuration and JSON Schema.

**Spec:** `docs/superpowers/specs/2026-09-15-intervention-experiments-design.md`

## Global constraints

- No rationale in direct JSON input; X includes policy in every independent view.
- Shared encoder, both JEPA gradients, k=0, no STP.
- CE accumulation weighted by supervised token counts; JEPA by example counts.
- Do not overwrite existing output directories or change original training scripts.
- Model revisions and exact data/config hashes must be saved with each run.
- Current checkout contains the prior authorized setup as uncommitted files.
  Preserve it on a feature branch; do not move or discard this state.

## Tasks

- [x] Config/data: tests for nested mapping, JSON validation, rationale requirements,
  strict config keys, path resolution and custom model revisions; implement APIs.
- [x] Models/formatting: tests for direct versus rationale versus sequential prompts,
  independent pooling, generic model decoder, exact checkpoint loading and gradients.
- [x] Objectives/training: test every recipe's branch use, masked CE, token/example
  accumulation equivalence, direct-path independence, adapter updates and save/reload.
- [x] Evaluation: test raw thought parsing, JSON Schema, structural equality,
  decision confusion counts, invalid output accounting and immutable base metadata.
- [x] Handoff: add example records/schema/configs, portable commands, extension map,
  implementation status, scientific limits, verified commands and remaining work.
- [x] Integration: run original + new tests, prepare all recipes, run tiny full cycles,
  bounded real E2B direct + rationale-JEPA smoke and code review; resolve findings.

## Execution ledger

- Config/data, model/formatting and evaluation assigned to disjoint-file subagents.
- Root owns objectives, training, configs/examples and integration/handoff.
- Ruling: JSON configs only; do not add a parallel YAML format or dynamic plugins.
- Ruling: Generic thinking formats are explicit extension points, not guessed delimiters.
- Config/data agent supplied fixtures; model/format agent drafted handoff docs;
  evaluation agent reviewed integration and fixed overflow/BOM parsing and local
  snapshot identity. Root integrated and finalized documentation.
- Final verification: 126 tests passed, no failures or skips; pip check passed.
  Eight recipes tokenized with the official tokenizer and completed tiny real-model
  train/save/reload/evaluate cycles. Real E2B rationale-JEPA ran for two steps.
- First E2B OFF evaluation produced fenced JSON (strict validity 0/4). Preserved
  that run. An explicit raw-JSON instruction and fresh two-step run yielded valid
  JSON 4/4, decision match 4/4, full structural match 2/4, no thought channels.
  These synthetic checks do not measure JEPA gains. H100 training remains unrun.
- Implementation was initially left uncommitted for review. The user subsequently
  authorized committing and pushing feat/intervention-jepa-experiments to their
  lhs-v/llm-jepa fork. Handoff now documents branch checkout and excluded artifacts.
