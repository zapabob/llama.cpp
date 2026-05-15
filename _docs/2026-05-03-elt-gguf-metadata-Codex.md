# 2026-05-03 ELT GGUF metadata

## Goal

Add GGUF metadata emission for ELT looped Qwen3.5 HF exports that store an
`elt_config` object in `config.json`, while keeping the converted tensor
architecture as the existing `qwen35` path.

## Files touched

- `convert_hf_to_gguf.py`
- `tests/test_elt_metadata.py`

## Key decisions

- Emit ELT metadata only when `hparams["elt_config"]` is a dictionary.
- Use stable `elt.*` keys:
  - `elt.loop.enabled`
  - `elt.loop.required`
  - `elt.loop.L_min`
  - `elt.loop.L_max`
  - `elt.loop.L_default`
- `elt.loop_unit`
- `elt.backbone_kind`
- `elt.source_model_id`
- `elt.gguf.runtime_status`
- `elt.model_family`
- `elt.loop.model_family`
- Do not add a new runtime graph or a new architecture name; Qwen3.5 tensors
  continue through the existing `qwen35` architecture.
- Use `ELT/Qwen3.5-looped` as the TurboQuant `model_family` when loop runtime
  metadata declares recurrence, and add an `elt` sub-object to
  `hypura.turboquant.weight.payload_json`; plain Qwen payloads are unchanged.

## Tests

- Added focused pytest coverage for ELT metadata emission, plain Qwen no-op
  behavior, and TurboQuant payload enrichment without changing `model_family`.

## Next session notes

- This is metadata-only plumbing. Runtime loop execution still needs a separate
  graph/runtime implementation if llama.cpp is later expected to execute ELT
  recurrence natively.
