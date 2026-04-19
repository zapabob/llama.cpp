# Gemma 4 MTMD Host-Weight Fix

## Overview

Implemented a focused `tools/mtmd/clip.cpp` loader fix for Gemma 4 multimodal projectors so Gemma 4 vision and audio mmproj weights stay host-resident by default while CUDA remains available for compute scheduling.

## Background

- Target issue: `ggml-org/llama.cpp#21402`
- Failure mode before the fix:
  - Gemma 4 multimodal runs aborted during mmproj loading in `clip_model_loader::load_tensors`
  - the crash path was consistent with accelerator weight-buffer allocation followed by unsafe buffer-usage handling
- HF Gemma 4 architecture review used as grounding:
  - `transformers/src/transformers/models/gemma4/modular_gemma4.py`
  - `transformers/src/transformers/models/gemma4/configuration_gemma4.py`

## Decision

Keep Gemma 4 multimodal compute on the GPU scheduler path, but keep `PROJECTOR_TYPE_GEMMA4V` and `PROJECTOR_TYPE_GEMMA4A` mmproj weights on a host buffer by default.

This is intentionally a loader-placement fix, not a Gemma 4 graph rewrite and not a new CLI surface.

## Changed Files

- `F:\wt-llama-upstream-sync-2026-04-19\tools\mtmd\clip.cpp`

## Implementation Details

- Added `clip_proj_type_prefers_host_weights(projector_type)` for Gemma 4 projectors.
- Added `clip_select_weight_buffer_type(const clip_ctx &)` to choose CPU buffer placement for Gemma 4 mmproj weights when a GPU backend is active.
- Updated `clip_model_loader::load_tensors` to:
  - use the new buffer selection helper
  - log the host-weight path for Gemma 4
  - retry failed non-host buffer allocation on the CPU buffer once
  - throw a readable runtime error instead of reaching a null-buffer usage path

## Verification

### Focused rebuild and runtime proof

Because the upstream-sync worktree did not already have a complete CUDA build tree, verification was performed by mirroring the same `clip.cpp` patch into the clean runtime checkout at:

- `C:\Users\downl\Desktop\triality-platform-main-sync\repos\llama.cpp`

and rebuilding against the existing successful CUDA target tree:

- `F:\triality-targets\llama-gemma-mtmd`

Rebuild evidence:

- `C:\Users\downl\Desktop\triality-platform-main-sync\artifacts\cuda-smoke\20260419-075302-gemma-mtmd\2026-04-19_rebuild_gemma_mtmd_patch.log`

### Gemma 4 server result

Patched server startup command:

```bat
F:\triality-targets\llama-gemma-mtmd\bin\Release\llama-server.exe ^
  -m "C:\Users\downl\Desktop\SO8T\gguf_models\HauhauCS\Gemma-4-E4B-Uncensored-HauhauCS-Aggressive\Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf" ^
  --mmproj "C:\Users\downl\Desktop\SO8T\gguf_models\HauhauCS\Gemma-4-E4B-Uncensored-HauhauCS-Aggressive\mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf" ^
  --host 127.0.0.1 --port 8094 --ctx-size 2048 --no-warmup
```

Runtime evidence:

- `C:\Users\downl\Desktop\gemma-server-patched-no-warmup.log`

Key proof points from that log:

- `load_tensors: keeping gemma4v weights on host buffer while CUDA0 handles compute`
- `load_tensors: keeping gemma4a weights on host buffer while CUDA0 handles compute`
- `srv    load_model: loaded multimodal model, '...mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf'`
- `main: server is listening on http://127.0.0.1:8094`

This demonstrates that the old Gemma 4 multimodal load-time crash path is no longer blocking server startup.

### Additional observation

`llama-mtmd-cli` still crashes later in the text-model initialization / warmup path on this local Gemma artifact pair, even with `--no-warmup`. That appears separate from the original mmproj loader abort and should be treated as a residual follow-up issue rather than part of this loader fix.

Relevant logs:

- `C:\Users\downl\Desktop\triality-platform-main-sync\artifacts\cuda-smoke\20260419-075302-gemma-mtmd\2026-04-19_llama-mtmd-gemma-image-only-patched.log`
- `C:\Users\downl\Desktop\triality-platform-main-sync\artifacts\cuda-smoke\20260419-075302-gemma-mtmd\2026-04-19_llama-mtmd-gemma-image-only-patched-no-warmup.log`

## Residual Risks

- Full Gemma 4 multimodal request execution is not yet fully closed by this patch alone; only the mmproj load/startup crash was fixed and verified.
- `llama-mtmd-cli` still shows a later crash path that needs a separate follow-up.
- Verification used a mirror patch in a clean runtime checkout for rebuild speed; before opening the `zapabob/llama.cpp` PR, the same runtime smoke should be repeated from a build tree rooted directly on the upstream-sync branch.

## Recommended Next Actions

1. Build the upstream-sync branch itself in a clean dedicated CUDA build tree and rerun the same Gemma server startup proof.
2. Open the `zapabob/llama.cpp` PR with this loader fix and reference the server log evidence.
3. Triage the remaining `llama-mtmd-cli` post-load crash separately from issue `#21402`.
