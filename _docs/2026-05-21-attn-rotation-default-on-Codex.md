# 2026-05-21 attention rotation default on

## Overview

Changed TurboQuant attention rotation from default-off to default-on for supported K/V cache sides while preserving explicit opt-out controls and Triality/TheTom compatibility.

## Background / requirements

- User request: "default on".
- Prior state: `llama_kv_cache` initialized `attn_rot_k` and `attn_rot_v` to `false`, then allowed `LLAMA_ATTN_ROT_K_OVERRIDE=1` and `LLAMA_ATTN_ROT_V_OVERRIDE=1` to enable supported sides.
- Required behavior: default to enabled when the selected K/V cache type is quantized and the head dimension satisfies the existing rotation guard.

## Decisions

- Keep `LLAMA_ATTN_ROT_DISABLE=1` as a hard global lock-out.
- Make K/V rotation default on only when the existing support checks pass:
  - nonzero effective head dimension
  - quantized K/V cache type
  - `head_dim % 64 == 0`
- Reuse existing `LLAMA_ATTN_ROT_K_OVERRIDE` and `LLAMA_ATTN_ROT_V_OVERRIDE` as per-side overrides:
  - `0` disables that side
  - `1` enables that side if support checks pass

## Changed files

- `src/llama-kv-cache.cpp`
- `_docs/2026-05-21-attn-rotation-default-on-Codex.md`

## Verification results

- `git diff --check`: passed.
- `python -m py_compile convert_hf_to_gguf.py conversion\base.py tests\test_elt_metadata.py`: passed.
- `python -m pytest tests\test_elt_metadata.py -q`: `3 passed`.
- Static C++ Triality/TurboQuant tests in `H:\llama-cpp-main-unify-triality-static-verify-20260521`:
  - `test-turboquant-runtime-reference.exe`: 9 tests, 30 assertions, 0 failures.
  - `test-turboquant-gguf-metadata.exe`: 7 tests, 32 assertions, 0 failures.
  - `test-turboquant-artifact.exe`: 4 tests, 15 assertions, 0 failures.
- CUDA sm86 configure/build in `H:\llama-cpp-zapabob-thetom-20260517-sm86`:
  - `cmake -S . -B H:\llama-cpp-zapabob-thetom-20260517-sm86 -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF -DLLAMA_TURBOQUANT=ON`
  - `ninja -C H:\llama-cpp-zapabob-thetom-20260517-sm86 llama-server llama-cli llama-turboquant`
  - Result: passed with existing CUDA template unused-variable warnings only.
- Local overwrite install:
  - Source: `H:\llama-cpp-zapabob-thetom-20260517-sm86\bin`
  - Destination: `C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin`
  - Backup: `C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin-backup-codex-20260521-032714`
  - Updated files: `ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll`, `ggml-cuda.dll`, `llama.dll`, `llama-common.dll`, `mtmd.dll`, `llama-server.exe`, `llama-cli.exe`, `llama-turboquant.exe`
- Installed runtime checks:
  - `C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin\llama-server.exe --version`: `version: 9462 (5a824f830)`.
  - `llama-server --version`: `version: 9462 (5a824f830)`.
  - `llama-server --list-devices`: `CUDA0: NVIDIA GeForce RTX 3060`.
  - `llama-server --help`: cache types include `turbo2`, `turbo3`, and `turbo4` for K/V and draft K/V.

## Residual risks

- The default is now intentionally more aggressive. `LLAMA_ATTN_ROT_DISABLE=1` remains available for full opt-out, and `LLAMA_ATTN_ROT_K_OVERRIDE=0` / `LLAMA_ATTN_ROT_V_OVERRIDE=0` can disable one cache side.
- No model-quality benchmark was rerun in this change; verification covered compilation, metadata compatibility, and Triality/TurboQuant reference behavior.
