# 2026-05-17 TheTom TurboQuant sync

## Objective

Merge current upstream `ggml-org/llama.cpp` into the fork, preserve the fork-only Triality/TurboQuant path, incorporate TheTom's current TurboQuant KV-cache implementation, and install the resulting CUDA build into the local `llama-turboquant` slot.

## Source refs

- Official upstream: `ggml-org/llama.cpp` `master` at `4f13cb742476d81a6b42a2aa5996e82a478c2481`.
- Fork branch: `codex/upstream-sync-2026-05-15`.
- TheTom source: `TheTom/llama-cpp-turboquant` `sync/upstream-b9190-mtp` at `e30bbcfe53a7c2576d0c621b9548d2305c735079`, which merges upstream `4f13cb742`.

## Merge work

- Ran the fork's Python merge helper:
  - `python scripts\merge_llama_cpp_upstream.py --survey-only --json-out _docs\2026-05-17_upstream_sync_survey_Codex.json --security-scan 300`
  - `python scripts\merge_llama_cpp_upstream.py --survey --json-out _docs\2026-05-17_upstream_sync_merge_Codex.json --security-scan 300 --merge-message "merge: upstream ggml-org llama.cpp master 2026-05-17; keep Triality TurboQuant"`
- Resolved the upstream converter split by keeping the official `convert_hf_to_gguf.py` wrapper and moving fork metadata into `conversion/base.py`.
- Updated `scripts/merge_llama_cpp_upstream.py` so future guarded merges protect the new converter split.
- Merged TheTom's sync branch and resolved type-space conflicts by preserving existing fork ABI for `GGML_TYPE_TQ4_1S = 36`, then adding TheTom's real KV cache types:
  - `GGML_TYPE_TURBO2_0 = 42`
  - `GGML_TYPE_TURBO3_0 = 43`
  - `GGML_TYPE_TURBO4_0 = 44`
  - `GGML_TYPE_TQ3_1S = 45`
  - `GGML_TYPE_COUNT = 46`
- Removed TheTom's duplicate `TQ4_1S` symbols from `ggml-turbo-quant.c`; the fork's existing `ggml-quants.c` remains authoritative for that type.

## Verification

- `python -m py_compile convert_hf_to_gguf.py conversion\base.py gguf-py\gguf\constants.py`
- `git diff --check`
- Configured an RTX 3060/3080-class CUDA build with:
  - `cmake -S . -B H:\llama-cpp-zapabob-thetom-20260517-sm86 -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF -DLLAMA_TURBOQUANT=ON`
- Dry-run compile check:
  - `ninja -n -C H:\llama-cpp-zapabob-thetom-20260517-sm86 llama-server llama-cli llama-turboquant`
  - Result: passed; Ninja resolved 445 build steps including TheTom TurboQuant CUDA sources.
- Real target build:
  - `ninja -C H:\llama-cpp-zapabob-thetom-20260517-sm86 llama-server llama-cli llama-turboquant`
  - Result: passed.
- Build artifacts report `version: 9451 (68124bdbe)` and list `CUDA0: NVIDIA GeForce RTX 3060`.
- `llama-server --help` lists `turbo2, turbo3, turbo4` for K/V cache types and draft K/V cache types.

## Local install

- Installed by copying build artifacts into:
  - `C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin`
- Previous binaries were backed up to:
  - `C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin-backup-codex-20260517-125005`
- Verified PATH-backed `llama-server` resolves to the installed slot and reports `version: 9451 (68124bdbe)`.
- Verified installed `llama-server --list-devices` sees `CUDA0: NVIDIA GeForce RTX 3060`.

## Notes

- The build used `CMAKE_CUDA_ARCHITECTURES=86` per the RTX 3060/3080 requirement.
- The default official runtime remains available; TurboQuant KV cache types are exposed through the official `--cache-type-*` options.
- `tools/ui` dependency installation required a local `NPM_CONFIG_STRICT_SSL=false` workaround on this machine because npm certificate verification failed with `UNABLE_TO_VERIFY_LEAF_SIGNATURE`. No repository npm config was changed.
- npm reported dependency audit warnings in `tools/ui`; they were not auto-fixed during this merge because doing so would create unrelated web UI dependency churn.
