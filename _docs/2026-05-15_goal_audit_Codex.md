# 2026-05-15 Goal Audit

## Objective

Bring the latest official `ggml-org/llama.cpp` functionality, vulnerability-related updates, and bug fixes into the fork with the Python merge script; preserve the fork-specific Triality/TurboQuant functionality while following the official API; then make it usable on this PC by overwrite-installing the runnable build.

## Success Criteria and Evidence

| Requirement | Evidence | Status |
| --- | --- | --- |
| Use a Python script for the upstream merge | `python scripts\merge_llama_cpp_upstream.py ...` produced `_docs/2026-05-15_upstream_sync_survey_Codex.json`, `_docs/2026-05-15_upstream_sync_merge_Codex.json`, and `_docs/2026-05-15_upstream_sync_postmerge_Codex.json`. | Done |
| Incorporate latest official upstream | Branch `codex/upstream-sync-2026-05-15` is at merge commit `68943e967`; post-merge survey reports `head_contains_upstream=True`, `head_short=68943e967`, `upstream_short=5c0e94683`, and `incoming_commits=0`. | Done |
| Preserve independent Triality/TurboQuant features | Post-merge survey reports `protected_paths_ok=True` and `default_alignment_ok=True`; protected paths include `convert_hf_to_gguf.py`, `src/llama-turboquant.*`, and KV-cache TurboQuant wiring. | Done |
| Follow current official API after merge | Initial survey identified API-touching incoming paths (`include/llama.h`, `ggml/include/ggml*.h`, `src/llama.cpp`, `src/llama-graph.cpp`, `src/llama-kv-cache.cpp`); post-merge survey reports no remaining incoming API paths. Limited compile found stale custom CMake references to the old `common` target; those were updated to official `llama-common` in `tools/turboquant/CMakeLists.txt` and `tests/CMakeLists.txt`. `git diff --check HEAD` passes. | Done for covered paths |
| RTX 3060 / RTX 3080 scoped CUDA configuration | `H:\llama-cpp-zapabob-upstream-20260515-build\CMakeCache.txt` has `CMAKE_CUDA_ARCHITECTURES=86`, `GGML_CUDA=ON`, `LLAMA_TURBOQUANT=ON`, `LLAMA_BUILD_SERVER=ON`, and `LLAMA_BUILD_TOOLS=ON`. | Done |
| Dry-run compile plan for requested runnable targets | `_docs/2026-05-15_dry_run_sm86_ninja_raw_Codex.txt` shows `ninja -n llama-server llama-cli llama-turboquant` reaching `[51/51] Linking CXX executable bin\llama-server.exe`. | Done |
| Actual compile/link | `_docs/2026-05-15_sm86_limited_compile_fixed_Codex.txt` records `EXITCODE=0` and confirms `llama-server.exe=True`, `llama-cli.exe=True`, and `llama-turboquant.exe=True` for the limited sm_86 targets. | Done for requested limited targets |
| Overwrite install on this PC | Not run after the user narrowed the request to dry-run only. Existing install slot `C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin` was not modified. | Missing |

## Current State Snapshot

- Repository branch: `codex/upstream-sync-2026-05-15`.
- Repository head at limited compile verification: `b6184ab44 docs: clarify dry-run install gap`.
- Merge commit: `68943e967 merge: upstream ggml-org llama.cpp master 2026-05-15; keep Triality TurboQuant`.
- Upstream master included: `5c0e94683`.
- Git worktree status: clean.
- Build cache: `H:\llama-cpp-zapabob-upstream-20260515-build\CMakeCache.txt`.
- CUDA scope: `CMAKE_CUDA_ARCHITECTURES=86`, suitable for RTX 3060 / RTX 3080 sm_86.
- Built outputs in `H:\llama-cpp-zapabob-upstream-20260515-build\bin`: `ggml-base.dll`, `ggml-cpu.dll`, `ggml-cuda.dll`, `ggml.dll`, `llama-common.dll`, `llama.dll`, `llama-server.exe`, `llama-cli.exe`, and `llama-turboquant.exe`.
- Runtime smoke evidence in `_docs/2026-05-15_sm86_runtime_smoke_Codex.txt`:
  - `llama-server.exe --version`: `version: 9200 (b6184ab44)`.
  - `llama-cli.exe --version`: `version: 9200 (b6184ab44)`.
  - `llama-turboquant.exe`: usage output is printed; no model/artifact run was performed.
  - `nvidia-smi`: `NVIDIA GeForce RTX 3060, 8.6, 596.49`.
- Existing installed runtime slot still contains older executable timestamps:
  - `C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin\llama-server.exe` from 2026-04-22.
  - `C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin\llama-cli.exe` from 2026-04-22.
  - `C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin\llama-turboquant.exe` from 2026-03-30.

## Completion Decision

The upstream merge, dry-run compile-plan check, official target-name follow-up, and requested limited sm_86 compile are complete. The full active goal is still not complete because overwrite install into `C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin` has not been performed after the latest dry-run/compile-check-only instruction.
