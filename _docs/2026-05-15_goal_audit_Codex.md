# 2026-05-15 Goal Audit

## Objective

Bring the latest official `ggml-org/llama.cpp` functionality, vulnerability-related updates, and bug fixes into the fork with the Python merge script; preserve the fork-specific Triality/TurboQuant functionality while following the official API; then make it usable on this PC by overwrite-installing the runnable build.

## Prompt-to-Artifact Checklist

| Requirement | Evidence | Status |
| --- | --- | --- |
| Use a Python script for the upstream merge | `python scripts\merge_llama_cpp_upstream.py ...` produced `_docs/2026-05-15_upstream_sync_survey_Codex.json`, `_docs/2026-05-15_upstream_sync_merge_Codex.json`, and `_docs/2026-05-15_upstream_sync_postmerge_Codex.json`. | Done |
| Incorporate latest official upstream | `git fetch upstream master` on 2026-05-15 kept `upstream/master` at `5c0e94683`; `git rev-list --left-right --count HEAD...upstream/master` reports `44 0`. Post-merge survey reports `head_contains_upstream=True`, `head_short=68943e967`, `upstream_short=5c0e94683`, and empty `incoming_upstream_commits_preview` / `incoming_upstream_files`. | Done |
| Include vulnerability/security-relevant upstream fixes | `_docs/2026-05-15_upstream_sync_survey_Codex.json` flagged incoming commit `527045bfb flush the gpu profile timestamp before the queryset is overflowed (#22995)`. `_docs/2026-05-15_upstream_sync_postmerge_Codex.json` shows `security_flagged_incoming_commits: []`, meaning the flagged incoming security-relevant commit is no longer pending after merge. | Done |
| Preserve independent Triality/TurboQuant features | Post-merge survey reports `protected_paths_ok=True` and `default_alignment_ok=True`; protected paths include `convert_hf_to_gguf.py`, `src/llama-turboquant.*`, and KV-cache TurboQuant wiring. | Done |
| Follow current official API after merge | Initial survey identified API-touching incoming paths (`include/llama.h`, `ggml/include/ggml*.h`, `src/llama.cpp`, `src/llama-graph.cpp`, `src/llama-kv-cache.cpp`); post-merge survey reports empty `incoming_api_touch_paths`. Limited compile found stale custom CMake references to the old `common` target; commit `a653752e3` updated them to official `llama-common` in `tools/turboquant/CMakeLists.txt` and `tests/CMakeLists.txt`. `git diff --check HEAD` passes. | Done |
| RTX 3060 / RTX 3080 scoped CUDA configuration | `_docs/2026-05-15_sm86_current_head_actual_build_Codex.txt` records actual build target scope `llama-server llama-cli llama-turboquant` with `cuda_architectures=86`. The build log records `Using CMAKE_CUDA_ARCHITECTURES=86 CMAKE_CUDA_ARCHITECTURES_NATIVE=86-real`. | Done |
| Dry-run compile plan for requested runnable targets | `_docs/2026-05-15_sm86_current_head_dryrun_targets_Codex.txt` shows current HEAD `7f2e34fcf` running `ninja -n llama-server llama-cli llama-turboquant` and reaching `[412/412] Linking CXX executable bin\llama-server.exe` with `DRY_RUN_EXITCODE=0`. | Done |
| Actual compile/link | `_docs/2026-05-15_sm86_current_head_actual_build_Codex.txt` records current HEAD `0f7cd1406` building `llama-server`, `llama-cli`, and `llama-turboquant` through `[412/412] Linking CXX executable bin\llama-server.exe`; a follow-up no-op `ninja -C H:\llama-cpp-zapabob-upstream-20260515-install-sccache llama-server llama-cli llama-turboquant` returned `EXITCODE=0`. | Done |
| Overwrite install on this PC | `_docs/2026-05-15_overwrite_install_Codex.txt` records old installed files from March/April 2026, backup to `C:\Users\downl\AppData\Local\Programs\llama-turboquant\backup-codex-20260515-1425`, and copied current build artifacts into `C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin`. | Done |
| Verify installed runtime is the one this PC uses | `_docs/2026-05-15_installed_runtime_verify_Codex.txt` records `where.exe llama-server` resolving the install slot first, `LLAMA_CPP_SERVER_EXE` pointing to the install slot, installed `llama-server.exe --version` and `llama-cli.exe --version` reporting `version: 9203 (0f7cd1406)`, `llama-turboquant.exe` printing usage, `nvidia-smi` showing `NVIDIA GeForce RTX 3060, 8.6`, and installed `llama-server.exe --list-devices` reporting `CUDA0: NVIDIA GeForce RTX 3060`. | Done |

## Current State Snapshot

- Repository branch: `codex/upstream-sync-2026-05-15`.
- Repository head at completion verification: `0f7cd1406 docs: record current sm86 dry-run verification`.
- Merge commit: `68943e967 merge: upstream ggml-org llama.cpp master 2026-05-15; keep Triality TurboQuant`.
- Upstream master included: `5c0e94683`.
- Fresh upstream check: `git fetch upstream master` on 2026-05-15 left `upstream/master=5c0e94683`; `HEAD...upstream/master` is `44 ahead, 0 behind`.
- Build cache used for installation: `H:\llama-cpp-zapabob-upstream-20260515-install-sccache`.
- CUDA scope: `CMAKE_CUDA_ARCHITECTURES=86`, suitable for RTX 3060 / RTX 3080 sm_86.
- Installed runtime slot: `C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin`.
- Installed versions: `llama-server.exe` and `llama-cli.exe` both report `version: 9203 (0f7cd1406)`.
- CUDA installed-runtime proof: `llama-server.exe --list-devices` reports `CUDA0: NVIDIA GeForce RTX 3060`.

## Completion Decision

The upstream merge, security-relevant upstream delta, custom Triality/TurboQuant preservation, official API follow-up, RTX 3060 / RTX 3080 scoped dry-run, actual sm_86 build, overwrite install, and installed CUDA runtime verification are complete. No required objective item remains open.
