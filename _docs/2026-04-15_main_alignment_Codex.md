# Overview

Aligned local `main` handling with the current fork baseline without importing the stale `import/codex-snapshot` mtmd rollback.

# Background / requirements

- User requested that Gemma4 handling be aligned with the official repository behavior.
- User also requested that fork-specific Triality / TurboQuant / Hypura context be preserved.
- `import/codex-snapshot` contains an older dirty snapshot that removes `gemma4a` support from `tools/mtmd`.

# Assumptions / decisions

- `main` at `df3913e92` is the authoritative runtime baseline for this fork.
- `main` already contains the full `gemma4a` path, matching the `hypura` vendored `llama.cpp` and upstream-aligned behavior.
- Because of that, no `mtmd` code from `import/codex-snapshot` was merged into `main`.
- The only safe main-branch integration needed in this pass was README documentation for the fork-specific ecosystem and measured local setup.

# Changed files

- `README.md`
- `_docs/2026-04-15_main_alignment_Codex.md`

# Implementation details

- Confirmed in `main` that `tools/mtmd` already includes:
  - `models/gemma4a.cpp`
  - `PROJECTOR_TYPE_GEMMA4A`
  - Gemma4 audio tensor-name macros (`TN_A_*`)
  - Gemma4 audio tensor members in `clip-model.h`
  - Gemma4 audio builder / tensor loading / embedding-dimension handling in `clip.cpp`
- Added fork-specific README sections to preserve local documentation context:
  - related repositories
  - TurboQuant support status
  - measured local environment and runtime settings
- Ensured the inserted `bash` fence is properly closed before `## Dependencies`.

# Commands run

```powershell
git -C C:\Users\downl\Desktop\llama.cpp-zapabob worktree add C:\Users\downl\Desktop\llama.cpp-main-sync main
Select-String -Path C:\Users\downl\Desktop\llama.cpp-main-sync\tools\mtmd\* -Pattern "GEMMA4A|TN_A_|gemma4a"
git -C C:\Users\downl\Desktop\llama.cpp-zapabob diff --stat main -- README.md tools/mtmd/CMakeLists.txt tools/mtmd/clip-impl.h tools/mtmd/clip-model.h tools/mtmd/clip.cpp tools/mtmd/models/models.h tools/mtmd/models/gemma4a.cpp
git -C C:\Users\downl\Desktop\llama.cpp-main-sync diff --check
git -C C:\Users\downl\Desktop\llama.cpp-main-sync status --short --branch
```

# Test / verification results

- Verified that local `main` already contains the full `gemma4a` implementation path.
- Verified that `README.md` now contains a closed fenced block before `## Dependencies`.
- `git diff --check` is expected to be clean for the README/docs-only change in this worktree.
- No C++ build or runtime tests were run in this pass because no executable code changed on `main`.

# Residual risks

- The large dirty `import/codex-snapshot` branch remains unresolved and should not be merged blindly into `main`.
- The added README benchmark values are fork-local documentation and may become stale over time.

# Recommended next actions

- If you want to salvage specific code from `import/codex-snapshot`, port it to `main` as small reviewed diffs instead of merging the snapshot branch wholesale.
- If desired, cut a local commit on `main` for the README/docs integration only.
