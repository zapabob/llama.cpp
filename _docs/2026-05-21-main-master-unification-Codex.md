# 2026-05-21 main/master unification

## Overview

Unified the remote `master` branch into `main` for `zapabob/llama.cpp`, preserving the `main`-only ELT GGUF metadata work while making `main` contain the current `master` state.

## Background / requirements

- User request: merge `master` and `main`, then make `main` the primary branch.
- Repository: `C:\Users\downl\Desktop\llama.cpp-zapabob`.
- GitHub repository: `zapabob/llama.cpp`.
- GitHub default branch was already `main` before the merge.
- `origin/main` had one unique merge commit: PR #17, `codex/elt-looped-gguf-metadata`.
- `origin/master` was ahead by the upstream-sync and TheTom TurboQuant work through PR #19.

## Decisions

- Kept `main` as the canonical/default branch.
- Merged `origin/master` into `main` with a normal merge commit instead of rebasing or force-pushing.
- Preserved the `main`-only ELT metadata tests and documentation.
- Left `master` intact for compatibility; `main` now contains its content.

## Changed files

- Merge commit: `merge: unify master into main`.
- This log: `_docs/2026-05-21-main-master-unification-Codex.md`.

## Commands run

- `git fetch origin --prune`
- `gh repo view zapabob/llama.cpp --json defaultBranchRef,nameWithOwner,url`
- `git switch main`
- `git merge --ff-only origin/main`
- `git merge --no-ff origin/master -m "merge: unify master into main"`
- `python -m py_compile convert_hf_to_gguf.py conversion\base.py tests\test_elt_metadata.py`
- `git diff --check`
- `python -m pytest tests\test_elt_metadata.py -q`

## Verification results

- `origin/main...origin/master` before merge: `1 580`.
- Merge completed with no conflicts.
- `python -m py_compile ...`: passed.
- `git diff --check`: passed.
- `python -m pytest tests\test_elt_metadata.py -q`: `3 passed`.
- After merge, local `main` contains `origin/master`; `main...origin/master` reports `2 0` before this log commit.

## Residual risks

- Full CUDA build was not rerun for this branch unification because no source conflict was resolved manually; this was a branch-topology merge plus targeted ELT regression verification.
- `master` remains on GitHub unless separately deleted or protected/retargeted.
