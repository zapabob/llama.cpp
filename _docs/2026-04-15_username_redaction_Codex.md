# Overview

Redacted user-identifying display text and local absolute paths from the fork-specific `main` documentation updates.

# Background / requirements

- User requested that usernames be hidden while continuing the main-branch integration work.
- The previous README and implementation log additions included:
  - visible fork-owner names in section labels and link text
  - local Windows absolute paths in examples / command history

# Assumptions / decisions

- Keep repository links intact so downstream references to TurboQuant-CUDA and Hypura still resolve.
- Hide usernames in rendered documentation where practical:
  - genericize section titles and link labels
  - replace local absolute paths with placeholders or environment-variable based examples

# Changed files

- `README.md`
- `_docs/2026-04-15_main_alignment_Codex.md`
- `_docs/2026-04-15_username_redaction_Codex.md`

# Implementation details

- Renamed the README section header from a user-specific fork label to a generic fork-related label.
- Changed rendered repository labels to neutral names while keeping the underlying GitHub URLs.
- Replaced the Windows user-specific model path example with a `%USERPROFILE%`-based path.
- Replaced local absolute paths in the implementation log command block with `<FORK_REPO_ROOT>` and `<MAIN_WORKTREE>` placeholders.

# Commands run

```powershell
Select-String -Path README.md,_docs\2026-04-15_main_alignment_Codex.md -Pattern "C:\\Users\\|USERPROFILE"
git diff --check
git status --short --branch
```

# Test / verification results

- Verified the README still contains a properly closed fenced `bash` block before `## Dependencies`.
- Verified the implementation log no longer embeds the local Windows username in command examples.
- No code, build, or runtime behavior changed in this pass.

# Residual risks

- GitHub URLs still contain the repository owner name by design.
- External citations produced by tooling may still use absolute local paths outside repository content.

# Recommended next actions

- If you want stronger anonymization for publication, review screenshots, commit history, and any external release notes separately.
