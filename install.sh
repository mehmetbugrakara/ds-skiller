#!/usr/bin/env bash
# ds-skiller installer (macOS / Linux)
# Renders each skills/ds-*/SKILL.md with this repo's absolute path into every agent's
# skills folder. Scripts stay in the repo (git pull updates all agents). Re-run after moving the repo.
#   ./install.sh               install
#   ./install.sh --uninstall   remove
#   DS_TARGETS="/a/skills /b/skills" ./install.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
UNINSTALL=0
[[ "${1:-}" == "--uninstall" ]] && UNINSTALL=1

if [[ -n "${DS_TARGETS:-}" ]]; then
  read -r -a TARGETS <<< "$DS_TARGETS"
else
  TARGETS=("$HOME/.claude/skills")
  [[ -d "$HOME/.codex" ]] && TARGETS+=("$HOME/.codex/skills")
  [[ -d "$HOME/.gemini" ]] && TARGETS+=("$HOME/.gemini/antigravity/skills")
fi

for t in "${TARGETS[@]}"; do
  mkdir -p "$t"
  n=0
  for s in "$REPO_ROOT"/skills/ds-*/; do
    name="$(basename "$s")"
    rm -rf "${t:?}/$name"
    n=$((n + 1))
    [[ $UNINSTALL -eq 1 ]] && continue
    mkdir -p "$t/$name"
    sed "s|{{DS_SKILLER_ROOT}}|$REPO_ROOT|g" "$s/SKILL.md" > "$t/$name/SKILL.md"
  done
  if [[ $UNINSTALL -eq 1 ]]; then echo "Removed $n ds-* skills -> $t"; else echo "Installed $n ds-* skills -> $t"; fi
done
[[ $UNINSTALL -eq 0 ]] && { command -v uv >/dev/null || echo "WARN: uv not found (https://docs.astral.sh/uv/)"; echo "Repo root: $REPO_ROOT (run 'uv sync' there once)"; }
