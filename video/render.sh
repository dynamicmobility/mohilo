#!/usr/bin/env bash
# Renders one scene from the repo root, or every scene when none is named.
# Usage: video/render.sh [SCENE] [MANIM FLAGS...]
set -euo pipefail
cd "$(dirname "$0")/.."

SCENES=(HiloScene ParetoScene MogpScene)
# A first argument that is a flag names no scene, so it applies to all of them.
if [[ $# -gt 0 && $1 != -* ]]; then
  SCENES=("$1")
  shift
fi

for scene in "${SCENES[@]}"; do
  case "$scene" in
    HiloScene) file=video/hilo.py ;;
    ParetoScene) file=video/pareto.py ;;
    MogpScene) file=video/mogp.py ;;
    *) echo "unknown scene: $scene" >&2; exit 1 ;;
  esac
  manim -qh ${1+"$@"} "$file" "$scene"
done
