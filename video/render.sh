#!/usr/bin/env bash
# Renders one scene from the repo root. Usage: video/render.sh [SCENE] [MANIM FLAGS...]
set -euo pipefail
cd "$(dirname "$0")/.."
SCENE=${1:-HiloScene}
shift || true
case "$SCENE" in
  HiloScene) FILE=video/hilo.py ;;
  ParetoScene) FILE=video/pareto.py ;;
  ValidationScene) FILE=video/validation.py ;;
  *) echo "unknown scene: $SCENE" >&2; exit 1 ;;
esac
exec manim -qh "${@}" "$FILE" "$SCENE"
