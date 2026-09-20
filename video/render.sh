#!/usr/bin/env bash
# Renders one scene from the repo root. Usage: video/render.sh [SCENE] [MANIM FLAGS...]
set -euo pipefail
cd "$(dirname "$0")/.."
SCENE=${1:-HiloScene}
shift || true
exec manim -qh "${@}" video/hilo.py "$SCENE"
