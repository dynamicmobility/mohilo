#!/usr/bin/env bash
# Renders one scene from the repo root, or every scene when none is named.
# Usage: video/render.sh [SCENE] [MANIM FLAGS...]
set -euo pipefail
cd "$(dirname "$0")/.."

SCENES=(HiloScene ParetoScene MogpScene ValidationScene MethodScene FrontScene ParetoSetsScene SubjectsGridScene SubjectActionsScene SubjectValidationScene TradeoffScene ResultTableScene)
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
    ValidationScene) file=video/validation.py ;;
    MethodScene) file=video/method.py ;;
    FrontScene) file=video/front.py ;;
    ParetoSetsScene) file=video/pareto_sets.py ;;
    SubjectsGridScene) file=video/subjects_grid.py ;;
    SubjectActionsScene) file=video/subject_actions.py ;;
    SubjectValidationScene) file=video/subjects_validation.py ;;
    TradeoffScene) file=video/tradeoff.py ;;
    ResultTableScene) file=video/result_table.py ;;
    *) echo "unknown scene: $scene" >&2; exit 1 ;;
  esac
  manim -qh ${1+"$@"} "$file" "$scene"
done
