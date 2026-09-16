#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -x .venv/bin/python ]]; then
  echo "Run ./scripts/setup-local.sh first." >&2
  exit 1
fi

exec .venv/bin/python -m gfootball.play_game \
  --action_set=full \
  --level=11_vs_11_kaggle \
  "$@"
