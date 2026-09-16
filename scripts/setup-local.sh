#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

if [[ "${1:-}" != "--skip-system-packages" ]]; then
  system_packages=(
    build-essential
    cmake
    libboost-filesystem-dev
    libboost-python-dev
    libboost-system-dev
    libboost-thread-dev
    libgl1-mesa-dev
    libsdl2-dev
    libsdl2-gfx-dev
    libsdl2-image-dev
    libsdl2-ttf-dev
    libst-dev
    mesa-utils
    python3-dev
    x11vnc
    xvfb
  )
  missing_packages=()
  for package in "${system_packages[@]}"; do
    if ! dpkg-query -W -f='${db:Status-Abbrev}' "$package" 2>/dev/null | grep -q '^ii '; then
      missing_packages+=("$package")
    fi
  done

  if (( ${#missing_packages[@]} )); then
    sudo apt-get update
    sudo apt-get install -y "${missing_packages[@]}"
  fi
fi

uv venv --allow-existing --python /usr/bin/python3 .venv
uv pip install --python .venv/bin/python \
  'setuptools==84.0.0' \
  'wheel==0.48.0' \
  'psutil==7.2.2' \
  'gym==0.26.2' \
  'pygame==2.6.1' \
  'opencv-python==5.0.0.93' \
  'absl-py==2.5.0' \
  'numpy==2.5.3' \
  'six==1.17.0'
uv pip install --python .venv/bin/python --no-build-isolation --no-deps .

echo "Setup complete. Start the game with ./scripts/play-kaggle.sh"
