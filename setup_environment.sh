#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"

echo "Setting up Python environment for py-ta..."

# Create ./venv (idempotent)
uv venv venv

# Install into *this* venv regardless of activation state
uv pip install --python ./venv -r requirements.txt

echo "Environment setup complete!"

if command -v direnv >/dev/null 2>&1 && [ -f .envrc ]; then
  echo "direnv detected. If first time, run: direnv allow"
  echo "After that, entering this directory will auto-activate the venv."
else
  echo "To activate manually: source venv/bin/activate"
fi
