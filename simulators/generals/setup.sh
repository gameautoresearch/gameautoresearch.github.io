#!/usr/bin/env bash
set -euo pipefail
GENERALS_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$GENERALS_ROOT"
GENERALS_PIN=954927f508aa47899019d66317cc313af3e9cc19
if [[ ! -d candidates/generals-bots/.git ]]; then
  git clone https://github.com/strakam/generals-bots.git candidates/generals-bots
  git -C candidates/generals-bots checkout --detach "$GENERALS_PIN"
fi
if [[ "$(git -C candidates/generals-bots rev-parse HEAD)" != "$GENERALS_PIN" ]]; then
  echo "Existing engine differs from frozen pin; preserve checkout and reconcile explicitly." >&2
  exit 1
fi
if [[ -n "$(git -C candidates/generals-bots status --porcelain --untracked-files=no)" ]]; then
  echo "Existing upstream tracked edits; refusing installation." >&2
  exit 1
fi
command -v uv >/dev/null
if [[ ! -d .venv-generals ]]; then
  uv venv --python 3.12 .venv-generals
fi
uv pip sync --python .venv-generals/bin/python integrations/generals/requirements-local.lock
.venv-generals/bin/python -c 'import jax, platform; print(platform.machine(), jax.__version__, jax.devices())'
