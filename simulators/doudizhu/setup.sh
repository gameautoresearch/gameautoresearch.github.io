#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
SOURCE="$ROOT/candidates/DouZero"
PIN=718a5c920bf3361e34178a38f3b80458e176b351
if [ ! -d "$SOURCE/.git" ]; then
  git clone https://github.com/kwai/DouZero.git "$SOURCE"
  git -C "$SOURCE" checkout --detach "$PIN"
fi
if [ "$(git -C "$SOURCE" rev-parse HEAD)" != "$PIN" ]; then
  echo 'Existing DouZero checkout differs from pin; refusing to alter it.' >&2
  exit 1
fi
if [ ! -f "$ROOT/.venv-doudizhu/pyvenv.cfg" ]; then
  uv venv --python 3.12.9 "$ROOT/.venv-doudizhu"
fi
uv pip sync --python "$ROOT/.venv-doudizhu/bin/python" --require-hashes "$ROOT/integrations/doudizhu/requirements-universal-lock.txt"
"$ROOT/.venv-doudizhu/bin/python" "$ROOT/integrations/doudizhu/run.py" manifest --out evidence/doudizhu-implementation/setup
