#!/bin/sh
set -eu
GENERALS_AR_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
while [ ! -f "$GENERALS_AR_ROOT/tools/restore_upstream_metadata.py" ]; do
  if [ "$GENERALS_AR_ROOT" = / ]; then echo 'Repository root not found.' >&2; exit 1; fi
  GENERALS_AR_ROOT=$(dirname "$GENERALS_AR_ROOT")
done
cd "$GENERALS_AR_ROOT"
if ! git -C candidates/generals-bots rev-parse HEAD >/dev/null 2>&1; then
  echo 'Restore offline upstream metadata with python3 tools/restore_upstream_metadata.py first.' >&2
  exit 1
fi
if [ ! -x .venv-generals-ar/bin/python ]; then uv venv --python 3.12.9 .venv-generals-ar; fi
uv pip install --python .venv-generals-ar/bin/python -r integrations/generals_autoresearch/requirements-host.txt
uv pip install --python .venv-generals-ar/bin/python --no-deps -e candidates/generals-bots
.venv-generals-ar/bin/python -c 'from integrations.generals_autoresearch.contract import frozen_manifest; print("Pinned engine:",frozen_manifest()["commit"])'
