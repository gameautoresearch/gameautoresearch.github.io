#!/bin/sh
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
uv venv --python 3.12 "$TASK_ROOT/.venv-clash-ar"
uv pip install --python "$TASK_ROOT/.venv-clash-ar/bin/python" -r "$TASK_ROOT/simulators/clash/autoresearch/requirements-host.txt"
