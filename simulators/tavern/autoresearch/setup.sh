#!/bin/sh
set -eu
cd "$(dirname "$0")/../../.."
uv venv --python 3.12 .venv-tavern-ar
uv pip install --python .venv-tavern-ar/bin/python -r simulators/tavern/autoresearch/requirements-host.txt
