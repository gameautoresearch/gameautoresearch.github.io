#!/usr/bin/env bash
# Run ONLY as the child of lease.py on an already authorized, externally fixed
# one-GPU/CPU/RAM reservation. This script never provisions or purchases compute.
set -euo pipefail
GENERALS_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$GENERALS_ROOT"
if [[ -z "${GENERALS_ALLOCATION_START_UNIX:-}" ]]; then
  echo "Run this worker under integrations.generals.lease." >&2
  exit 1
fi
GENERALS_RECIPE="${1:-integrations/generals/recipes/starter.json}"
GENERALS_OUTPUT="${2:-evidence/generals-implementation/starter-gpu}"
# Installation, including downloading/building CUDA dependencies, consumes the
# same allocated-worker clock as training and evaluation.
integrations/generals/setup.sh
uv pip install --python .venv-generals/bin/python 'jax[cuda12]==0.12.0.1'
.venv-generals/bin/python -m integrations.generals train --device gpu \
  --recipe "$GENERALS_RECIPE" --output "$GENERALS_OUTPUT"
.venv-generals/bin/python -m integrations.generals evaluate --device gpu \
  --recipe "$GENERALS_RECIPE" --checkpoint "$GENERALS_OUTPUT/latest.msgpack" \
  --split heldout --output "$GENERALS_OUTPUT/heldout"
