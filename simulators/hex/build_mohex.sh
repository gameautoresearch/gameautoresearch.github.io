#!/bin/sh
# Build MoHex 2.0 from the vendored benzene tree.
#   sh simulators/hex/build_mohex.sh [SOURCE_DIR] [BUILD_DIR]
# Needs cmake, a C++ compiler, Boost (thread system filesystem program_options date_time)
# and Berkeley DB: on Ubuntu `apt-get install cmake g++ libboost-all-dev libdb-dev`;
# on macOS `brew install cmake boost berkeley-db` (keg-only, found through CPATH/LIBRARY_PATH).
# Prints the binary path. The worker image runs this before any GPU is attached.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${1:-$HERE/../../vendor/engines/benzene}
OUT=${2:-$HERE/../../.cache/hex/mohex-build}
if [ "$(uname)" = Darwin ] && command -v brew >/dev/null 2>&1; then
    DB=$(brew --prefix berkeley-db 2>/dev/null || true)
    if [ -n "$DB" ]; then
        export CPATH="$DB/include${CPATH:+:$CPATH}" LIBRARY_PATH="$DB/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    fi
fi
SRC=$(cd "$SRC" && pwd)
mkdir -p "$OUT"
cd "$OUT"
cmake "$SRC" -DCMAKE_BUILD_TYPE=Release >cmake.log 2>&1 || { tail -40 cmake.log >&2; exit 1; }
make -j"${HEX_BUILD_JOBS:-4}" mohex >make.log 2>&1 || { grep -m20 -i error make.log >&2; tail -20 make.log >&2; exit 1; }
echo "$OUT/src/mohex/mohex"
