#!/usr/bin/env python3
"""Start the allocation clock before heavy imports; spawn children avoid torch import."""
import sys
import time
from pathlib import Path

if __name__ == '__main__':
    started_perf, started_unix = time.monotonic(), time.time()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from integrations.doudizhu.runner import main
    main(started_perf, started_unix)
