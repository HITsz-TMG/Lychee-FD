#!/usr/bin/env python3
from __future__ import annotations

import sys

from user_interruption_bench_common import DEFAULT_TURN_TAKING_DATA_ROOT
from run_user_interruption_online_client import main


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--subset",
        "turn_taking",
        "--data-root",
        str(DEFAULT_TURN_TAKING_DATA_ROOT),
        *sys.argv[1:],
    ]
    raise SystemExit(main())
