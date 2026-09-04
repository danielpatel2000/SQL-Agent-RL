#!/usr/bin/env python3
"""Runnable shim so `python data/download_chinook.py` works from a clean checkout.

The canonical implementation lives in ``sql_agent_rl/data/download_chinook.py``
so it can also be imported by test fixtures (``fetch()``).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sql_agent_rl.data.download_chinook import main  # noqa: E402

if __name__ == "__main__":
    main()
