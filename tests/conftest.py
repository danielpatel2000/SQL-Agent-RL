"""Shared fixtures.

The Chinook database is downloaded **once** per test session (and cached on
disk between sessions) rather than regenerated per test: it is a fixed, real
dataset, not synthetic data with a seed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sql_agent_rl.data.download_chinook import fetch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "data" / "chinook.db"


@pytest.fixture(scope="session")
def chinook_db() -> Path:
    """Path to the Chinook database, downloading it if absent."""
    path = Path(os.environ.get("CHINOOK_DB", DEFAULT_DB))
    if not path.exists():
        if os.environ.get("SQL_AGENT_RL_NO_DOWNLOAD"):
            pytest.skip(f"{path} missing and downloads are disabled")
        fetch(path)
    return path


@pytest.fixture(scope="session")
def sandbox(chinook_db: Path):
    from sql_agent_rl.sandbox import SQLSandbox

    with SQLSandbox(chinook_db) as sb:
        yield sb
