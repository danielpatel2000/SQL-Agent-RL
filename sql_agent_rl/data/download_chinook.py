#!/usr/bin/env python3
"""
Download the Chinook sample SQLite database.

Chinook is a digital-media-store sample database (customers, invoices,
invoice_items, tracks, albums, artists, employees, playlists) maintained at:
https://github.com/lerocha/chinook-database

No external dependencies required (uses only the standard library).

Usage:
    python download_chinook.py [--output PATH] [--force]
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Primary source is the raw file straight off GitHub's content host.
# Fallbacks cover a redirect-based URL and a long-lived fork, in case the
# primary path or branch name ever changes.
CHINOOK_URLS = [
    "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite",
    "https://github.com/lerocha/chinook-database/raw/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite",
    "https://raw.githubusercontent.com/jimfrenette/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite",
]


def download(url: str, dest: Path, chunk_size: int = 1 << 16) -> None:
    req = Request(url, headers={"User-Agent": "chinook-downloader/1.0"})
    with urlopen(req, timeout=30) as response, open(dest, "wb") as f:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        while chunk := response.read(chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                print(f"\r  {downloaded / 1e6:6.2f} MB / {total / 1e6:.2f} MB ({pct}%)", end="")
    print()


def verify(dest: Path) -> None:
    """Open the DB and print table names + row counts as a sanity check."""
    con = sqlite3.connect(dest)
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cur.fetchall()]
    if not tables:
        raise RuntimeError("Downloaded file has no tables — likely a failed or corrupted download.")
    print(f"Verified {len(tables)} tables:")
    for t in tables:
        cur.execute(f'SELECT COUNT(*) FROM "{t}"')
        print(f"  - {t}: {cur.fetchone()[0]} rows")
    con.close()


def fetch(output: Path, force: bool = False) -> Path:
    """Programmatic entry point used by tests/fixtures.

    Downloads the database to ``output`` unless it already exists (and
    ``force`` is False). Returns the path to the verified database.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        return output
    errors = []
    for url in CHINOOK_URLS:
        try:
            download(url, output)
            verify(output)
            return output
        except (URLError, HTTPError, RuntimeError, TimeoutError, sqlite3.Error) as e:
            errors.append(f"{url}: {e}")
            output.unlink(missing_ok=True)
    raise RuntimeError("All Chinook download sources failed:\n  " + "\n  ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", type=Path, default=Path("data/chinook.db"),
                         help="Destination path (default: data/chinook.db)")
    parser.add_argument("--force", action="store_true",
                         help="Re-download even if the file already exists")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.output.exists() and not args.force:
        print(f"{args.output} already exists — use --force to re-download.\n")
        verify(args.output)
        return

    for url in CHINOOK_URLS:
        print(f"Downloading from {url}")
        try:
            download(url, args.output)
            verify(args.output)
            print(f"\nSaved to {args.output.resolve()}")
            return
        except (URLError, HTTPError, RuntimeError, TimeoutError) as e:
            print(f"  Failed: {e}")
            args.output.unlink(missing_ok=True)

    print("\nAll download sources failed. Grab it manually from:", file=sys.stderr)
    print("  https://github.com/lerocha/chinook-database", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
