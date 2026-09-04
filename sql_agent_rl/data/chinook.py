"""Pandas view of the Chinook database, used to compute ground truth.

Everything the grader needs is derived here, in Python, from the raw tables --
never from SQL the agent (or a reference solution) wrote. That independence is
the whole point: a reward that checks "does this match my reference query's
output" only tests SQL equivalence to one solution, not correctness.

Canonical revenue definition
----------------------------
Revenue is ``SUM(InvoiceLine.UnitPrice * InvoiceLine.Quantity)``.

Chinook offers a second route -- ``SUM(Invoice.Total)`` -- and the two agree
exactly on this build (all 412 invoices, grand total 2328.60 both ways; see
``tests/test_ground_truth.py::test_two_revenue_definitions_agree``). The
line-item source is canonical because it is the more granular one and because
it forces the agent through a real ``Customer -> Invoice -> InvoiceLine`` join
rather than reading a pre-aggregated column.

Currency values are rounded half-up to 2 decimal places at the *end* of an
aggregation, never mid-way.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from pathlib import Path

import pandas as pd

#: Tables the environment considers "the dataset". Playlist/PlaylistTrack are
#: present in Chinook but carry no revenue signal, so no template uses them.
CORE_TABLES = (
    "Album",
    "Artist",
    "Customer",
    "Employee",
    "Genre",
    "Invoice",
    "InvoiceLine",
    "MediaType",
    "Track",
)


def round_currency(value: float) -> float:
    """Round half-up to cents.

    Python's built-in ``round`` is banker's rounding (``round(0.125, 2) == 0.12``),
    which is the wrong convention for money and would make ground truth differ
    from what a SQL engine reports for the same aggregate.
    """
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class ChinookFrames:
    """Raw tables plus a denormalised line-level ``sales`` frame.

    ``sales`` has one row per ``InvoiceLine`` and carries every dimension the
    task templates aggregate over, so each ground-truth computation stays a
    short, obviously-correct groupby.
    """

    customer: pd.DataFrame
    invoice: pd.DataFrame
    invoice_line: pd.DataFrame
    track: pd.DataFrame
    album: pd.DataFrame
    artist: pd.DataFrame
    genre: pd.DataFrame
    employee: pd.DataFrame
    sales: pd.DataFrame

    # -- discovered facts about this particular build -------------------------

    @property
    def years(self) -> list[int]:
        """Calendar years that actually contain invoices, ascending.

        Discovered from the data -- never hardcoded. Upstream Chinook shifts
        its invoice dates forward periodically, so any literal year would rot.
        """
        return sorted(int(y) for y in self.sales["year"].unique())

    @property
    def countries(self) -> list[str]:
        return sorted(self.customer["Country"].dropna().unique().tolist())

    @property
    def genres(self) -> list[str]:
        return sorted(self.genre["Name"].dropna().unique().tolist())

    def date_range(self) -> tuple[str, str]:
        dates = self.invoice["InvoiceDate"]
        return str(dates.min()), str(dates.max())

    def customer_display_name(self, customer_id: int) -> str:
        row = self.customer.loc[self.customer["CustomerId"] == customer_id].iloc[0]
        return f"{row['FirstName']} {row['LastName']}"


def _read_all(db_path: Path) -> dict[str, pd.DataFrame]:
    con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        return {t: pd.read_sql_query(f'SELECT * FROM "{t}"', con) for t in CORE_TABLES}
    finally:
        con.close()


def _build_sales(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    invoice = tables["Invoice"].copy()
    invoice["InvoiceDate"] = pd.to_datetime(invoice["InvoiceDate"])
    invoice["year"] = invoice["InvoiceDate"].dt.year.astype(int)
    invoice["quarter"] = invoice["InvoiceDate"].dt.quarter.astype(int)

    customer = tables["Customer"].copy()
    customer["CustomerName"] = (
        customer["FirstName"].fillna("") + " " + customer["LastName"].fillna("")
    ).str.strip()

    sales = (
        tables["InvoiceLine"]
        .merge(
            invoice[["InvoiceId", "CustomerId", "InvoiceDate", "year", "quarter"]],
            on="InvoiceId",
            how="left",
            validate="many_to_one",
        )
        .merge(
            customer[["CustomerId", "CustomerName", "Country"]],
            on="CustomerId",
            how="left",
            validate="many_to_one",
        )
        .merge(
            tables["Track"][["TrackId", "Name", "AlbumId", "GenreId"]].rename(
                columns={"Name": "TrackName"}
            ),
            on="TrackId",
            how="left",
            validate="many_to_one",
        )
        .merge(
            tables["Album"][["AlbumId", "ArtistId", "Title"]].rename(
                columns={"Title": "AlbumTitle"}
            ),
            on="AlbumId",
            how="left",
            validate="many_to_one",
        )
        .merge(
            tables["Artist"][["ArtistId", "Name"]].rename(columns={"Name": "ArtistName"}),
            on="ArtistId",
            how="left",
            validate="many_to_one",
        )
        .merge(
            tables["Genre"][["GenreId", "Name"]].rename(columns={"Name": "GenreName"}),
            on="GenreId",
            how="left",
            validate="many_to_one",
        )
    )
    sales["revenue"] = sales["UnitPrice"] * sales["Quantity"]
    return sales


@lru_cache(maxsize=4)
def _load_cached(resolved: str, mtime: float, size: int) -> ChinookFrames:
    # mtime/size are cache-key components only: if the file is re-downloaded,
    # the cache misses and the frames are rebuilt.
    del mtime, size
    tables = _read_all(Path(resolved))
    return ChinookFrames(
        customer=tables["Customer"],
        invoice=tables["Invoice"],
        invoice_line=tables["InvoiceLine"],
        track=tables["Track"],
        album=tables["Album"],
        artist=tables["Artist"],
        genre=tables["Genre"],
        employee=tables["Employee"],
        sales=_build_sales(tables),
    )


def load_chinook(db_path: str | Path) -> ChinookFrames:
    """Load (and cache) the Chinook tables as pandas frames."""
    path = Path(db_path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Database not found at {path}. Run `python data/download_chinook.py` first."
        )
    stat = path.stat()
    return _load_cached(path.as_posix(), stat.st_mtime, stat.st_size)
