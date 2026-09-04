"""Read-only SQLite execution sandbox with timeouts, row limits and audit data.

``SQLSandbox.execute`` never raises for agent-caused problems: every failure
comes back as a :class:`QueryResult` with ``ok=False`` and a structured,
explanatory ``error``. The agent is supposed to read that error and recover,
so a crash would destroy the very behaviour the environment is trying to
teach.

Three independent defence layers guard the database:

1. :mod:`sql_agent_rl.sandbox.safety` -- lexical screening for good error
   messages and early rejection of obviously disallowed statements.
2. A connection opened ``file:<path>?mode=ro&immutable=1`` with
   ``PRAGMA query_only=ON``. SQLite itself refuses to write.
3. An ``sqlite3`` authorizer that denies every action except reads
   (``SQLITE_SELECT``, ``SQLITE_READ``, ``SQLITE_FUNCTION`` and the like).

Layer 3 does double duty: because SQLite reports each ``SQLITE_READ`` with the
table and column it resolved, the authorizer is also an *exact* record of which
schema objects a query genuinely touched. That is what the rubric's grounding
component uses to tell a real query apart from ``SELECT 'Acme Corp'`` -- no
regex guessing about table names, just the engine's own resolution.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from sql_agent_rl.sandbox.safety import SafetyError, check_statement_safety

DEFAULT_MAX_ROWS = 50
DEFAULT_TIMEOUT_S = 5.0

#: How often SQLite calls the progress handler (in VM instructions). Small
#: enough that the deadline is honoured promptly, large enough not to slow
#: ordinary queries down noticeably.
_PROGRESS_INSTRUCTIONS = 1000

# Authorizer action codes that are safe in a read-only analytical sandbox.
# Everything not listed here is denied. sqlite3 exposes these as module
# constants; a couple are missing on older builds, hence the getattr fallback.
_ALLOWED_ACTIONS = {
    code
    for code in (
        getattr(sqlite3, name, None)
        for name in (
            "SQLITE_SELECT",
            "SQLITE_READ",
            "SQLITE_FUNCTION",
            "SQLITE_RECURSIVE",
        )
    )
    if code is not None
}


class ErrorType(str, Enum):
    """Machine-readable failure categories, logged per tool call."""

    EMPTY_QUERY = "empty_query"
    MULTIPLE_STATEMENTS = "multiple_statements"
    DISALLOWED_STATEMENT = "disallowed_statement"
    FORBIDDEN_KEYWORD = "forbidden_keyword"
    FORBIDDEN_FUNCTION = "forbidden_function"
    NOT_AUTHORIZED = "not_authorized"
    SYNTAX_ERROR = "syntax_error"
    TIMEOUT = "timeout"
    RUNTIME_ERROR = "runtime_error"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True)
class QueryResult:
    """Outcome of one ``execute_sql`` call.

    ``tables_read``/``columns_read`` are populated by the authorizer and record
    what SQLite *actually resolved* while compiling the statement -- the
    ground truth for the rubric's grounding check.
    """

    sql: str
    ok: bool
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[Any, ...], ...] = ()
    row_count: int = 0
    truncated: bool = False
    error: str | None = None
    error_type: ErrorType | None = None
    elapsed_s: float = 0.0
    tables_read: frozenset[str] = field(default_factory=frozenset)
    columns_read: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    #: Catalogue tables (``sqlite_master`` and friends) touched by the query.
    #: Tracked for observability; deliberately excluded from ``tables_read``
    #: so that browsing the catalogue cannot satisfy the grounding check.
    metadata_tables_read: frozenset[str] = field(default_factory=frozenset)

    @property
    def touched_real_tables(self) -> bool:
        """True when the query read at least one real base table.

        ``SELECT 'Acme Corp'`` and ``SELECT 1`` are False; any query that
        joins or scans a Chinook table is True.
        """
        return bool(self.tables_read)

    def to_observation(self) -> str:
        """Render the result as the text the agent sees."""
        if not self.ok:
            return f"SQL ERROR [{self.error_type}]: {self.error}"
        if not self.columns:
            return "Query executed successfully but returned no columns."
        lines = [" | ".join(self.columns)]
        lines.append("-" * len(lines[0]))
        for row in self.rows:
            lines.append(" | ".join("NULL" if v is None else str(v) for v in row))
        if not self.rows:
            lines.append("(no rows)")
        footer = f"({self.row_count} row{'s' if self.row_count != 1 else ''} returned"
        if self.truncated:
            footer += (
                f"; TRUNCATED -- only the first {self.row_count} rows are shown and "
                "the full result set is larger. Aggregate or filter in SQL rather "
                "than relying on these rows being complete"
            )
        footer += ")"
        lines.append(footer)
        return "\n".join(lines)


class SQLSandbox:
    """A read-only, time-limited, row-limited SQL execution surface.

    One instance per episode. Instances are cheap: the database file is opened
    read-only and never modified, so episodes can share the same file without
    snapshotting.
    """

    def __init__(
        self,
        db_path: str | Path,
        max_rows: int = DEFAULT_MAX_ROWS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database not found at {self.db_path}. "
                "Run `python data/download_chinook.py` first."
            )
        self.max_rows = max_rows
        self.timeout_s = timeout_s
        self._con: sqlite3.Connection | None = None
        # Populated by the authorizer during the current execute() call.
        self._tables_read: set[str] = set()
        self._columns_read: set[tuple[str, str]] = set()
        self._metadata_tables_read: set[str] = set()

    # -- connection management ------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._con is None:
            uri = f"file:{self.db_path.resolve().as_posix()}?mode=ro&immutable=1"
            con = sqlite3.connect(uri, uri=True, timeout=self.timeout_s)
            con.execute("PRAGMA query_only = ON")
            con.set_authorizer(self._authorizer)
            self._con = con
        return self._con

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def __enter__(self) -> SQLSandbox:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- authorizer -----------------------------------------------------------

    def _authorizer(
        self,
        action: int,
        arg1: str | None,
        arg2: str | None,
        db_name: str | None,
        trigger: str | None,
    ) -> int:
        if action == getattr(sqlite3, "SQLITE_READ", -1):
            # arg1 = table, arg2 = column. SQLite emits one event per column
            # touched; an empty arg2 means the table was referenced without a
            # specific column (e.g. `COUNT(*)`).
            #
            # `sqlite_master` reads are recorded separately: browsing the
            # catalogue is legitimate, but it is metadata, not data, so it must
            # not satisfy the rubric's "you actually queried the data" check.
            if arg1:
                if arg1.lower().startswith("sqlite_"):
                    self._metadata_tables_read.add(arg1)
                else:
                    self._tables_read.add(arg1)
                    if arg2:
                        self._columns_read.add((arg1, arg2))
            return sqlite3.SQLITE_OK
        if action in _ALLOWED_ACTIONS:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    # -- schema introspection -------------------------------------------------

    def table_names(self) -> list[str]:
        """Base table names, read outside the authorizer (sqlite_master is
        not readable under the read-only authorizer policy)."""
        con = sqlite3.connect(
            f"file:{self.db_path.resolve().as_posix()}?mode=ro", uri=True
        )
        try:
            return [
                r[0]
                for r in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
        finally:
            con.close()

    def describe_schema(self, sample_rows: int = 0) -> str:
        """Render a CREATE-TABLE-style schema description for the agent."""
        con = sqlite3.connect(
            f"file:{self.db_path.resolve().as_posix()}?mode=ro", uri=True
        )
        try:
            tables = [
                r[0]
                for r in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
            blocks: list[str] = []
            for table in tables:
                cols = con.execute(f'PRAGMA table_info("{table}")').fetchall()
                fks = con.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()
                n = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                lines = [f"TABLE {table}  ({n} rows)"]
                for _cid, name, ctype, notnull, _dflt, pk in cols:
                    flags = []
                    if pk:
                        flags.append("PRIMARY KEY")
                    if notnull:
                        flags.append("NOT NULL")
                    suffix = f"  [{', '.join(flags)}]" if flags else ""
                    lines.append(f"  {name} {ctype}{suffix}")
                for fk in fks:
                    lines.append(f"  FOREIGN KEY ({fk[3]}) -> {fk[2]}({fk[4]})")
                if sample_rows:
                    sample = con.execute(
                        f'SELECT * FROM "{table}" LIMIT {int(sample_rows)}'
                    ).fetchall()
                    names = [c[1] for c in cols]
                    for row in sample:
                        pairs = ", ".join(
                            f"{k}={v!r}" for k, v in zip(names, row)
                        )
                        lines.append(f"  -- sample: {pairs}")
                blocks.append("\n".join(lines))
            return "\n\n".join(blocks)
        finally:
            con.close()

    # -- execution ------------------------------------------------------------

    def execute(self, sql: str) -> QueryResult:
        """Run one read-only statement. Never raises for agent-caused errors."""
        sql = "" if sql is None else str(sql)
        try:
            check_statement_safety(sql)
        except SafetyError as exc:
            return QueryResult(
                sql=sql,
                ok=False,
                error=str(exc),
                error_type=ErrorType(exc.reason),
            )

        con = self._connect()
        self._tables_read = set()
        self._columns_read = set()
        self._metadata_tables_read = set()

        deadline = time.monotonic() + self.timeout_s
        con.set_progress_handler(
            lambda: 1 if time.monotonic() > deadline else 0, _PROGRESS_INSTRUCTIONS
        )
        started = time.monotonic()
        try:
            cur = con.execute(sql)
            # Fetch one extra row so truncation can be reported honestly.
            fetched = cur.fetchmany(self.max_rows + 1)
            columns = tuple(d[0] for d in (cur.description or ()))
            truncated = len(fetched) > self.max_rows
            rows = tuple(tuple(r) for r in fetched[: self.max_rows])
            cur.close()
            return QueryResult(
                sql=sql,
                ok=True,
                columns=columns,
                rows=rows,
                row_count=len(rows),
                truncated=truncated,
                elapsed_s=time.monotonic() - started,
                tables_read=frozenset(self._tables_read),
                columns_read=frozenset(self._columns_read),
                metadata_tables_read=frozenset(self._metadata_tables_read),
            )
        except sqlite3.OperationalError as exc:
            message = str(exc)
            if "interrupted" in message.lower():
                etype, message = (
                    ErrorType.TIMEOUT,
                    f"Query exceeded the {self.timeout_s:g}s execution limit and was "
                    "cancelled. Try a more selective query.",
                )
            elif "not authorized" in message.lower():
                etype = ErrorType.NOT_AUTHORIZED
                message = (
                    "Not authorized: this connection may only read data from the "
                    f"sample tables. ({message})"
                )
            else:
                etype = ErrorType.SYNTAX_ERROR
            return QueryResult(
                sql=sql,
                ok=False,
                error=message,
                error_type=etype,
                elapsed_s=time.monotonic() - started,
                tables_read=frozenset(self._tables_read),
                columns_read=frozenset(self._columns_read),
                metadata_tables_read=frozenset(self._metadata_tables_read),
            )
        except sqlite3.DatabaseError as exc:
            return QueryResult(
                sql=sql,
                ok=False,
                error=str(exc),
                error_type=ErrorType.RUNTIME_ERROR,
                elapsed_s=time.monotonic() - started,
                tables_read=frozenset(self._tables_read),
                columns_read=frozenset(self._columns_read),
                metadata_tables_read=frozenset(self._metadata_tables_read),
            )
        finally:
            con.set_progress_handler(None, 0)
