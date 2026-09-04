"""Static safety checks applied to agent-supplied SQL before it reaches SQLite.

This is the *first* of three defence layers (see ``executor.SQLSandbox``):

1. this module -- lexical screening, so the agent gets a clear, early,
   explanatory error rather than an opaque driver exception;
2. a read-only connection (``file:...?mode=ro`` plus ``PRAGMA query_only``);
3. an ``sqlite3`` authorizer callback that vetoes every operation other than
   reads.

Layer 1 alone is not trusted for security -- lexical screening of SQL is
notoriously leaky. It exists for *observation quality*: the agent should be
told "DROP is not allowed here, this database is read-only" instead of
"attempt to write a readonly database". Layers 2 and 3 are the ones that
actually make an escape impossible.
"""

from __future__ import annotations

import re

#: Statements the agent is allowed to start. Everything else is rejected.
ALLOWED_LEADING_KEYWORDS = frozenset({"SELECT", "WITH", "EXPLAIN", "VALUES"})

#: Keywords that must not appear anywhere outside a string literal. These are
#: either mutating (INSERT/DROP/...), or escape hatches out of the sandboxed
#: database file (ATTACH/DETACH), or capable of changing engine behaviour
#: (PRAGMA).
#:
#: Deliberately *absent*, because they collide with ordinary analytical SQL and
#: are already neutralised by layers 2 and 3: ``BEGIN``/``END`` (``CASE ... END``
#: is ubiquitous) and ``REPLACE`` (a standard scalar string function; a
#: ``REPLACE INTO`` statement is still rejected by the leading-keyword check).
FORBIDDEN_KEYWORDS = frozenset(
    {
        "ALTER",
        "ANALYZE",
        "ATTACH",
        "COMMIT",
        "CREATE",
        "DELETE",
        "DETACH",
        "DROP",
        "INSERT",
        "PRAGMA",
        "REINDEX",
        "RELEASE",
        "ROLLBACK",
        "SAVEPOINT",
        "TRUNCATE",
        "UPDATE",
        "UPSERT",
        "VACUUM",
    }
)

#: SQLite functions that touch the filesystem or the loader. Standard CPython
#: builds do not expose most of these, but a build with the CLI extensions
#: loaded would, and ``load_extension`` is reachable if it is ever enabled.
FORBIDDEN_FUNCTIONS = frozenset(
    {
        "edit",
        "fts3_tokenizer",
        "load_extension",
        "readfile",
        "writefile",
        "zipfile",
    }
)

_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_STRING_LITERAL = re.compile(
    r"'(?:[^']|'')*'"  # standard single-quoted strings, '' escapes
    r"|\"(?:[^\"]|\"\")*\""  # double-quoted identifiers (or strings, in SQLite)
    r"|\[[^\]]*\]"  # MSSQL-style bracket identifiers
    r"|`[^`]*`"  # MySQL-style backtick identifiers
)
_WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


class SafetyError(Exception):
    """Raised when a statement fails the static safety screen.

    ``reason`` is a short machine-readable slug; ``str(exc)`` is the
    human-readable message shown to the agent.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def strip_sql_noise(sql: str) -> str:
    """Blank out comments and quoted spans, preserving offsets loosely.

    Quoted spans are replaced by a placeholder rather than removed so that
    ``SELECT 'DROP TABLE x'`` does not trip the keyword screen, while
    ``SELECT * FROM t; DROP TABLE x`` still does.
    """
    sql = _BLOCK_COMMENT.sub(" ", sql)
    sql = _LINE_COMMENT.sub(" ", sql)
    return _STRING_LITERAL.sub(" '' ", sql)


def _split_statements(scrubbed: str) -> list[str]:
    """Split on semicolons, dropping empty trailing fragments."""
    return [part for part in (p.strip() for p in scrubbed.split(";")) if part]


def check_statement_safety(sql: str) -> str:
    """Validate an agent-supplied statement.

    Returns the leading keyword (upper-cased) on success.
    Raises :class:`SafetyError` otherwise.
    """
    if sql is None or not str(sql).strip():
        raise SafetyError("empty_query", "Empty query. Provide a SELECT statement.")

    scrubbed = strip_sql_noise(sql)
    statements = _split_statements(scrubbed)

    if not statements:
        raise SafetyError(
            "empty_query",
            "Query contains no executable statement (only comments or whitespace).",
        )
    if len(statements) > 1:
        raise SafetyError(
            "multiple_statements",
            f"Only one statement may be executed per call, got {len(statements)}. "
            "Remove the extra statements (and any trailing text after the first ';').",
        )

    body = statements[0]
    words = _WORD.findall(body)
    if not words:
        raise SafetyError("empty_query", "Query contains no SQL keywords.")

    leading = words[0].upper()
    if leading not in ALLOWED_LEADING_KEYWORDS:
        raise SafetyError(
            "disallowed_statement",
            f"Statement type '{leading}' is not allowed. This database is read-only; "
            f"queries must start with one of: {', '.join(sorted(ALLOWED_LEADING_KEYWORDS))}.",
        )

    upper_words = {w.upper() for w in words}
    banned = sorted(upper_words & FORBIDDEN_KEYWORDS)
    if banned:
        raise SafetyError(
            "forbidden_keyword",
            f"Query contains forbidden keyword(s): {', '.join(banned)}. "
            "This database is read-only -- only SELECT/WITH/EXPLAIN queries are permitted.",
        )

    lowered = {w.lower() for w in words}
    banned_fns = sorted(lowered & FORBIDDEN_FUNCTIONS)
    if banned_fns:
        raise SafetyError(
            "forbidden_function",
            f"Query references forbidden function(s): {', '.join(banned_fns)}.",
        )

    return leading
