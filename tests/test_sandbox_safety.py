"""Safety-layer tests: destructive statements blocked, limits enforced."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from sql_agent_rl.sandbox import ErrorType, SQLSandbox, SafetyError, check_statement_safety

DESTRUCTIVE = [
    "DROP TABLE Customer",
    "DELETE FROM Customer",
    "DELETE FROM Customer WHERE CustomerId = 1",
    "UPDATE Customer SET LastName = 'x'",
    "INSERT INTO Genre (Name) VALUES ('x')",
    "REPLACE INTO Genre (GenreId, Name) VALUES (1, 'x')",
    "CREATE TABLE evil (a INT)",
    "CREATE VIEW evil AS SELECT 1",
    "ALTER TABLE Customer ADD COLUMN evil TEXT",
    "ATTACH DATABASE '/tmp/evil.db' AS evil",
    "DETACH DATABASE main",
    "PRAGMA writable_schema = ON",
    "VACUUM",
    "REINDEX",
    "ANALYZE",
    "BEGIN; DELETE FROM Customer; COMMIT",
    "WITH t AS (SELECT 1) INSERT INTO Genre SELECT 99, 'x'",
    "SELECT 1; DROP TABLE Customer",
    "SELECT 1;\n-- innocuous\nDROP TABLE Customer",
    "select 1; drop table Customer",
    "SELECT load_extension('/tmp/evil.so')",
    "SELECT writefile('/tmp/pwned', 'x')",
    "SELECT readfile('/etc/passwd')",
]


@pytest.mark.parametrize("stmt", DESTRUCTIVE)
def test_destructive_statements_are_blocked(sandbox: SQLSandbox, stmt: str) -> None:
    result = sandbox.execute(stmt)
    assert not result.ok, f"expected {stmt!r} to be rejected"
    assert result.error_type in {
        ErrorType.DISALLOWED_STATEMENT,
        ErrorType.FORBIDDEN_KEYWORD,
        ErrorType.FORBIDDEN_FUNCTION,
        ErrorType.MULTIPLE_STATEMENTS,
        ErrorType.NOT_AUTHORIZED,
    }
    # The rejection must be an explanatory observation, not a silent failure.
    assert result.error and len(result.error) > 20
    assert result.rows == ()


@pytest.mark.parametrize("stmt", DESTRUCTIVE)
def test_destructive_statements_leave_data_intact(
    sandbox: SQLSandbox, chinook_db: Path, stmt: str
) -> None:
    """Belt and braces: prove the row counts really are unchanged."""
    before = sandbox.execute("SELECT COUNT(*) FROM Customer").rows
    sandbox.execute(stmt)
    after = sandbox.execute("SELECT COUNT(*) FROM Customer").rows
    assert before == after == ((59,),)


def test_read_only_connection_rejects_writes_even_without_static_screen(
    chinook_db: Path,
) -> None:
    """Layer 1 is not load-bearing: layers 2 and 3 block writes on their own."""
    sb = SQLSandbox(chinook_db)
    con = sb._connect()  # noqa: SLF001 - deliberately testing the raw connection
    with pytest.raises(sqlite3.DatabaseError):
        con.execute("DELETE FROM Customer")
    with pytest.raises(sqlite3.DatabaseError):
        con.execute("CREATE TABLE evil (a INT)")
    sb.close()


def test_empty_and_comment_only_queries_are_rejected(sandbox: SQLSandbox) -> None:
    for stmt in ["", "   ", "-- just a comment", "/* nothing */"]:
        result = sandbox.execute(stmt)
        assert not result.ok
        assert result.error_type == ErrorType.EMPTY_QUERY


def test_string_literals_do_not_trip_the_keyword_screen(sandbox: SQLSandbox) -> None:
    result = sandbox.execute("SELECT 'DROP TABLE Customer' AS harmless")
    assert result.ok, result.error


def test_case_end_and_replace_are_allowed(sandbox: SQLSandbox) -> None:
    """Regression: CASE...END and replace() are ordinary analytical SQL."""
    result = sandbox.execute(
        "SELECT CASE WHEN Total > 5 THEN 'big' ELSE 'small' END AS bucket, "
        "replace(BillingCountry, 'USA', 'US') AS c FROM Invoice LIMIT 3"
    )
    assert result.ok, result.error
    assert result.row_count == 3


def test_allowed_statement_kinds(sandbox: SQLSandbox) -> None:
    for stmt in [
        "SELECT COUNT(*) FROM Invoice",
        "WITH t AS (SELECT CustomerId FROM Invoice) SELECT COUNT(*) FROM t",
        "EXPLAIN QUERY PLAN SELECT * FROM Customer",
    ]:
        assert sandbox.execute(stmt).ok, stmt


def test_row_limit_truncates_and_says_so(chinook_db: Path) -> None:
    with SQLSandbox(chinook_db, max_rows=10) as sb:
        result = sb.execute("SELECT TrackId FROM Track")
        assert result.ok
        assert result.row_count == 10
        assert result.truncated
        assert "TRUNCATED" in result.to_observation()


def test_row_limit_not_flagged_when_result_fits(chinook_db: Path) -> None:
    with SQLSandbox(chinook_db, max_rows=10) as sb:
        result = sb.execute("SELECT TrackId FROM Track LIMIT 4")
        assert result.row_count == 4
        assert not result.truncated
        assert "TRUNCATED" not in result.to_observation()


def test_execution_timeout_is_enforced(chinook_db: Path) -> None:
    with SQLSandbox(chinook_db, timeout_s=1.0) as sb:
        started = time.monotonic()
        result = sb.execute(
            "SELECT COUNT(*) FROM Invoice a, Invoice b, Invoice c, "
            "InvoiceLine d, InvoiceLine e"
        )
        elapsed = time.monotonic() - started
    assert not result.ok
    assert result.error_type == ErrorType.TIMEOUT
    assert elapsed < 5.0, "timeout did not fire promptly"


def test_errors_are_structured_not_exceptions(sandbox: SQLSandbox) -> None:
    result = sandbox.execute("SELECT * FROM NoSuchTable")
    assert not result.ok
    assert result.error_type == ErrorType.SYNTAX_ERROR
    assert "NoSuchTable" in result.error

    result = sandbox.execute("SELECT COUNT( FROM Invoice")
    assert not result.ok
    assert result.error_type == ErrorType.SYNTAX_ERROR


def test_authorizer_records_tables_actually_read(sandbox: SQLSandbox) -> None:
    result = sandbox.execute(
        "SELECT c.LastName, SUM(il.UnitPrice * il.Quantity) AS revenue "
        "FROM Customer c "
        "JOIN Invoice i ON i.CustomerId = c.CustomerId "
        "JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId "
        "GROUP BY c.CustomerId"
    )
    assert result.ok, result.error
    assert result.tables_read == frozenset({"Customer", "Invoice", "InvoiceLine"})
    assert ("InvoiceLine", "UnitPrice") in result.columns_read
    assert result.touched_real_tables


def test_literal_only_query_touches_no_tables(sandbox: SQLSandbox) -> None:
    """The signal the grounding rubric relies on."""
    result = sandbox.execute("SELECT 'Acme Corp' AS name, 123.45 AS revenue")
    assert result.ok
    assert result.tables_read == frozenset()
    assert not result.touched_real_tables


def test_catalogue_reads_do_not_count_as_data_reads(sandbox: SQLSandbox) -> None:
    result = sandbox.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    assert result.ok
    assert result.tables_read == frozenset()
    assert not result.touched_real_tables
    assert "sqlite_master" in result.metadata_tables_read


def test_check_statement_safety_returns_leading_keyword() -> None:
    assert check_statement_safety("SELECT 1") == "SELECT"
    assert check_statement_safety("  with t as (select 1) select * from t") == "WITH"
    with pytest.raises(SafetyError) as exc:
        check_statement_safety("DROP TABLE x")
    assert exc.value.reason == "disallowed_statement"


def test_describe_schema_lists_tables_and_keys(sandbox: SQLSandbox) -> None:
    schema = sandbox.describe_schema()
    for table in ["Customer", "Invoice", "InvoiceLine", "Track", "Album", "Artist"]:
        assert f"TABLE {table}" in schema
    assert "FOREIGN KEY" in schema
    assert "CustomerId" in schema
