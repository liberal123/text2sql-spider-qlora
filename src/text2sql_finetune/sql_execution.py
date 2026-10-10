"""Read-only SQLite execution and a deliberately custom Text-to-SQL scorer.

This is not Spider's official evaluator. Text match only normalizes comments,
unquoted token case, whitespace and one trailing semicolon. Execution match
compares complete result rows, keeping NULL, string values and duplicates.
Only a top-level ORDER BY in the reference SQL makes row order significant.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Sequence


def _sql_tokens(sql: str) -> list[str]:
    """Tokenize enough SQL to normalize text without rewriting literals."""
    tokens: list[str] = []
    i = 0
    while i < len(sql):
        char = sql[i]
        if char.isspace():
            i += 1
            continue
        if sql.startswith("--", i):
            end = sql.find("\n", i + 2)
            i = len(sql) if end < 0 else end + 1
            continue
        if sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            if end < 0:
                tokens.append(sql[i:])
                break
            i = end + 2
            continue
        if char in ("'", '"', "`", "["):
            start = i
            closer = "]" if char == "[" else char
            i += 1
            while i < len(sql):
                if sql[i] == closer:
                    i += 1
                    if closer != "]" and i < len(sql) and sql[i] == closer:
                        i += 1
                        continue
                    break
                i += 1
            tokens.append(sql[start:i])
            continue
        if char.isalpha() or char == "_":
            start = i
            i += 1
            while i < len(sql) and (sql[i].isalnum() or sql[i] in "_$"):
                i += 1
            tokens.append(sql[start:i].casefold())
            continue
        if char.isdigit():
            match = re.match(r"\d+(?:\.\d*)?(?:[eE][+-]?\d+)?", sql[i:])
            assert match is not None
            tokens.append(match.group(0).casefold())
            i += len(match.group(0))
            continue
        operator = next(
            (op for op in ("->>", "<=", ">=", "!=", "<>", "||", "<<", ">>", "->")
             if sql.startswith(op, i)),
            None,
        )
        if operator is not None:
            tokens.append(operator)
            i += len(operator)
        else:
            tokens.append(char)
            i += 1
    return tokens


def normalize_sql_text(sql: str) -> str:
    """Conservative textual normalization; case inside quotes stays intact."""
    tokens = _sql_tokens(sql)
    if tokens and tokens[-1] == ";":
        tokens.pop()
    return " ".join(tokens)


# Compatibility for callers that historically imported formatting.normalize_sql.
normalize_sql = normalize_sql_text


def has_top_level_order_by(sql: str) -> bool:
    tokens = _sql_tokens(sql)
    depth = 0
    for index, token in enumerate(tokens):
        if token == "(":
            depth += 1
        elif token == ")":
            depth = max(0, depth - 1)
        elif (depth == 0 and token == "order" and index + 1 < len(tokens)
              and tokens[index + 1] == "by"):
            return True
    return False


def _typed_row(row: Sequence[Any]) -> tuple[tuple[str, Any], ...]:
    # No string conversion: NULL != '', 1 != '1', and duplicate rows survive.
    # This custom scorer also deliberately distinguishes INTEGER and REAL.
    return tuple((type(value).__name__, value) for value in row)


def results_match(
    predicted: Sequence[Sequence[Any]],
    reference: Sequence[Sequence[Any]],
    *,
    ordered: bool = False,
) -> bool:
    pred_rows = [_typed_row(row) for row in predicted]
    gold_rows = [_typed_row(row) for row in reference]
    return pred_rows == gold_rows if ordered else Counter(pred_rows) == Counter(gold_rows)


@dataclass(frozen=True)
class QueryResult:
    sql: str
    rows: list[tuple[Any, ...]] | None
    error: str | None
    error_type: str | None
    elapsed_seconds: float

    @property
    def ok(self) -> bool:
        return self.error is None


_DENIED_ACTIONS = {
    getattr(sqlite3, name)
    for name in (
        "SQLITE_INSERT", "SQLITE_UPDATE", "SQLITE_DELETE", "SQLITE_ALTER_TABLE",
        "SQLITE_CREATE_INDEX", "SQLITE_CREATE_TABLE", "SQLITE_CREATE_TRIGGER",
        "SQLITE_CREATE_VIEW", "SQLITE_CREATE_TEMP_INDEX", "SQLITE_CREATE_TEMP_TABLE",
        "SQLITE_CREATE_TEMP_TRIGGER", "SQLITE_CREATE_TEMP_VIEW", "SQLITE_DROP_INDEX",
        "SQLITE_DROP_TABLE", "SQLITE_DROP_TRIGGER", "SQLITE_DROP_VIEW",
        "SQLITE_DROP_TEMP_INDEX", "SQLITE_DROP_TEMP_TABLE", "SQLITE_DROP_TEMP_TRIGGER",
        "SQLITE_DROP_TEMP_VIEW", "SQLITE_REINDEX", "SQLITE_ANALYZE", "SQLITE_ATTACH",
        "SQLITE_DETACH", "SQLITE_PRAGMA", "SQLITE_TRANSACTION", "SQLITE_SAVEPOINT",
    )
    if hasattr(sqlite3, name)
}


def _readonly_authorizer(
    action: int, arg1: str | None, arg2: str | None,
    database: str | None, trigger: str | None,
) -> int:
    if action in _DENIED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        function_name = (arg2 or arg1 or "").casefold()
        if function_name in {"load_extension", "writefile"}:
            return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def execute_readonly(
    db_path: str | Path,
    sql: str,
    *,
    timeout_seconds: float = 5.0,
    max_rows: int = 100_000,
) -> QueryResult:
    """Execute SELECT/WITH with a wall-clock budget and bounded result size.

    SQLite's connect timeout only bounds lock waits. A progress handler here
    checks a monotonic deadline during query execution, and fetch checks share
    that deadline. Progress callbacks run every 1,000 VM instructions; this is
    a cooperative budget, not an OS-level hard kill of an individual opcode.
    A result exceeding max_rows is an error, never a truncated match.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    start = time.monotonic()
    deadline = start + timeout_seconds
    connection: sqlite3.Connection | None = None
    interrupted_for_timeout = False

    def failed(message: str, error_type: str) -> QueryResult:
        return QueryResult(sql, None, message, error_type, time.monotonic() - start)

    def progress() -> int:
        nonlocal interrupted_for_timeout
        interrupted_for_timeout = time.monotonic() >= deadline
        return int(interrupted_for_timeout)

    tokens = _sql_tokens(sql)
    if not tokens or tokens[0] not in {"select", "with"}:
        return failed("Only a single SELECT or WITH query is allowed.", "unsafe_query")
    path = Path(db_path)
    if not path.is_file():
        return failed("SQLite database file does not exist: " + str(path), "database_missing")
    try:
        connection = sqlite3.connect(
            path.resolve().as_uri() + "?mode=ro",
            uri=True,
            timeout=timeout_seconds,
        )
        connection.execute("PRAGMA query_only = ON")
        connection.set_authorizer(_readonly_authorizer)
        connection.set_progress_handler(progress, 1_000)
        cursor = connection.execute(sql)
        if cursor.description is None:
            return failed("Query did not produce a result set.", "unsafe_query")
        rows: list[tuple[Any, ...]] = []
        while True:
            if time.monotonic() >= deadline:
                return failed("Query exceeded its wall-time budget.", "timeout")
            row = cursor.fetchone()
            if row is None:
                break
            if len(rows) >= max_rows:
                return failed("Query result exceeds max_rows=" + str(max_rows), "result_limit")
            rows.append(tuple(row))
        if time.monotonic() >= deadline:
            return failed("Query exceeded its wall-time budget.", "timeout")
        return QueryResult(sql, rows, None, None, time.monotonic() - start)
    except sqlite3.Error as exc:
        if interrupted_for_timeout or time.monotonic() >= deadline:
            return failed("Query exceeded its wall-time budget.", "timeout")
        return failed(str(exc), "sqlite_error")
    except (OSError, ValueError) as exc:
        return failed(str(exc), "database_error")
    finally:
        if connection is not None:
            connection.close()
