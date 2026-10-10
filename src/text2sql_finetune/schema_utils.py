from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def quote_identifier(name: str) -> str:
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def load_spider_tables(tables_path: str | Path) -> dict[str, dict[str, Any]]:
    tables = json.loads(Path(tables_path).read_text(encoding="utf-8"))
    return {item["db_id"]: item for item in tables}


def spider_schema_to_context(schema: dict[str, Any], use_original_names: bool = True) -> str:
    table_key = "table_names_original" if use_original_names else "table_names"
    column_key = "column_names_original" if use_original_names else "column_names"
    table_names = schema[table_key]
    column_names = schema[column_key]
    column_types = schema.get("column_types", [])
    primary_keys = set(schema.get("primary_keys", []))
    foreign_keys = schema.get("foreign_keys", [])

    table_columns: dict[int, list[tuple[int, str, str]]] = {i: [] for i in range(len(table_names))}
    for col_idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx == -1:
            continue
        col_type = column_types[col_idx] if col_idx < len(column_types) else "text"
        table_columns[table_idx].append((col_idx, col_name, col_type))

    lines: list[str] = []
    for table_idx, table_name in enumerate(table_names):
        column_lines = []
        table_pk_names = [name for index, name, _ in table_columns.get(table_idx, []) if index in primary_keys]
        for col_idx, col_name, col_type in table_columns.get(table_idx, []):
            sql_type = _spider_type_to_sql(col_type)
            suffix = " PRIMARY KEY" if col_idx in primary_keys and len(table_pk_names) == 1 else ""
            column_lines.append(f"  {quote_identifier(col_name)} {sql_type}{suffix}")
        if len(table_pk_names) > 1:
            column_lines.append("  PRIMARY KEY (" + ", ".join(quote_identifier(name) for name in table_pk_names) + ")")
        if not column_lines:
            column_lines.append("  id TEXT")
        lines.append(f"CREATE TABLE {quote_identifier(table_name)} (\n" + ",\n".join(column_lines) + "\n);")

    fk_lines = []
    for source_col, target_col in foreign_keys:
        source = _column_ref(column_names, table_names, source_col)
        target = _column_ref(column_names, table_names, target_col)
        if source and target:
            fk_lines.append(f"-- Foreign key: {source} -> {target}")
    if fk_lines:
        lines.append("\n".join(fk_lines))
    return "\n\n".join(lines)


def sqlite_schema_to_context(db_path: str | Path) -> str:
    db_path = Path(db_path)
    conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        chunks = []
        for (table_name,) in tables:
            create_row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name = ?", (table_name,)
            ).fetchone()
            if create_row and create_row[0]:
                chunks.append(create_row[0].strip().rstrip(";") + ";")
            else:
                columns = conn.execute(f"PRAGMA table_info({quote_identifier(table_name)})").fetchall()
                col_defs = [f"  {quote_identifier(col[1])} {col[2] or 'TEXT'}" for col in columns]
                chunks.append(f"CREATE TABLE {quote_identifier(table_name)} (\n" + ",\n".join(col_defs) + "\n);")
        return "\n\n".join(chunks)
    finally:
        conn.close()


def spider_db_path(spider_dir: str | Path, db_id: str) -> Path:
    spider_dir = Path(spider_dir)
    return spider_dir / "database" / db_id / f"{db_id}.sqlite"


def _spider_type_to_sql(col_type: str) -> str:
    col_type = (col_type or "text").lower()
    if col_type in {"number", "real", "float", "double", "decimal"}:
        return "REAL"
    if col_type in {"time", "date", "datetime", "year"}:
        return "TEXT"
    if col_type in {"boolean", "bool"}:
        return "INTEGER"
    return "TEXT"


def _column_ref(column_names: list[list[Any]], table_names: list[str], col_idx: int) -> str | None:
    if col_idx >= len(column_names):
        return None
    table_idx, col_name = column_names[col_idx]
    if table_idx == -1 or table_idx >= len(table_names):
        return None
    return f"{table_names[table_idx]}.{col_name}"
