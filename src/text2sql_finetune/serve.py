"""FastAPI SQL demo and model service with a read-only database boundary.

``--demo-mode`` runs a fixed script over an artificial database. It does not
load Qwen, use a trained adapter, or measure model quality.
"""

from __future__ import annotations

import argparse
import math
import re
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


DEMO_NOTICE = "脚本演示：使用人工数据库和固定 SQL，不是 Qwen 推理，也不代表模型评估结果。"
MODEL_NOTICE = "真实模型推理：执行反馈最多修复一次；查询结果不等同于基准评估成绩。"
DEMO_QUESTIONS = (
    "列出所有员工的姓名和部门",
    "各部门的平均工资是多少？",
    "工资最高的员工是谁？",
)


class Predictor(Protocol):
    def generate(self, question: str, schema_context: str) -> str: ...

    def repair(
        self, question: str, schema_context: str, bad_sql: str, error: str
    ) -> str: ...


class DemoPredictor:
    """Fixed scripts for demonstrating the request/execute/repair pipeline."""

    def generate(self, question: str, schema_context: str) -> str:
        if question.strip() == DEMO_QUESTIONS[0]:
            # A deliberately misspelled column exercises the feedback branch.
            return (
                "SELECT e.employee_name, d.name AS department "
                "FROM employees AS e JOIN departments AS d "
                "ON e.department_id = d.id ORDER BY e.id"
            )
        if question.strip() == DEMO_QUESTIONS[1]:
            return (
                "SELECT d.name AS department, ROUND(AVG(e.salary), 2) AS average_salary "
                "FROM employees AS e JOIN departments AS d "
                "ON e.department_id = d.id GROUP BY d.id, d.name ORDER BY d.id"
            )
        if question.strip() == DEMO_QUESTIONS[2]:
            return "SELECT name, salary FROM employees ORDER BY salary DESC, id LIMIT 1"
        raise ValueError("脚本演示只支持页面列出的三个问题；自由提问请使用真实模型模式。")

    def repair(
        self, question: str, schema_context: str, bad_sql: str, error: str
    ) -> str:
        if question.strip() != DEMO_QUESTIONS[0]:
            raise ValueError("该固定脚本没有预设修复 SQL。")
        return (
            "SELECT e.name, d.name AS department "
            "FROM employees AS e JOIN departments AS d "
            "ON e.department_id = d.id ORDER BY e.id"
        )


class ModelPredictor:
    def __init__(self, base_model: str, adapter_path: str | None, load_4bit: bool):
        # Keep torch/PEFT/Transformers completely outside demo mode imports.
        from .generate import generate_sql, load_model, repair_sql

        self.model, self.tokenizer = load_model(base_model, adapter_path, load_4bit)
        self._generate_sql = generate_sql
        self._repair_sql = repair_sql

    def generate(self, question: str, schema_context: str) -> str:
        return self._generate_sql(self.model, self.tokenizer, question, schema_context)

    def repair(
        self, question: str, schema_context: str, bad_sql: str, error: str
    ) -> str:
        return self._repair_sql(
            self.model, self.tokenizer, question, schema_context, bad_sql, error
        )


class QueryRequest(BaseModel):
    database: str = Field(min_length=1, max_length=512)
    question: str = Field(min_length=1, max_length=2000)
    repair_on_error: bool = True
    max_rows: int = Field(default=50, ge=1, le=200)


class QueryResponse(BaseModel):
    question: str
    database: str
    mode: str
    notice: str
    initial_sql: str
    sql: str
    repaired_sql: str | None = None
    repair_attempted: bool = False
    repaired: bool = False
    initial_error: str | None = None
    error: str | None = None
    rows: list[list[Any]] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    truncated: bool = False


def _resolve_database(db_root: Path, database: str) -> Path:
    """Resolve only database IDs enumerated beneath the configured root."""
    if "\x00" in database:
        raise HTTPException(status_code=400, detail="无效的数据库 ID。")
    relative = Path(database)
    if (
        relative.is_absolute()
        or "\\" in database
        or ":" in database
        or any(part in {"..", "."} for part in database.split("/"))
    ):
        raise HTTPException(status_code=400, detail="请选择数据库列表中的 database ID。")
    try:
        candidate = (db_root / relative).resolve()
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="无效的数据库 ID。") from exc
    try:
        candidate.relative_to(db_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="数据库超出 db-root 范围。") from exc
    if not candidate.is_file() or candidate.suffix.lower() not in {".sqlite", ".sqlite3", ".db"}:
        raise HTTPException(status_code=404, detail="数据库不存在或扩展名不受支持。")
    return candidate


def _readonly_connection(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=2.0)
    connection.execute("PRAGMA query_only = ON")
    return connection


def _schema_context(database: Path) -> str:
    connection = _readonly_connection(database)
    try:
        tables = connection.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        parts = [statement for _, statement in tables if statement]
        for table, _ in tables:
            quoted = table.replace('"', '""')
            for foreign_key in connection.execute(f'PRAGMA foreign_key_list("{quoted}")'):
                parts.append(
                    f"-- Foreign key: {table}.{foreign_key[3]} -> "
                    f"{foreign_key[2]}.{foreign_key[4]}"
                )
        return "\n\n".join(parts)
    finally:
        connection.close()


def _sql_authorizer(action: int, arg1: str | None, arg2: str | None, *_: Any) -> int:
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
    allowed.add(getattr(sqlite3, "SQLITE_RECURSIVE", 33))
    if action not in allowed:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        function = (arg2 or arg1 or "").lower()
        if function in {"load_extension", "readfile", "writefile", "eval"}:
            return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _json_cell(value: Any) -> Any:
    if isinstance(value, bytes):
        return "0x" + value.hex()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _execute_query(
    database: Path, sql: str, max_rows: int, timeout_seconds: float = 5.0
) -> tuple[list[str], list[list[Any]], bool]:
    stripped = re.sub(r"/\*.*?\*/|--[^\n]*", " ", sql, flags=re.S).lstrip()
    if not re.match(r"(?:SELECT|WITH)\b", stripped, flags=re.I):
        raise ValueError("服务只允许只读 SELECT/WITH 查询。")
    connection = _readonly_connection(database)
    deadline = time.monotonic() + timeout_seconds
    connection.set_authorizer(_sql_authorizer)
    connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
    try:
        cursor = connection.execute(sql)
        columns = [item[0] for item in (cursor.description or [])]
        fetched = cursor.fetchmany(max_rows + 1)
        rows = [[_json_cell(value) for value in row] for row in fetched[:max_rows]]
        return columns, rows, len(fetched) > max_rows
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            raise ValueError(f"SQL 查询超过 {timeout_seconds:g} 秒，已中断。") from exc
        raise
    finally:
        connection.close()


def create_app(
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    adapter_path: str | None = None,
    load_4bit: bool = True,
    *,
    db_root: str | Path = "examples/demo_databases",
    demo_mode: bool = False,
    predictor: Predictor | None = None,
) -> FastAPI:
    root = Path(db_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"db-root 目录不存在：{root}")
    mode = "scripted_demo" if demo_mode else "model"
    notice = DEMO_NOTICE if demo_mode else MODEL_NOTICE
    model_lock = threading.Lock()
    chosen_predictor = predictor or (DemoPredictor() if demo_mode else None)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        nonlocal chosen_predictor
        if chosen_predictor is None:
            chosen_predictor = ModelPredictor(base_model, adapter_path, load_4bit)
        yield

    application = FastAPI(title="Text-to-SQL 执行反馈演示", lifespan=lifespan)

    @application.get("/", include_in_schema=False)
    def index():
        html = Path(__file__).resolve().parents[2] / "web" / "index.html"
        if not html.is_file():
            raise HTTPException(status_code=404, detail="展示页面缺失。请使用源码目录启动服务。")
        return FileResponse(html)

    @application.get("/health")
    def health():
        return {"status": "ok", "mode": mode, "notice": notice, "ready": chosen_predictor is not None}

    @application.get("/api/databases")
    def databases():
        items = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".sqlite", ".sqlite3", ".db"}:
                continue
            try:
                path.resolve().relative_to(root)
            except ValueError:
                continue
            identifier = path.relative_to(root).as_posix()
            items.append({"id": identifier, "name": identifier})
        return {
            "mode": mode,
            "notice": notice,
            "databases": items,
            "demo_questions": list(DEMO_QUESTIONS) if demo_mode else [],
        }

    @application.post("/api/query", response_model=QueryResponse)
    def query(request: QueryRequest):
        if not request.question.strip():
            raise HTTPException(status_code=422, detail="问题不能为空白。")
        database = _resolve_database(root, request.database)
        try:
            schema = _schema_context(database)
        except sqlite3.Error as exc:
            raise HTTPException(status_code=400, detail=f"无法读取数据库 schema：{exc}") from exc
        if chosen_predictor is None:
            raise HTTPException(status_code=503, detail="模型尚未加载完成。")
        try:
            with model_lock:
                initial_sql = chosen_predictor.generate(request.question, schema)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"SQL 生成失败：{type(exc).__name__}") from exc
        response = QueryResponse(
            question=request.question, database=request.database, mode=mode,
            notice=notice, initial_sql=initial_sql, sql=initial_sql,
        )
        try:
            response.columns, response.rows, response.truncated = _execute_query(
                database, initial_sql, request.max_rows
            )
            return response
        except (sqlite3.Error, ValueError) as exc:
            response.initial_error = str(exc)
            response.error = str(exc)
        if not request.repair_on_error:
            return response
        response.repair_attempted = True
        try:
            with model_lock:
                repaired_sql = chosen_predictor.repair(
                    request.question, schema, initial_sql, response.initial_error or ""
                )
            response.repaired_sql = repaired_sql
            response.sql = repaired_sql
            response.columns, response.rows, response.truncated = _execute_query(
                database, repaired_sql, request.max_rows
            )
            response.repaired = True
            response.error = None
        except Exception as exc:
            # A failed second execution must remain visibly unsuccessful.
            response.error = str(exc)
        return response

    return application


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--db-root", required=True, help="仅允许读取此目录内的 SQLite 文件")
    parser.add_argument("--demo-mode", action="store_true", help="人工数据库固定 SQL 演示，不加载模型")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()
    application = create_app(
        args.base_model, args.adapter_path, not args.no_4bit,
        db_root=args.db_root, demo_mode=args.demo_mode,
    )
    import uvicorn

    uvicorn.run(application, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
