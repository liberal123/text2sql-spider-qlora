"""Create an artificial SQLite database for the scripted pipeline demo.

This program uses only Python's standard library. The rows are hand-written
examples, not Spider data or model evaluation outputs.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


QUESTIONS = (
    "列出所有员工的姓名和部门",
    "各部门的平均工资是多少？",
    "工资最高的员工是谁？",
)


def create_demo_database(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    database = root / "company_demo.sqlite"
    if database.exists():
        raise FileExistsError(f"演示数据库已存在，请复用或选择新的输出目录：{database}")
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE departments (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            );
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department_id INTEGER NOT NULL REFERENCES departments(id),
                salary REAL NOT NULL
            );
            """
        )
        connection.executemany(
            "INSERT INTO departments (id, name) VALUES (?, ?)",
            [(1, "研发"), (2, "产品"), (3, "运营")],
        )
        connection.executemany(
            "INSERT INTO employees (id, name, department_id, salary) VALUES (?, ?, ?, ?)",
            [
                (1, "陈晨", 1, 18000),
                (2, "林晓", 1, 22000),
                (3, "王宁", 2, 16000),
                (4, "赵然", 3, 14000),
                (5, "周敏", 2, 20000),
            ],
        )
        connection.commit()
    finally:
        connection.close()
    return database.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="examples/demo_databases")
    args = parser.parse_args()
    try:
        database = create_demo_database(args.output_dir)
    except FileExistsError as exc:
        parser.exit(0, str(exc) + "\n")
    print(f"人工演示数据库：{database}")
    print("固定脚本只支持以下问题：")
    for question in QUESTIONS:
        print(f"  - {question}")
    print("第一个问题故意生成错误字段，再用预设 SQL 修复，以展示执行反馈管道。")
    print("该演示不是 Qwen 推理，也不产生模型基准评估指标。")


if __name__ == "__main__":
    main()
