"""Behavioral checks for the artificial demo and the read-only API boundary."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
spec = importlib.util.spec_from_file_location("artificial_demo", PROJECT_ROOT / "examples" / "demo.py")
assert spec is not None and spec.loader is not None
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)

try:
    from fastapi.testclient import TestClient
    from text2sql_finetune.serve import DEMO_QUESTIONS, _execute_query, create_app
except ModuleNotFoundError:
    TestClient = None


class FixedPredictor:
    def __init__(self, initial_sql: str, repaired_sql: str = "SELECT missing FROM employees"):
        self.initial_sql = initial_sql
        self.repaired_sql = repaired_sql
        self.repairs = 0

    def generate(self, question: str, schema_context: str) -> str:
        return self.initial_sql

    def repair(self, question: str, schema_context: str, bad_sql: str, error: str) -> str:
        self.repairs += 1
        return self.repaired_sql


class ArtificialDatabaseTests(unittest.TestCase):
    def test_creation_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = demo.create_demo_database(temporary)
            with closing(sqlite3.connect(path)) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM employees").fetchone()[0], 5)
            with self.assertRaises(FileExistsError):
                demo.create_demo_database(temporary)


@unittest.skipIf(TestClient is None, "Install requirements-demo.txt to run FastAPI checks")
class DemoApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.database = demo.create_demo_database(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def client(self, predictor=None):
        return TestClient(create_app(db_root=self.root, demo_mode=True, predictor=predictor))

    def payload(self, **extra):
        return {"database": "company_demo.sqlite", "question": DEMO_QUESTIONS[0], **extra}

    def test_demo_is_labeled_and_repairs_once(self):
        with self.client() as client:
            self.assertEqual(client.get("/health").json()["mode"], "scripted_demo")
            databases = client.get("/api/databases").json()
            self.assertEqual(databases["databases"][0]["id"], "company_demo.sqlite")
            response = client.post("/api/query", json=self.payload())
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("不是 Qwen", result["notice"])
        self.assertIn("employee_name", result["initial_sql"])
        self.assertIn("no such column", result["initial_error"])
        self.assertTrue(result["repair_attempted"])
        self.assertTrue(result["repaired"])
        self.assertIsNone(result["error"])
        self.assertEqual(len(result["rows"]), 5)
        self.assertEqual(result["rows"][0], ["陈晨", "研发"])

    def test_no_repair_option_preserves_failure(self):
        with self.client() as client:
            result = client.post("/api/query", json=self.payload(repair_on_error=False)).json()
        self.assertFalse(result["repair_attempted"])
        self.assertFalse(result["repaired"])
        self.assertIsNotNone(result["error"])
        self.assertEqual(result["rows"], [])

    def test_failed_repair_is_reported_and_never_repeated(self):
        predictor = FixedPredictor("SELECT missing FROM employees")
        with self.client(predictor) as client:
            result = client.post("/api/query", json=self.payload()).json()
        self.assertEqual(predictor.repairs, 1)
        self.assertTrue(result["repair_attempted"])
        self.assertFalse(result["repaired"])
        self.assertIsNotNone(result["error"])

    def test_path_escape_and_invalid_limits_are_rejected(self):
        with self.client() as client:
            for identifier in ("../private.sqlite", "C:/private.sqlite", "..\\private.sqlite"):
                response = client.post("/api/query", json=self.payload(database=identifier))
                self.assertEqual(response.status_code, 400)
            self.assertEqual(client.post("/api/query", json=self.payload(max_rows=0)).status_code, 422)
            self.assertEqual(client.post("/api/query", json=self.payload(max_rows=201)).status_code, 422)
            self.assertEqual(client.post("/api/query", json=self.payload(question="   ")).status_code, 422)

    def test_write_attempt_is_blocked_without_changing_database(self):
        predictor = FixedPredictor("WITH chosen AS (SELECT 1) DELETE FROM employees")
        with self.client(predictor) as client:
            result = client.post("/api/query", json=self.payload(repair_on_error=False)).json()
        self.assertIsNotNone(result["error"])
        with closing(sqlite3.connect(self.database)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM employees").fetchone()[0], 5)

    def test_output_limit_is_explicit(self):
        predictor = FixedPredictor("SELECT name FROM employees ORDER BY id")
        with self.client(predictor) as client:
            result = client.post("/api/query", json=self.payload(max_rows=2)).json()
        self.assertEqual(result["rows"], [["陈晨"], ["林晓"]])
        self.assertTrue(result["truncated"])

    def test_expensive_recursive_query_times_out(self):
        sql = (
            "WITH RECURSIVE counter(n) AS (SELECT 1 UNION ALL "
            "SELECT n + 1 FROM counter WHERE n < 100000000) SELECT SUM(n) FROM counter"
        )
        with self.assertRaisesRegex(ValueError, "已中断"):
            _execute_query(self.database, sql, 50, timeout_seconds=0.01)

    def test_external_file_functions_and_multiple_statements_are_blocked(self):
        for sql in ("SELECT load_extension('missing')", "SELECT 1; DELETE FROM employees"):
            with self.assertRaises((sqlite3.Error, ValueError)):
                _execute_query(self.database, sql, 50)


if __name__ == "__main__":
    unittest.main()
