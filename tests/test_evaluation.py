"""Offline fixture checks: these are engineering tests, not Spider results."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql_finetune.evaluate import (  # noqa: E402
    _index_predictions, build_parser, evaluate_example, main, summarize,
)
from text2sql_finetune.sql_execution import (  # noqa: E402
    execute_readonly, has_top_level_order_by, normalize_sql_text, results_match,
)


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.spider = self.root / "spider"
        self.db = self.spider / "database" / "demo" / "demo.sqlite"
        self.db.parent.mkdir(parents=True)
        connection = sqlite3.connect(self.db)
        try:
            connection.execute("CREATE TABLE numbers (value INTEGER)")
            connection.executemany("INSERT INTO numbers VALUES (?)", [(1,), (2,), (2,)])
            connection.execute("CREATE TABLE words (value TEXT)")
            connection.executemany("INSERT INTO words VALUES (?)", [(None,), ("Alpha",), ("Alpha",), ("",)])
            connection.commit()
        finally:
            connection.close()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def item(self, answer: str) -> dict[str, str]:
        return {"id": "one", "db_id": "demo", "question": "fixture question", "context": "fixture schema", "answer": answer}

    def test_readonly_select_and_cte(self) -> None:
        result = execute_readonly(self.db, "WITH x AS (SELECT value FROM numbers) SELECT value FROM x")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.rows, [(1,), (2,), (2,)])

    def test_writes_and_multiple_statements_cannot_modify_database(self) -> None:
        for sql in (
            "DELETE FROM numbers", "DROP TABLE numbers", "ATTACH DATABASE ':memory:' AS other",
            "WITH x AS (SELECT 1) DELETE FROM numbers", "SELECT 1; DELETE FROM numbers",
            "PRAGMA writable_schema=ON",
        ):
            with self.subTest(sql=sql):
                self.assertFalse(execute_readonly(self.db, sql).ok)
        self.assertEqual(execute_readonly(self.db, "SELECT count(*) FROM numbers").rows, [(3,)])

    def test_missing_database_is_not_created(self) -> None:
        path = self.root / "missing.sqlite"
        result = execute_readonly(path, "SELECT 1")
        self.assertEqual(result.error_type, "database_missing")
        self.assertFalse(path.exists())

    def test_query_budget_interrupts_computation(self) -> None:
        sql = (
            "WITH RECURSIVE counter(x) AS (SELECT 0 UNION ALL "
            "SELECT x+1 FROM counter WHERE x<100000000) SELECT sum(x) FROM counter"
        )
        result = execute_readonly(self.db, sql, timeout_seconds=0.02)
        self.assertEqual(result.error_type, "timeout")
        self.assertLess(result.elapsed_seconds, 2.0)

    def test_row_limit_is_error_instead_of_partial_success(self) -> None:
        result = execute_readonly(self.db, "SELECT value FROM numbers", max_rows=2)
        self.assertEqual(result.error_type, "result_limit")
        self.assertIsNone(result.rows)

    def test_invalid_resource_limits_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            execute_readonly(self.db, "SELECT 1", timeout_seconds=0)
        with self.assertRaises(ValueError):
            execute_readonly(self.db, "SELECT 1", max_rows=0)

    def test_normalization_preserves_quoted_values_and_comment_characters(self) -> None:
        self.assertEqual(normalize_sql_text("SELECT a FROM t; -- comment"), normalize_sql_text("select a from t"))
        self.assertNotEqual(normalize_sql_text("SELECT 'Alpha'"), normalize_sql_text("select 'alpha'"))
        self.assertNotEqual(normalize_sql_text("SELECT 'a--b'"), normalize_sql_text("SELECT 'a'"))
        self.assertIn("'O''Reilly'", normalize_sql_text("SELECT 'O''Reilly'"))

    def test_order_detection_ignores_strings_comments_and_inner_query(self) -> None:
        self.assertFalse(has_top_level_order_by("SELECT 'ORDER BY' FROM (SELECT value FROM numbers ORDER BY value) -- ORDER BY"))
        self.assertTrue(has_top_level_order_by("WITH x AS (SELECT value FROM numbers ORDER BY value) SELECT * FROM x ORDER BY value"))

    def test_result_comparison_preserves_null_strings_types_and_duplicates(self) -> None:
        self.assertFalse(results_match([(None,)], [("",)]))
        self.assertFalse(results_match([(1,)], [("1",)]))
        self.assertFalse(results_match([(1,)], [(1.0,)]))
        self.assertFalse(results_match([(1,), (1,)], [(1,)]))
        self.assertTrue(results_match([(2,), (1,), (2,)], [(1,), (2,), (2,)]))
        self.assertFalse(results_match([(2,), (1,)], [(1,), (2,)], ordered=True))

    def test_executed_null_and_duplicate_changes_are_not_correct(self) -> None:
        null_changed = evaluate_example(self.item("SELECT value FROM words"), "SELECT coalesce(value, '') FROM words", self.db)
        duplicate_changed = evaluate_example(self.item("SELECT value FROM numbers"), "SELECT DISTINCT value FROM numbers", self.db)
        self.assertFalse(null_changed["execution_match"])
        self.assertFalse(duplicate_changed["execution_match"])

    def test_only_reference_outer_order_by_controls_comparison(self) -> None:
        gold = "SELECT value FROM numbers ORDER BY value DESC"
        predicted = "SELECT value FROM numbers ORDER BY value ASC"
        record = evaluate_example(self.item(gold), predicted, self.db)
        self.assertFalse(record["execution_match"])
        self.assertTrue(record["order_sensitive"])
        record = evaluate_example(self.item("SELECT value FROM numbers"), predicted, self.db)
        self.assertTrue(record["execution_match"])

    def test_prediction_error_can_be_repaired_and_initial_result_survives(self) -> None:
        calls = []

        def repair(bad: str, error: str, attempt: int) -> str:
            calls.append((bad, error, attempt))
            return "SELECT value FROM numbers"

        record = evaluate_example(self.item("SELECT value FROM numbers"), "SELECT missing FROM numbers", self.db, repair_fn=repair)
        self.assertEqual(len(calls), 1)
        self.assertIn("missing", calls[0][1])
        self.assertFalse(record["initial_execution_match"])
        self.assertTrue(record["execution_match"])
        self.assertTrue(record["repair_fixed"])
        self.assertTrue(record["repair_correct"])
        self.assertEqual(record["repair_attempts"], 1)
        self.assertIsNotNone(record["first_execution_error"])

    def test_executable_but_wrong_prediction_does_not_receive_reference_feedback(self) -> None:
        calls = []

        def repair(bad: str, error: str, attempt: int) -> str:
            calls.append(error)
            return "SELECT value FROM numbers"

        record = evaluate_example(self.item("SELECT value FROM numbers"), "SELECT 99", self.db, repair_fn=repair)
        self.assertFalse(record["execution_match"])
        self.assertEqual(calls, [])
        self.assertEqual(record["repair_attempts"], 0)

    def test_reference_failure_is_excluded_and_never_sent_to_repair(self) -> None:
        calls = []

        def repair(bad: str, error: str, attempt: int) -> str:
            calls.append(error)
            return "SELECT 1"

        record = evaluate_example(self.item("SELECT secret_reference_field FROM numbers"), "SELECT missing FROM numbers", self.db, repair_fn=repair)
        self.assertFalse(record["execution_evaluable"])
        self.assertIsNone(record["execution_match"])
        self.assertIn("secret_reference_field", record["gold_execution_error"])
        self.assertEqual(calls, [])

    def test_executable_repair_is_reported_separately_from_correct_repair(self) -> None:
        record = evaluate_example(self.item("SELECT value FROM numbers"), "SELECT missing FROM numbers", self.db, repair_fn=lambda bad, error, attempt: "SELECT 99")
        metrics = summarize([record])
        self.assertEqual(metrics["repair_fixed"], 1)
        self.assertEqual(metrics["repair_correct"], 0)
        self.assertEqual(metrics["repair_executable_but_incorrect"], 1)
        self.assertEqual(metrics["execution_accuracy"], 0.0)

    def test_zero_repair_budget_never_invokes_callback(self) -> None:
        def forbidden(bad: str, error: str, attempt: int) -> str:
            self.fail("Repair callback should not be called")

        record = evaluate_example(self.item("SELECT value FROM numbers"), "SELECT missing FROM numbers", self.db, repair_fn=forbidden, max_repair_attempts=0)
        self.assertEqual(record["repair_attempts"], 0)

    def test_denominators_and_zero_denominator_are_explicit(self) -> None:
        correct = evaluate_example(self.item("SELECT value FROM numbers"), "SELECT value FROM numbers", self.db)
        invalid_reference = evaluate_example(self.item("SELECT missing FROM numbers"), "SELECT value FROM numbers", self.db)
        metrics = summarize([correct, invalid_reference])
        self.assertEqual(metrics["exact_match_count"], 1)
        self.assertEqual(metrics["exact_match_total"], 2)
        self.assertEqual(metrics["execution_match_count"], 1)
        self.assertEqual(metrics["execution_total"], 2)
        self.assertEqual(metrics["execution_evaluable_total"], 1)
        self.assertEqual(metrics["gold_execution_error_count"], 1)
        self.assertEqual(metrics["execution_accuracy"], 0.5)
        self.assertEqual(metrics["eligible_execution_accuracy"], 1.0)
        self.assertEqual(summarize([invalid_reference])["execution_accuracy"], 0.0)
        self.assertIsNone(summarize([invalid_reference])["eligible_execution_accuracy"])
        self.assertIsNone(summarize([])["execution_accuracy"])

    def test_prediction_ids_are_explicit_and_unique(self) -> None:
        with self.assertRaises(ValueError):
            _index_predictions([{"pred_sql": "SELECT 1"}])
        with self.assertRaises(ValueError):
            _index_predictions([{"id": 1, "pred_sql": "SELECT 1"}, {"id": "1", "pred_sql": "SELECT 2"}])

    def test_adapter_is_optional_for_base_model_baseline(self) -> None:
        self.assertIsNone(build_parser().parse_args([]).adapter_path)

    def test_offline_cli_aligns_ids_and_writes_auditable_results_without_model(self) -> None:
        items = [
            {**self.item("SELECT value FROM numbers"), "id": "a"},
            {**self.item("SELECT value FROM numbers ORDER BY value"), "id": "b"},
            {**self.item("SELECT gold_missing FROM numbers"), "id": "c"},
        ]
        predictions = [
            {"id": "c", "pred_sql": "SELECT value FROM numbers"},
            {"id": "b", "pred_sql": "SELECT missing FROM numbers", "repair_sqls": ["SELECT value FROM numbers ORDER BY value"]},
            {"id": "a", "pred_sql": "SELECT value FROM numbers"},
        ]
        test_file, prediction_file = self.root / "test.jsonl", self.root / "input_predictions.jsonl"
        for path, records in ((test_file, items), (prediction_file, predictions)):
            path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        out, metrics_file = self.root / "out.jsonl", self.root / "metrics.json"
        args = [
            "--test-file", str(test_file), "--predictions-file", str(prediction_file),
            "--spider-dir", str(self.spider), "--out-file", str(out),
            "--metrics-file", str(metrics_file), "--repair-on-error",
        ]
        # Any accidental import of the generation module fails in this route.
        with patch.dict(sys.modules, {"text2sql_finetune.generate": None}):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                main(args)
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        output = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([record["id"] for record in output], ["a", "b", "c"])
        self.assertEqual(metrics["initial_exact_match_count"], 1)
        self.assertEqual(metrics["exact_match_count"], 2)
        self.assertEqual(metrics["execution_match_count"], 2)
        self.assertEqual(metrics["execution_evaluable_total"], 2)
        self.assertEqual(metrics["repair_correct"], 1)
        self.assertFalse(metrics["official_spider_evaluation"])
        self.assertEqual(metrics["run"]["source"], "pre_generated_sql")
        self.assertEqual(len(metrics["run"]["test_file_sha256"]), 64)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(args)  # Existing outputs require explicit --overwrite.


if __name__ == "__main__":
    unittest.main()
