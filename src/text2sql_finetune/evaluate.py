"""Custom SQLite evaluation, with optional model inference and error repair.

The --predictions-file route uses only Python's standard library. This module
does not import torch/Transformers/PEFT unless online generation is requested.
These metrics are intentionally NOT the official Spider evaluation metrics.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping

from .sql_execution import (
    QueryResult, execute_readonly, has_top_level_order_by,
    normalize_sql_text, results_match,
)


RepairFunction = Callable[[str, str, int], str | None]


def _text_match(predicted: str, gold: str) -> bool:
    normalized_pred = normalize_sql_text(predicted)
    return bool(normalized_pred) and normalized_pred == normalize_sql_text(gold)


def _execution_match(predicted: QueryResult, gold: QueryResult, *, ordered: bool) -> bool | None:
    if not gold.ok:
        return None
    if not predicted.ok:
        return False
    assert predicted.rows is not None and gold.rows is not None
    return results_match(predicted.rows, gold.rows, ordered=ordered)


def evaluate_example(
    item: Mapping[str, Any],
    initial_sql: str,
    db_path: str | Path,
    *,
    repair_fn: RepairFunction | None = None,
    max_repair_attempts: int = 1,
    timeout_seconds: float = 5.0,
    max_rows: int = 100_000,
) -> dict[str, Any]:
    """Evaluate one example. A failed reference query never reaches repair_fn.

    Repair receives only the prediction and its SQLite error. Successful but
    semantically wrong SQL is not repaired using the reference or its result.
    """
    gold_sql = item.get("answer", item.get("gold_sql"))
    if not isinstance(gold_sql, str):
        raise ValueError("Each test record needs a string answer or gold_sql")
    if not isinstance(initial_sql, str):
        raise ValueError("Prediction SQL must be a string")
    if max_repair_attempts < 0:
        raise ValueError("max_repair_attempts must be nonnegative")
    execution_options = {"timeout_seconds": timeout_seconds, "max_rows": max_rows}
    gold = execute_readonly(db_path, gold_sql, **execution_options)
    initial = execute_readonly(db_path, initial_sql, **execution_options)
    final = initial
    ordered = has_top_level_order_by(gold_sql)
    attempts: list[dict[str, Any]] = []
    if gold.ok and repair_fn is not None:
        for attempt in range(1, max_repair_attempts + 1):
            if final.ok:
                break
            assert final.error is not None
            repaired_sql = repair_fn(final.sql, final.error, attempt)
            if repaired_sql is None:
                break
            if not isinstance(repaired_sql, str):
                raise ValueError("repair_fn must return SQL text or None")
            final = execute_readonly(db_path, repaired_sql, **execution_options)
            attempts.append({
                "attempt": attempt,
                "pred_sql": repaired_sql,
                "executable": final.ok,
                "execution_error": final.error,
                "execution_error_type": final.error_type,
                "elapsed_seconds": final.elapsed_seconds,
            })
    initial_match = _execution_match(initial, gold, ordered=ordered)
    final_match = _execution_match(final, gold, ordered=ordered)
    fixed = bool(attempts) and not initial.ok and final.ok
    return {
        "id": item.get("id"),
        "db_id": item.get("db_id"),
        "difficulty": item.get("difficulty"),
        "question": item.get("question"),
        "gold_sql": gold_sql,
        "initial_pred_sql": initial_sql,
        "pred_sql": final.sql,
        "initial_exact_match": _text_match(initial_sql, gold_sql),
        "exact_match": _text_match(final.sql, gold_sql),
        "gold_executable": gold.ok,
        "gold_execution_error": gold.error,
        "gold_execution_error_type": gold.error_type,
        "execution_evaluable": gold.ok,
        "order_sensitive": ordered,
        "initial_pred_executable": initial.ok,
        "pred_executable": final.ok,
        "initial_execution_match": initial_match,
        "execution_match": final_match,
        "execution_error": final.error,
        "execution_error_type": final.error_type,
        "first_execution_error": initial.error,
        "first_execution_error_type": initial.error_type,
        "repair_attempts": len(attempts),
        "repair_history": attempts,
        "repair_fixed": fixed,
        "repair_correct": fixed and final_match is True,
        "repair_executable_but_incorrect": fixed and final_match is False,
        "gold_elapsed_seconds": gold.elapsed_seconds,
        "initial_elapsed_seconds": initial.elapsed_seconds,
        "final_elapsed_seconds": final.elapsed_seconds,
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    total = len(rows)
    valid = sum(bool(row["execution_evaluable"]) for row in rows)
    initial_em = sum(bool(row["initial_exact_match"]) for row in rows)
    final_em = sum(bool(row["exact_match"]) for row in rows)
    initial_ex = sum(row["initial_execution_match"] is True for row in rows)
    final_ex = sum(row["execution_match"] is True for row in rows)
    initial_executable = sum(bool(row["initial_pred_executable"]) for row in rows)
    final_executable = sum(bool(row["pred_executable"]) for row in rows)
    return {
        "evaluator": "custom_sqlite_v1",
        "official_spider_evaluation": False,
        "total": total,
        "initial_exact_match_count": initial_em,
        "exact_match_count": final_em,
        "exact_match_total": total,
        "initial_exact_match": _ratio(initial_em, total),
        "exact_match": _ratio(final_em, total),
        "initial_execution_match_count": initial_ex,
        "execution_match_count": final_ex,
        "execution_total": total,
        "execution_evaluable_total": valid,
        "initial_execution_accuracy": _ratio(initial_ex, total),
        "execution_accuracy": _ratio(final_ex, total),
        "eligible_execution_total": valid,
        "initial_eligible_execution_accuracy": _ratio(initial_ex, valid),
        "eligible_execution_accuracy": _ratio(final_ex, valid),
        "gold_execution_error_count": total - valid,
        "initial_pred_executable_count": initial_executable,
        "pred_executable_count": final_executable,
        "pred_executable_total": total,
        "initial_pred_execution_rate": _ratio(initial_executable, total),
        "pred_execution_rate": _ratio(final_executable, total),
        "initial_execution_error_count": total - initial_executable,
        "execution_error_count": total - final_executable,
        "repair_samples": sum(row["repair_attempts"] > 0 for row in rows),
        "repair_attempts": sum(row["repair_attempts"] for row in rows),
        "repair_fixed": sum(bool(row["repair_fixed"]) for row in rows),
        "repair_correct": sum(bool(row["repair_correct"]) for row in rows),
        "repair_executable_but_incorrect": sum(
            bool(row["repair_executable_but_incorrect"]) for row in rows
        ),
        "metric_definitions": {
            "exact_match": "Conservative normalized SQL text equality; denominator is all selected rows.",
            "execution_accuracy": "Complete typed result equality; denominator is all selected rows, including reference/database failures.",
            "eligible_execution_accuracy": "Supplementary result equality rate excluding reference execution failures; eligible_execution_total is its denominator.",
            "ordering": "Reference top-level ORDER BY requires ordered equality; otherwise a row multiset is compared.",
            "result_values": "NULL, strings, INTEGER/REAL type distinctions and duplicate rows are preserved.",
            "repair_fixed": "Initially invalid SQL becomes executable; this alone does not mean it is correct.",
            "rates": "All rates are fractions in [0, 1]; zero denominator is reported as null.",
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}, line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object in {path}, line {line_number}")
            rows.append(row)
    return rows


def _index_predictions(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if row.get("id") is None:
            raise ValueError("Every prediction record must have an id matching the test record")
        key = str(row["id"])
        if key in indexed:
            raise ValueError("Duplicate prediction id: " + key)
        sql = row.get("pred_sql", row.get("sql"))
        if not isinstance(sql, str):
            raise ValueError("Prediction record requires a string pred_sql or sql: " + key)
        repairs = row.get("repair_sqls", [])
        if not isinstance(repairs, list) or not all(isinstance(value, str) for value in repairs):
            raise ValueError("repair_sqls must be an array of SQL strings: " + key)
        indexed[key] = row
    return indexed


def _database_path(spider_dir: Path, db_id: str) -> Path:
    # Reject traversal before joining ids from an external JSONL file.
    if not db_id or db_id in {".", ".."} or "/" in db_id or "\\" in db_id or ":" in db_id:
        raise ValueError("Invalid db_id: " + repr(db_id))
    return spider_dir / "database" / db_id / (db_id + ".sqlite")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--adapter-path", default=None, help="Omit to evaluate the unmodified base model")
    parser.add_argument("--test-file", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument("--predictions-file", type=Path, help="Offline JSONL containing id and pred_sql; no model dependencies")
    parser.add_argument("--spider-dir", type=Path, default=Path("data/raw/spider"))
    parser.add_argument("--out-file", type=Path, default=Path("artifacts/predictions/test_predictions.jsonl"))
    parser.add_argument("--metrics-file", type=Path, default=Path("artifacts/predictions/test_metrics.json"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--repair-on-error", action="store_true", help="Only repair prediction execution errors with a valid reference; offline mode consumes repair_sqls")
    parser.add_argument("--max-repair-attempts", type=int, default=1)
    parser.add_argument("--query-timeout", type=float, default=5.0, help="Per-query wall-time budget in seconds")
    parser.add_argument("--max-rows", type=int, default=100_000, help="Reject larger results instead of scoring a truncated result")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly allow replacing existing output files")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.max_repair_attempts < 0 or args.query_timeout <= 0 or args.max_rows <= 0:
        parser.error("Repair attempts must be nonnegative; timeout and max rows must be positive")
    input_paths = {args.test_file.resolve()}
    if args.predictions_file is not None:
        input_paths.add(args.predictions_file.resolve())
    if args.out_file.resolve() == args.metrics_file.resolve():
        parser.error("Prediction and metrics output paths must be different")
    for output in (args.out_file, args.metrics_file):
        if output.resolve() in input_paths:
            parser.error("An output path cannot replace an input file")
        if output.exists() and not args.overwrite:
            parser.error(f"Output exists: {output}. Choose another path or pass --overwrite")

    items = _read_jsonl(args.test_file)
    seen: set[str] = set()
    for index, item in enumerate(items):
        if item.get("id") is None:
            item["id"] = str(index)
        key = str(item["id"])
        if key in seen:
            raise ValueError("Duplicate test id: " + key)
        seen.add(key)
        if not isinstance(item.get("db_id"), str):
            raise ValueError("Each test row needs a string db_id")
        _database_path(args.spider_dir, item["db_id"])
    if args.limit is not None:
        items = items[:args.limit]
    if not items:
        parser.error("The selected test set is empty")

    predictions = None
    model = tokenizer = None
    if args.predictions_file is not None:
        predictions = _index_predictions(_read_jsonl(args.predictions_file))
        for item in items:
            key = str(item["id"])
            if key not in predictions:
                raise ValueError("Missing prediction for test id: " + key)
            predicted_db = predictions[key].get("db_id")
            if predicted_db is not None and predicted_db != item["db_id"]:
                raise ValueError("Prediction db_id differs from test db_id: " + key)
    else:
        from .generate import generate_sql, load_model, repair_sql
        model, tokenizer = load_model(
            args.base_model, args.adapter_path, load_4bit=not args.no_4bit
        )

    records: list[dict[str, Any]] = []
    for index, item in enumerate(items, 1):
        repair_fn: RepairFunction | None = None
        if predictions is not None:
            prediction = predictions[str(item["id"])]
            initial_sql = prediction.get("pred_sql", prediction.get("sql"))
            if args.repair_on_error:
                candidates = prediction.get("repair_sqls", [])
                repair_fn = lambda bad, error, attempt, candidates=candidates: (
                    candidates[attempt - 1] if attempt <= len(candidates) else None
                )
        else:
            initial_sql = generate_sql(model, tokenizer, item["question"], item["context"])
            if args.repair_on_error:
                repair_fn = lambda bad, error, attempt, item=item: repair_sql(
                    model, tokenizer, item["question"], item["context"], bad, error
                )
        records.append(evaluate_example(
            item, initial_sql,
            _database_path(args.spider_dir, item["db_id"]),
            repair_fn=repair_fn,
            max_repair_attempts=args.max_repair_attempts,
            timeout_seconds=args.query_timeout,
            max_rows=args.max_rows,
        ))
        if index % 25 == 0 or index == len(items):
            print(f"Evaluated {index}/{len(items)}", file=sys.stderr)

    metrics = summarize(records)
    metrics["run"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_file": str(args.test_file),
        "test_file_sha256": hashlib.sha256(args.test_file.read_bytes()).hexdigest(),
        "predictions_file": str(args.predictions_file) if args.predictions_file else None,
        "predictions_file_sha256": hashlib.sha256(args.predictions_file.read_bytes()).hexdigest() if args.predictions_file else None,
        "source": "pre_generated_sql" if predictions is not None else "model_generation",
        "base_model": args.base_model if predictions is None else None,
        "adapter_path": args.adapter_path if predictions is None else None,
        "load_4bit": not args.no_4bit if predictions is None else None,
        "repair_on_error": args.repair_on_error,
        "repair_source": (
            "pre_generated_candidates" if predictions is not None else "model_generation"
        ) if args.repair_on_error else None,
        "max_repair_attempts": args.max_repair_attempts,
        "query_timeout_seconds": args.query_timeout,
        "max_rows": args.max_rows,
        "limit": args.limit,
    }
    for output in (args.out_file, args.metrics_file):
        output.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open("w" if args.overwrite else "x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with args.metrics_file.open("w" if args.overwrite else "x", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
