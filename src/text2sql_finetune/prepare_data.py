"""Prepare Spider SFT data with a reproducible, database-disjoint holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Sequence

from .formatting import build_prompt
from .schema_utils import load_spider_tables, spider_schema_to_context


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sample_id(item: dict[str, Any], source: str, source_index: int) -> str:
    """Content/source based identifier, independent of Python's randomized hash()."""
    payload = {
        "source": source,
        "source_index": source_index,
        "db_id": item["db_id"],
        "question": item["question"],
        "query": item["query"],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def split_dev_by_database(
    rows: Sequence[dict[str, Any]], test_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Assign entire dev databases to validation or test; preserve source row order."""
    if not 0 < test_ratio < 1:
        raise ValueError("dev-test-ratio must be strictly between 0 and 1")
    db_ids = sorted({row["db_id"] for row in rows})
    if len(db_ids) < 2:
        raise ValueError("dev.json needs at least two databases for a disjoint valid/test split")
    random.Random(seed).shuffle(db_ids)
    test_db_count = max(1, min(len(db_ids) - 1, round(len(db_ids) * test_ratio)))
    test_db_ids = set(db_ids[:test_db_count])
    valid = [row for row in rows if row["db_id"] not in test_db_ids]
    test = [row for row in rows if row["db_id"] in test_db_ids]
    return valid, test


def assert_database_disjoint(splits: dict[str, Sequence[dict[str, Any]]]) -> None:
    names = list(splits)
    for index, name in enumerate(names):
        db_ids = {row["db_id"] for row in splits[name]}
        for other in names[index + 1 :]:
            overlap = db_ids & {row["db_id"] for row in splits[other]}
            if overlap:
                raise ValueError(f"Database leakage between {name} and {other}: {sorted(overlap)}")


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required Spider file is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a nonempty JSON list in {path}")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Invalid example at {path.name}:{index}")
        for field in ("db_id", "question", "query"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing/non-string {field} at {path.name}:{index}")
        row["_source"] = path.name
        row["_source_index"] = index
    return rows


def convert_rows(rows: Sequence[dict[str, Any]], tables: dict[str, Any]) -> list[dict[str, Any]]:
    contexts: dict[str, str] = {}
    converted = []
    for row in rows:
        db_id = row["db_id"]
        if db_id not in tables:
            raise ValueError(f"No tables.json schema for database {db_id!r}")
        if db_id not in contexts:
            contexts[db_id] = spider_schema_to_context(tables[db_id])
        context = contexts[db_id]
        sql = row["query"].strip()
        converted.append({
            "id": stable_sample_id(row, row["_source"], row["_source_index"]),
            "db_id": db_id,
            "question": row["question"],
            "context": context,
            "answer": sql,
            "query": sql,
            "difficulty": row.get("difficulty", "unknown"),
            "prompt": build_prompt(row["question"], context),
            "completion": sql,
            "source": row["_source"],
            "source_index": row["_source_index"],
        })
    return converted


def describe_split(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    db_ids = sorted({row["db_id"] for row in rows})
    return {"sample_count": len(rows), "database_count": len(db_ids), "db_ids": db_ids}


def limit_rows(rows: list[dict[str, Any]], maximum: int | None) -> list[dict[str, Any]]:
    if maximum is not None and maximum < 1:
        raise ValueError("Smoke-test sample limits must be positive")
    return rows if maximum is None else rows[:maximum]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spider-dir", type=Path, default=Path("data/raw/spider"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--include-train-others", action=argparse.BooleanOptionalAction, default=True,
                        help="Include train_others.json; disable with --no-include-train-others")
    parser.add_argument("--dev-test-ratio", type=float, default=0.5,
                        help="Fraction of DEV DATABASES assigned to test (not fraction of rows)")
    parser.add_argument("--seed", type=int, default=42)
    for name in ("train", "valid", "test"):
        parser.add_argument(f"--max-{name}", type=int)
    args = parser.parse_args()

    tables_path = args.spider_dir / "tables.json"
    if not tables_path.is_file():
        raise FileNotFoundError(f"Required Spider file is missing: {tables_path}")
    tables = load_spider_tables(tables_path)
    if not tables:
        raise ValueError("tables.json contains no schemas")
    source_paths = [args.spider_dir / "train_spider.json"]
    if args.include_train_others:
        source_paths.append(args.spider_dir / "train_others.json")
    source_paths.append(args.spider_dir / "dev.json")
    source_rows = {path.name: read_rows(path) for path in source_paths}
    train_rows = list(source_rows["train_spider.json"])
    if args.include_train_others:
        train_rows.extend(source_rows["train_others.json"])
    valid_rows, test_rows = split_dev_by_database(source_rows["dev.json"], args.dev_test_ratio, args.seed)
    full_splits = {"train": train_rows, "valid": valid_rows, "test": test_rows}
    assert_database_disjoint(full_splits)
    output_splits = {
        name: convert_rows(limit_rows(rows, getattr(args, f"max_{name}")), tables)
        for name, rows in full_splits.items()
    }
    assert_database_disjoint(output_splits)

    existing = [args.out_dir / name for name in ('train.jsonl', 'valid.jsonl', 'test.jsonl', 'dataset_meta.json')
                if (args.out_dir / name).exists()]
    if existing:
        raise FileExistsError(f"Prepared data already exists: {existing}. Choose a new --out-dir.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_files = {}
    for name, rows in output_splits.items():
        path = args.out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        output_files[name] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    source_files = {
        path.name: {
            "path": str(path.resolve()), "sha256": file_sha256(path),
            **({"sample_count": len(source_rows[path.name])} if path.name in source_rows else {}),
        }
        for path in [tables_path, *source_paths]
    }
    metadata = {
        "format_version": 2,
        "split_protocol": "official_train_plus_database_grouped_official_dev_holdout",
        "test_partition_note": "Internal holdout from official Spider dev; not the official hidden test set",
        "seed": args.seed,
        "dev_test_database_ratio": args.dev_test_ratio,
        "include_train_others": args.include_train_others,
        "schema_count_observed": len(tables),
        "sample_id": "SHA256 of canonical source filename/index and db_id/question/query",
        "source_files": source_files,
        "full_splits_before_limits": {name: describe_split(rows) for name, rows in full_splits.items()},
        "output_splits": {name: describe_split(rows) for name, rows in output_splits.items()},
        "smoke_limits": {name: getattr(args, f"max_{name}") for name in full_splits},
        "output_files": output_files,
    }
    meta_path = args.out_dir / "dataset_meta.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metadata": str(meta_path.resolve()), "output_splits": metadata["output_splits"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
