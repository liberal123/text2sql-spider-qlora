"""CPU-only protocol tests; these tests do not load a model or claim GPU results."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from text2sql_finetune.prepare_data import (
    assert_database_disjoint,
    describe_split,
    file_sha256,
    limit_rows,
    main as prepare_main,
    read_rows,
    split_dev_by_database,
    stable_sample_id,
)
from text2sql_finetune.sft_encoding import build_completion_only_sample, encode_sql_completion
from text2sql_finetune.train_qlora import SFTDataset, validate_training_config


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}


class DataProtocolTests(unittest.TestCase):
    def setUp(self):
        # Unequal database sizes make a row-level 50/50 split visibly different.
        self.rows = [
            {"db_id": db_id, "question": f"q{index}", "query": "SELECT 1"}
            for db_id, count in (("db_a", 7), ("db_b", 2), ("db_c", 4), ("db_d", 1))
            for index in range(count)
        ]

    def test_dev_split_groups_whole_databases_and_is_reproducible(self):
        valid, test = split_dev_by_database(self.rows, 0.5, 42)
        self.assertEqual((valid, test), split_dev_by_database(self.rows, 0.5, 42))
        self.assertFalse({row["db_id"] for row in valid} & {row["db_id"] for row in test})
        self.assertEqual(len(valid) + len(test), len(self.rows))
        self.assertEqual(len({row["db_id"] for row in test}), 2)
        for db_id in {row["db_id"] for row in self.rows}:
            destination = valid if any(row["db_id"] == db_id for row in valid) else test
            self.assertEqual(sum(row["db_id"] == db_id for row in destination),
                             sum(row["db_id"] == db_id for row in self.rows))

    def test_database_leakage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Database leakage"):
            assert_database_disjoint({"train": self.rows[:1], "valid": self.rows[:1]})

    def test_stable_sample_id_uses_sha256_and_tracks_content(self):
        identifier = stable_sample_id(self.rows[0], "dev.json", 0)
        self.assertEqual(len(identifier), 64)
        self.assertEqual(identifier, stable_sample_id(dict(self.rows[0]), "dev.json", 0))
        self.assertNotEqual(identifier, stable_sample_id({**self.rows[0], "query": "SELECT 2"}, "dev.json", 0))
        self.assertNotEqual(identifier, stable_sample_id(self.rows[0], "dev.json", 1))

    def test_missing_required_source_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                read_rows(Path(folder) / "dev.json")

    def test_smoke_count_reports_actual_rows_and_databases(self):
        stats = describe_split(limit_rows(self.rows, 3))
        self.assertEqual(stats, {"sample_count": 3, "database_count": 1, "db_ids": ["db_a"]})
        with self.assertRaises(ValueError):
            limit_rows(self.rows, 0)

    def test_data_prepare_writes_actual_counts_hashes_and_split_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "raw"
            source.mkdir()
            (source / "tables.json").write_text("[]", encoding="utf-8")
            train = [{"db_id": "train_db", "question": "q", "query": "SELECT 1"}]
            (source / "train_spider.json").write_text(json.dumps(train), encoding="utf-8")
            (source / "dev.json").write_text(json.dumps(self.rows), encoding="utf-8")
            schemas = {db_id: {} for db_id in {row["db_id"] for row in [*train, *self.rows]}}
            output = root / "processed"
            with patch("text2sql_finetune.prepare_data.load_spider_tables", return_value=schemas), \
                    patch("text2sql_finetune.prepare_data.spider_schema_to_context", return_value="CREATE TABLE t (n INT);"), \
                    patch("sys.argv", ["prepare_data", "--spider-dir", str(source), "--out-dir", str(output),
                                       "--no-include-train-others", "--max-valid", "2", "--max-test", "2"]):
                prepare_main()
            meta = json.loads((output / "dataset_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["schema_count_observed"], 5)
            self.assertEqual(meta["full_splits_before_limits"]["train"]["sample_count"], 1)
            self.assertEqual(meta["output_splits"]["valid"]["sample_count"], 2)
            self.assertEqual(meta["output_splits"]["test"]["sample_count"], 2)
            self.assertEqual(meta["output_files"]["test"]["sha256"], file_sha256(output / "test.jsonl"))
            self.assertFalse(set(meta["output_splits"]["valid"]["db_ids"]) & set(meta["output_splits"]["test"]["db_ids"]))
            self.assertFalse(meta["include_train_others"])


class CompletionOnlyTests(unittest.TestCase):
    def test_prompt_mask_sql_and_eos_labels_are_complete(self):
        result = build_completion_only_sample([11, 12], [31, 32], 99, 5)
        self.assertEqual(result["input_ids"], [11, 12, 31, 32, 99])
        self.assertEqual(result["labels"], [-100, -100, 31, 32, 99])
        self.assertEqual(result["attention_mask"], [1] * 5)

    def test_overlong_prompt_or_sql_is_discarded_without_partial_labels(self):
        self.assertIsNone(build_completion_only_sample([11] * 6, [31], 99, 5))
        self.assertIsNone(build_completion_only_sample([11], [31] * 4, 99, 5))
        self.assertIsNotNone(build_completion_only_sample([11], [31] * 3, 99, 5))

    def test_empty_sql_is_rejected(self):
        with self.assertRaises(ValueError):
            build_completion_only_sample([11], [], 99, 5)
        with self.assertRaises(ValueError):
            encode_sql_completion(FakeTokenizer(), "prompt", " ", 30)

    def test_dataset_records_filtering_and_retains_only_complete_sql(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "train.jsonl"
            rows = [
                {"id": "keep", "db_id": "train_db", "prompt": "p", "completion": "SQL"},
                {"id": "drop", "db_id": "train_db", "prompt": "long prompt", "completion": "SQL"},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            dataset = SFTDataset(path, FakeTokenizer(), 5)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.stats["input_sample_count"], 2)
            self.assertEqual(dataset.stats["filtered_over_length_count"], 1)
            self.assertEqual(dataset.filtered_sample_ids, ["drop"])
            self.assertEqual(dataset[0]["labels"], [-100, ord("S"), ord("Q"), ord("L"), 99])

    def test_all_filtered_examples_fail_before_training(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "train.jsonl"
            path.write_text(json.dumps({"id": "drop", "db_id": "train_db", "prompt": "long prompt", "completion": "SQL"}) + "\n",
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No usable examples"):
                SFTDataset(path, FakeTokenizer(), 2)

    def test_best_checkpoint_steps_must_match(self):
        self.assertEqual(validate_training_config({"eval_steps": 5, "save_steps": 5}), (5, 5))
        with self.assertRaises(ValueError):
            validate_training_config({"eval_steps": 5, "save_steps": 10})


if __name__ == "__main__":
    unittest.main()
