from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="artifacts/predictions/test_predictions.jsonl")
    parser.add_argument("--out-csv", default="artifacts/predictions/error_analysis_by_difficulty.csv")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.predictions).read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("difficulty", "unknown")].append(row)

    summary = []
    for difficulty, items in sorted(groups.items()):
        total = len(items)
        em = sum(int(x["exact_match"]) for x in items) / total if total else 0.0
        ex = sum(int(x["execution_match"]) for x in items) / total if total else 0.0
        summary.append({"difficulty": difficulty, "total": total, "exact_match": em, "execution_accuracy": ex})

    df = pd.DataFrame(summary)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
