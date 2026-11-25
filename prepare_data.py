"""Download the dataset at a pinned revision and write training and evaluation files.

Written to --out:
  sft/              DatasetDict on disk (train, validation) with a `prompt`
                    column of chat messages, the format Halo SFT reads
  validation.jsonl  {"text": ..., "label": 0 or 1} rows for score.py
  test.jsonl        same, the untouched public test split
  stats.json        row counts before and after cleaning
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

from prompts import PROMPTS, SELECTED, user_message

DATASET = "Toygar/turkish-offensive-language-detection"
REVISION = "5aff92f9c824061b0781a5ff1bbf1e8246de5840"


def dedup_key(text: str) -> str:
    return " ".join(text.lower().split())


def drop_seen(rows: list[dict], seen: set[str]) -> list[dict]:
    """Keep rows whose text is not in `seen` yet, and add the kept texts to `seen`."""
    kept = []
    for row in rows:
        key = dedup_key(row["text"])
        if key not in seen:
            seen.add(key)
            kept.append(row)
    return kept


def sft_rows(rows: list[dict]) -> Dataset:
    answers = PROMPTS[SELECTED].answers
    return Dataset.from_list(
        [{"prompt": [user_message(r["text"]), {"role": "assistant", "content": answers[r["label"]]}]} for r in rows]
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"text": r["text"], "label": r["label"]}, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the training and evaluation files.")
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()

    raw = load_dataset(DATASET, revision=REVISION)
    splits = {name: [{"text": r["text"], "label": int(r["label"])} for r in raw[name]] for name in raw}

    # The test split is used as published. Validation rows that repeat a test
    # text, and train rows that repeat a test, validation or earlier train
    # text, are dropped so that no evaluated text is seen in training.
    test = splits["test"]
    seen = {dedup_key(r["text"]) for r in test}
    validation = drop_seen(splits["validation"], seen)
    train = drop_seen(splits["train"], seen)

    args.out.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": sft_rows(train), "validation": sft_rows(validation)}).save_to_disk(str(args.out / "sft"))
    write_jsonl(args.out / "validation.jsonl", validation)
    write_jsonl(args.out / "test.jsonl", test)

    stats: dict = {"dataset": DATASET, "revision": REVISION}
    for name, kept in (("train", train), ("validation", validation), ("test", test)):
        labels = Counter(r["label"] for r in kept)
        stats[name] = {
            "rows_published": len(splits[name]),
            "rows_used": len(kept),
            "not_offensive": labels[0],
            "offensive": labels[1],
        }
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
