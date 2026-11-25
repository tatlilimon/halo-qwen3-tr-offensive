"""Download OffensEval-TR 2020 at a pinned revision and write training and evaluation files.

Written to --out:
  sft/              DatasetDict on disk (train, validation) with a `prompt`
                    column of chat messages, the format Halo SFT reads
  train.jsonl       {"text": ..., "label": 0 or 1} rows, read by baseline_tfidf.py
  validation.jsonl  the same for the development split
  test.jsonl        the same for the official test split, used as published
  stats.json        row counts before and after cleaning
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download

from prompts import PROMPTS, SELECTED, user_message

# The official training set split into train and dev by Stefan Schweter, with
# the official test set unchanged. User mentions are already replaced by @USER.
DATASET = "stefan-it/offenseval2020_tr"
REVISION = "f34679caea00504400fc70421adb853765c2a0f7"
FILES = {"train": "train.txt", "validation": "dev.txt", "test": "test.txt"}
LINE = re.compile(r"__label__(NOT|OFF) (.*)")


def read_split(filename: str) -> list[dict]:
    path = hf_hub_download(DATASET, filename, repo_type="dataset", revision=REVISION)
    rows = []
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        if not line:
            continue
        match = LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"unexpected line in {filename}: {line[:80]!r}")
        rows.append({"text": match.group(2), "label": int(match.group(1) == "OFF")})
    return rows


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
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the training and evaluation files.")
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()

    splits = {name: read_split(filename) for name, filename in FILES.items()}

    # The test split is used as published. Validation rows that repeat a test
    # text, and train rows that repeat a test, validation or earlier train
    # text, are dropped so that no evaluated text is seen in training.
    test = splits["test"]
    seen = {dedup_key(r["text"]) for r in test}
    validation = drop_seen(splits["validation"], seen)
    train = drop_seen(splits["train"], seen)

    args.out.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": sft_rows(train), "validation": sft_rows(validation)}).save_to_disk(str(args.out / "sft"))
    write_jsonl(args.out / "train.jsonl", train)
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
