"""Fit a TF-IDF plus logistic regression baseline and write its scores in the format of score.py.

Character n-grams with a linear classifier are a cheap and strong reference for
offensive-language detection, so the fine-tuned model is compared against one.
The regularisation strength C is chosen from three values by ROC-AUC on the
validation split. The score is the classifier's log-odds of "offensive".
"""

import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

C_VALUES = (1.0, 4.0, 16.0)


def read_rows(path: Path) -> tuple[list[str], list[int]]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [r["text"] for r in rows], [r["label"] for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the TF-IDF baseline and write its scores.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    splits = {name: read_rows(args.data / f"{name}.jsonl") for name in ("train", "validation", "test")}
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    features = {"train": vectorizer.fit_transform(splits["train"][0])}
    for name in ("validation", "test"):
        features[name] = vectorizer.transform(splits[name][0])

    fitted = {}
    for c in C_VALUES:
        model = LogisticRegression(C=c, max_iter=2000).fit(features["train"], splits["train"][1])
        auc = roc_auc_score(splits["validation"][1], model.decision_function(features["validation"]))
        fitted[c] = (auc, model)
        print(f"C={c}: validation ROC-AUC {auc:.4f}")
    best_c = max(fitted, key=lambda c: fitted[c][0])
    model = fitted[best_c][1]
    print(f"selected C={best_c}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    for name in ("validation", "test"):
        scores = model.decision_function(features[name])
        with (args.results_dir / f"tfidf_{name}.jsonl").open("w", encoding="utf-8") as f:
            for label, score in zip(splits[name][1], scores):
                f.write(json.dumps({"label": label, "score": float(score)}) + "\n")


if __name__ == "__main__":
    main()
