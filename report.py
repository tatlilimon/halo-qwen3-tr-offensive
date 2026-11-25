"""Compute the result tables from the score files written by score.py.

Read from the results directory:
  base_validation_<prompt>.jsonl  zero-shot base model on validation, one file per prompt wording
  base_test.jsonl                 zero-shot base model on test, selected wording
  finetuned_test.jsonl            base model plus adapter on test, selected wording

Two Markdown tables are printed and metrics.json is written next to the score files.

Prompt table (validation split): the zero-shot base model with each wording.

Result table (test split):
  constant      always answers "not offensive", the majority class of the training split
  base          Qwen3-0.6B zero-shot, decision threshold 0
  base, tuned   the same scores with the threshold that maximises macro-F1 on
                the validation split, to separate what the base model knows
                from how well its two answers happen to be calibrated
  fine-tuned    Qwen3-0.6B with the LoRA adapter, decision threshold 0
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from prompts import PROMPTS, SELECTED


def read_scores(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    labels = np.array([r["label"] for r in rows])
    scores = np.array([r["score"] for r in rows])
    return labels, scores, [r["top"] for r in rows]


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Threshold t that maximises macro-F1 of the rule `score > t`.

    Every way of cutting the score-sorted rows into "top k are positive" is
    scored at once with cumulative sums.
    """
    order = np.argsort(-scores, kind="stable")
    y, s = labels[order], scores[order]
    n, positives = len(y), int(y.sum())
    k = np.arange(n + 1)
    tp = np.concatenate([[0], np.cumsum(y)])
    fp = k - tp
    fn = positives - tp
    tn = (n - positives) - fp
    with np.errstate(invalid="ignore"):
        f1_pos = np.nan_to_num(2 * tp / (2 * tp + fp + fn))
        f1_neg = np.nan_to_num(2 * tn / (2 * tn + fp + fn))
    macro_f1 = (f1_pos + f1_neg) / 2
    # A cut between two equal scores cannot be expressed as a threshold.
    macro_f1[1:n][s[:-1] == s[1:]] = -1.0
    best = int(np.argmax(macro_f1))
    if best == 0:
        return float(s[0]) + 1.0
    if best == n:
        return float(s[-1]) - 1.0
    return float((s[best - 1] + s[best]) / 2)


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(np.mean(f1)),
        "f1_offensive": float(f1[1]),
        "precision_offensive": float(precision[1]),
        "recall_offensive": float(recall[1]),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def scored_system(path: Path, threshold: float | None = None) -> dict:
    labels, scores, tops = read_scores(path)
    threshold = 0.0 if threshold is None else threshold
    return {
        "rows": int(len(labels)),
        "threshold": threshold,
        **metrics(labels, (scores > threshold).astype(int)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        # What free greedy decoding would start with, most frequent first.
        "top_tokens": dict(Counter(tops).most_common(3)),
    }


def build_report(results_dir: Path) -> dict:
    prompts = {name: scored_system(results_dir / f"base_validation_{name}.jsonl") for name in PROMPTS}
    best = max(prompts, key=lambda name: prompts[name]["roc_auc"])
    if best != SELECTED:
        raise ValueError(f"prompts.SELECTED is {SELECTED!r} but {best!r} has the best zero-shot ROC-AUC on validation")

    val_labels, val_scores, _ = read_scores(results_dir / f"base_validation_{SELECTED}.jsonl")
    test_labels, _, _ = read_scores(results_dir / "base_test.jsonl")
    if not np.array_equal(test_labels, read_scores(results_dir / "finetuned_test.jsonl")[0]):
        raise ValueError("base_test.jsonl and finetuned_test.jsonl do not hold the same rows")
    return {
        "prompt_selection_on_validation": prompts,
        "test": {
            "constant": {"rows": int(len(test_labels)), **metrics(test_labels, np.zeros_like(test_labels))},
            "base": scored_system(results_dir / "base_test.jsonl"),
            "base_tuned": scored_system(results_dir / "base_test.jsonl", best_threshold(val_labels, val_scores)),
            "finetuned": scored_system(results_dir / "finetuned_test.jsonl"),
        },
    }


def prompt_table(prompts: dict) -> str:
    lines = ["| Prompt wording | ROC-AUC | Macro-F1 | Accuracy |", "|---|---:|---:|---:|"]
    for name, m in prompts.items():
        lines.append(f"| `{name}` | {m['roc_auc']:.4f} | {m['macro_f1']:.4f} | {m['accuracy']:.4f} |")
    return "\n".join(lines)


def result_table(test: dict) -> str:
    names = {
        "constant": 'Constant "not offensive"',
        "base": "Qwen3-0.6B zero-shot",
        "base_tuned": "Qwen3-0.6B zero-shot, threshold tuned on validation",
        "finetuned": "Qwen3-0.6B + LoRA (this repo)",
    }
    lines = [
        "| System | Accuracy | Macro-F1 | F1 offensive | Precision | Recall | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, name in names.items():
        m = test[key]
        auc = f"{m['roc_auc']:.4f}" if "roc_auc" in m else "n/a"
        lines.append(
            f"| {name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['f1_offensive']:.4f} "
            f"| {m['precision_offensive']:.4f} | {m['recall_offensive']:.4f} | {auc} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the result tables from score files.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    report = build_report(args.results_dir)
    (args.results_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Zero-shot base model on the validation split, by prompt wording:\n\n{prompt_table(report['prompt_selection_on_validation'])}\n")
    print(f"Test split, prompt wording `{SELECTED}`:\n\n{result_table(report['test'])}")


if __name__ == "__main__":
    main()
