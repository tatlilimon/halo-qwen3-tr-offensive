"""Compute the result tables from the score files in the results directory.

Score files, one JSON line per row with `label` and `score` (positive means offensive):
  base_validation_<prompt>.jsonl  zero-shot base model on validation, one file per prompt wording
  base_test.jsonl                 zero-shot base model on test, selected wording
  finetuned_validation.jsonl      base model plus adapter on validation
  finetuned_test.jsonl            base model plus adapter on test
  tfidf_validation.jsonl          TF-IDF baseline on validation
  tfidf_test.jsonl                TF-IDF baseline on test

Three Markdown tables are printed and metrics.json is written next to the score files.

Prompt table: the zero-shot base model on the validation split with each wording.

Result tables, one per split. Every system is scored at decision threshold 0,
which for the language model is greedy decoding between its two answers. On the
test split each system is also scored at the threshold that maximises macro-F1
on the validation split, which separates what a system knows from how well its
default decision happens to be calibrated.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from prompts import PROMPTS, SELECTED

SYSTEMS = {
    "tfidf": "TF-IDF character n-grams + logistic regression",
    "base": "Qwen3-0.6B zero-shot",
    "finetuned": "Qwen3-0.6B + LoRA (this repo)",
}


def read_scores(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    labels = np.array([r["label"] for r in rows])
    scores = np.array([r["score"] for r in rows])
    return labels, scores, [r["top"] for r in rows if "top" in r]


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


def scored_system(path: Path, tune_on: Path | None = None) -> dict:
    """Metrics of one score file; `tune_on` adds the metrics at the threshold tuned on that file."""
    labels, scores, tops = read_scores(path)
    result = {
        "rows": int(len(labels)),
        **metrics(labels, (scores > 0).astype(int)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }
    if tops:
        # What free greedy decoding would start with, most frequent first.
        result["top_tokens"] = dict(Counter(tops).most_common(3))
    if tune_on is not None:
        threshold = best_threshold(*read_scores(tune_on)[:2])
        result["tuned"] = {"threshold": threshold, **metrics(labels, (scores > threshold).astype(int))}
    return result


def constant_system(path: Path) -> dict:
    labels, _, _ = read_scores(path)
    return {"rows": int(len(labels)), **metrics(labels, np.zeros_like(labels))}


def build_report(results_dir: Path) -> dict:
    prompts = {name: scored_system(results_dir / f"base_validation_{name}.jsonl") for name in PROMPTS}
    best = max(prompts, key=lambda name: prompts[name]["roc_auc"])
    if best != SELECTED:
        raise ValueError(f"prompts.SELECTED is {SELECTED!r} but {best!r} has the best zero-shot ROC-AUC on validation")

    validation = {name: results_dir / f"{name}_validation.jsonl" for name in SYSTEMS}
    validation["base"] = results_dir / f"base_validation_{SELECTED}.jsonl"
    test = {name: results_dir / f"{name}_test.jsonl" for name in SYSTEMS}
    for files in (validation, test):
        labels = [read_scores(path)[0] for path in files.values()]
        if not all(np.array_equal(labels[0], other) for other in labels[1:]):
            raise ValueError(f"score files do not hold the same rows: {[p.name for p in files.values()]}")
    return {
        "prompt_selection_on_validation": prompts,
        "validation": {
            "constant": constant_system(validation["base"]),
            **{name: scored_system(validation[name]) for name in SYSTEMS},
        },
        "test": {
            "constant": constant_system(test["base"]),
            **{name: scored_system(test[name], tune_on=validation[name]) for name in SYSTEMS},
        },
    }


def prompt_table(prompts: dict) -> str:
    lines = ["| Prompt wording | ROC-AUC | Macro-F1 | Accuracy |", "|---|---:|---:|---:|"]
    for name, m in prompts.items():
        lines.append(f"| `{name}` | {m['roc_auc']:.4f} | {m['macro_f1']:.4f} | {m['accuracy']:.4f} |")
    return "\n".join(lines)


def result_table(systems: dict) -> str:
    tuned = any("tuned" in m for m in systems.values())
    header = "| System | Macro-F1 | Accuracy | Precision | Recall | ROC-AUC |"
    rule = "|---|---:|---:|---:|---:|---:|"
    if tuned:
        header += " Macro-F1, tuned threshold |"
        rule += "---:|"
    lines = [header, rule]
    for key, m in systems.items():
        name = SYSTEMS.get(key, 'Constant "not offensive"')
        auc = f"{m['roc_auc']:.4f}" if "roc_auc" in m else "n/a"
        line = (
            f"| {name} | {m['macro_f1']:.4f} | {m['accuracy']:.4f} | {m['precision_offensive']:.4f} "
            f"| {m['recall_offensive']:.4f} | {auc} |"
        )
        if tuned:
            line += f" {m['tuned']['macro_f1']:.4f} |" if "tuned" in m else " n/a |"
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the result tables from score files.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    report = build_report(args.results_dir)
    (args.results_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        "Zero-shot base model on the validation split, by prompt wording:\n\n"
        f"{prompt_table(report['prompt_selection_on_validation'])}\n"
    )
    for split in ("test", "validation"):
        rows = report[split]["constant"]["rows"]
        print(f"{split.capitalize()} split, {rows} rows:\n\n{result_table(report[split])}\n")


if __name__ == "__main__":
    main()
