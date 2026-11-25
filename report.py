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

Macro-F1 comes with a 95% bootstrap interval over the evaluated rows. It
describes the sampling noise of the split, not the variance between training runs.
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


def macro_f1_from_counts(tp, fp, fn, tn):
    """Macro-F1 from confusion counts; works on arrays of counts. A class with no rows scores 0."""
    with np.errstate(invalid="ignore"):
        f1_offensive = np.nan_to_num(2 * tp / (2 * tp + fp + fn))
        f1_not_offensive = np.nan_to_num(2 * tn / (2 * tn + fp + fn))
    return (f1_offensive + f1_not_offensive) / 2


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Threshold t that maximises macro-F1 of the rule `score > t`.

    Every way of cutting the score-sorted rows into "top k are positive" is
    scored at once with cumulative sums.
    """
    order = np.argsort(-scores, kind="stable")
    y, s = labels[order], scores[order]
    n, positives = len(y), int(y.sum())
    tp = np.concatenate([[0], np.cumsum(y)])
    fp = np.arange(n + 1) - tp
    fn = positives - tp
    macro_f1 = macro_f1_from_counts(tp, fp, fn, (n - positives) - fp)
    # A cut between two equal scores cannot be expressed as a threshold.
    macro_f1[1:n][s[:-1] == s[1:]] = -1.0
    best = int(np.argmax(macro_f1))
    if best == 0:
        return float(s[0]) + 1.0
    if best == n:
        return float(s[-1]) - 1.0
    return float((s[best - 1] + s[best]) / 2)


def bootstrap_interval(labels: np.ndarray, predictions: np.ndarray, resamples: int = 2000) -> list[float]:
    """95% percentile interval of macro-F1 over rows drawn with replacement."""
    rows = np.random.default_rng(0).integers(0, len(labels), (resamples, len(labels)))
    y, p = labels[rows] == 1, predictions[rows] == 1
    tp, fp, fn = (y & p).sum(axis=1), (~y & p).sum(axis=1), (y & ~p).sum(axis=1)
    values = macro_f1_from_counts(tp, fp, fn, len(labels) - tp - fp - fn)
    return [float(v) for v in np.percentile(values, [2.5, 97.5])]


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(np.mean(f1)),
        "macro_f1_interval": bootstrap_interval(labels, predictions),
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
        lines.append(f"| `{name}` | {m['roc_auc']:.3f} | {m['macro_f1']:.3f} | {m['accuracy']:.3f} |")
    return "\n".join(lines)


def with_interval(m: dict) -> str:
    low, high = m["macro_f1_interval"]
    return f"{m['macro_f1']:.3f} [{low:.3f}, {high:.3f}]"


def result_table(systems: dict) -> str:
    tuned = any("tuned" in m for m in systems.values())
    header = "| System | Macro-F1 [95% interval] | Accuracy | Precision | Recall | ROC-AUC |"
    rule = "|---|---:|---:|---:|---:|---:|"
    if tuned:
        header += " Macro-F1, tuned threshold |"
        rule += "---:|"
    lines = [header, rule]
    for key, m in systems.items():
        name = SYSTEMS.get(key, 'Constant "not offensive"')
        auc = f"{m['roc_auc']:.3f}" if "roc_auc" in m else "n/a"
        line = (
            f"| {name} | {with_interval(m)} | {m['accuracy']:.3f} | {m['precision_offensive']:.3f} "
            f"| {m['recall_offensive']:.3f} | {auc} |"
        )
        if tuned:
            line += f" {with_interval(m['tuned'])} |" if "tuned" in m else " n/a |"
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the result tables from score files.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--file", type=Path, help="print the metrics of one score file instead of the report")
    args = parser.parse_args()
    if args.file:
        print(json.dumps(scored_system(args.file), indent=2, ensure_ascii=False))
        return
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
