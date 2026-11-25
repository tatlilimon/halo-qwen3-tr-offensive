"""CPU checks that run on a laptop: prompt rendering, the Halo answer marker, the threshold search and a tiny scoring run."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

import report
import score
from prompts import PROMPTS, SELECTED, user_message

CONFIGS = [yaml.safe_load(path.read_text()) for path in sorted((Path(__file__).parent.parent / "configs").glob("*.yaml"))]
MARKER = CONFIGS[0]["assistant_message_template"]
# The CPU checks use the smaller model; both models ship the same tokenizer and chat template.
SMALL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(SMALL, revision=score.MODELS[SMALL])


def test_configs_train_the_models_that_are_scored():
    assert {c["model_name_or_path"]: c["model_revision"] for c in CONFIGS} == score.MODELS
    assert all(c["assistant_message_template"] == MARKER for c in CONFIGS)


def test_models_share_the_chat_template(tokenizer):
    for model, revision in score.MODELS.items():
        assert AutoTokenizer.from_pretrained(model, revision=revision).chat_template == tokenizer.chat_template


def test_training_row_extends_the_scoring_prompt(tokenizer):
    messages = [user_message("örnek metin")]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    for label, word in enumerate(PROMPTS[SELECTED].answers):
        row = tokenizer.apply_chat_template([*messages, {"role": "assistant", "content": word}], tokenize=False)
        assert row == prompt + word + "<|im_end|>\n"
        # The position score.py reads is where a training row holds the first answer token.
        row_ids = tokenizer.encode(row, add_special_tokens=False)
        assert row_ids[: len(prompt_ids)] == prompt_ids
        assert row_ids[len(prompt_ids)] == score.answer_token_ids(tokenizer, SELECTED)[label]


def test_marker_occurs_once_in_a_tokenized_row(tokenizer):
    # Halo finds the trained span by searching the marker's token ids in the row's token ids.
    row = tokenizer.apply_chat_template(
        [user_message("örnek metin"), {"role": "assistant", "content": PROMPTS[SELECTED].answers[1]}], tokenize=False
    )
    row_ids = tokenizer.encode(row, add_special_tokens=False)
    marker_ids = tokenizer.encode(MARKER, add_special_tokens=False)
    assert sum(row_ids[i : i + len(marker_ids)] == marker_ids for i in range(len(row_ids))) == 1


@pytest.mark.parametrize("prompt", sorted(PROMPTS))
def test_answers_start_with_different_tokens(tokenizer, prompt):
    negative, positive = score.answer_token_ids(tokenizer, prompt)
    assert negative != positive


def test_best_threshold_matches_brute_force():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 200)
    scores = np.round(rng.normal(labels.astype(float), 1.0), 1)  # rounding creates tied scores
    # Every distinct outcome of `score > t`: all positive, then one cut at each distinct score.
    candidates = [scores.min() - 1, *np.unique(scores)]
    best = max(f1_score(labels, scores > t, average="macro", zero_division=0) for t in candidates)
    found = report.best_threshold(labels, scores)
    assert f1_score(labels, scores > found, average="macro", zero_division=0) == pytest.approx(best)


def test_bootstrap_interval_contains_the_point_estimate():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 500)
    predictions = np.where(rng.random(500) < 0.8, labels, 1 - labels)
    low, high = report.bootstrap_interval(labels, predictions)
    assert low < f1_score(labels, predictions, average="macro") < high
    assert high - low < 0.1


def test_scoring_runs_on_cpu():
    tokenizer, model = score.load(SMALL, None, "cpu")
    rows = score.score_texts(tokenizer, model, ["Günaydın, bugün hava çok güzel.", "Bu filmi çok beğendim."], batch_size=2)
    assert len(rows) == 2
    assert all(np.isfinite(r["score"]) and isinstance(r["top"], str) for r in rows)
