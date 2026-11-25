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

CONFIG = yaml.safe_load((Path(__file__).parent.parent / "configs" / "sft-qwen3-0.6b-lora.yaml").read_text())
MARKER = CONFIG["assistant_message_template"]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(score.BASE_MODEL, revision=score.BASE_REVISION)


def test_config_trains_the_model_that_is_scored():
    assert CONFIG["model_name_or_path"] == score.BASE_MODEL
    assert CONFIG["model_revision"] == score.BASE_REVISION


def test_training_row_extends_the_scoring_prompt(tokenizer):
    messages = [user_message("örnek metin")]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert prompt.endswith(MARKER)
    for word in PROMPTS[SELECTED].answers:
        row = tokenizer.apply_chat_template([*messages, {"role": "assistant", "content": word}], tokenize=False)
        assert row == prompt + word + "<|im_end|>\n"


def test_marker_token_ids_locate_the_answer(tokenizer):
    # Halo finds the trained span by searching the marker's token ids in the row's token ids.
    offensive = PROMPTS[SELECTED].answers[1]
    row = tokenizer.apply_chat_template(
        [user_message("örnek metin"), {"role": "assistant", "content": offensive}], tokenize=False
    )
    row_ids = tokenizer.encode(row, add_special_tokens=False)
    marker_ids = tokenizer.encode(MARKER, add_special_tokens=False)
    starts = [i for i in range(len(row_ids)) if row_ids[i : i + len(marker_ids)] == marker_ids]
    assert len(starts) == 1
    assert row_ids[starts[0] + len(marker_ids)] == score.answer_token_ids(tokenizer, SELECTED)[1]


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


def test_scoring_runs_on_cpu():
    tokenizer, model = score.load(None, "cpu")
    rows = score.score_texts(tokenizer, model, ["Günaydın, bugün hava çok güzel.", "Bu filmi çok beğendim."], batch_size=2)
    assert len(rows) == 2
    assert all(np.isfinite(r["score"]) and isinstance(r["top"], str) for r in rows)
