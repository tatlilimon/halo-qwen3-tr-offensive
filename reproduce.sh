#!/usr/bin/env bash
# Rerun every step behind the tables in the README, in order.
# Needs uv and a Modal token (`uv run modal token new`).
set -euo pipefail
cd "$(dirname "$0")"

score() { uv run modal run modal_app.py::score "$@"; }

uv sync
uv run pytest
uv run python prepare_data.py
uv run python baseline_tfidf.py

for pair in "qwen3-0.6b Qwen/Qwen3-0.6B" "qwen3-1.7b Qwen/Qwen3-1.7B"; do
    read -r tag model <<<"$pair"
    for prompt in en en_text_first tr; do
        score --split validation --name "${tag}_zeroshot_validation_${prompt}" --model "$model" --prompt "$prompt"
    done
    uv run modal run modal_app.py::train --config "sft-${tag}-lora"
    # 908 optimizer steps: 1,814 packed blocks, 4 per step, 2 epochs.
    adapter="runs/sft-${tag}-lora/checkpoint-908"
    score --split validation --name "${tag}_lora_validation" --model "$model" --adapter "$adapter"
    score --split test --name "${tag}_lora_test" --model "$model" --adapter "$adapter"
done

# gpt-oss-20b is trained and scored with Halo's SFT chat template for that family.
gpt_oss() { score "$@" --model unsloth/gpt-oss-20b-BF16 --chat-template jinja-templates/gpt-oss/gpt-oss-multiturn.jinja; }
for prompt in en en_text_first tr; do
    gpt_oss --split validation --name "gpt-oss-20b_zeroshot_validation_${prompt}" --prompt "$prompt"
done
uv run modal run modal_app.py::train --config sft-gpt-oss-20b-lora
# 938 optimizer steps: 29,986 rows, 64 per step, 2 epochs.
gpt_oss --split validation --name gpt-oss-20b_lora_validation --adapter runs/sft-gpt-oss-20b-lora/checkpoint-938
gpt_oss --split test --name gpt-oss-20b_lora_test --adapter runs/sft-gpt-oss-20b-lora/checkpoint-938

# Each base model is scored on test with the wording that has its best validation ROC-AUC.
score --split test --name qwen3-0.6b_zeroshot_test_tr --model Qwen/Qwen3-0.6B --prompt tr
score --split test --name qwen3-1.7b_zeroshot_test_en_text_first --model Qwen/Qwen3-1.7B --prompt en_text_first
gpt_oss --split test --name gpt-oss-20b_zeroshot_test_en_text_first --prompt en_text_first

uv run python report.py
