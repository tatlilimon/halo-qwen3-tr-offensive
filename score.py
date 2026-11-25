"""Score labelled texts with the base model, or with the base model plus a LoRA adapter.

Each text is put into the prompt and the model's next-token logits are read
once. The score is the logit of the first token of the "offensive" answer minus
the logit of the first token of the "not offensive" answer, so a positive score
predicts "offensive". This equals greedy decoding restricted to the two answers
and needs no output parsing.

One JSON line is written per input row:
  {"label": 0 or 1, "score": float, "top": the most likely next token over the whole vocabulary}
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import PROMPTS, SELECTED, user_message

# Hub id and pinned revision of each base model.
MODELS = {
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    "unsloth/gpt-oss-20b-BF16": "5a6fe4c0c1a86147cb9dc1bb99486e61d12a3c9e",
}
# Stands in for the answer while a row is rendered, to find where the answer starts.
ANSWER_SLOT = "<<ANSWER>>"


def answer_token_ids(tokenizer, prompt: str) -> tuple[int, int]:
    """First token id of the (not offensive, offensive) answers of a prompt."""
    negative, positive = (tokenizer.encode(word, add_special_tokens=False)[0] for word in PROMPTS[prompt].answers)
    if negative == positive:
        raise ValueError(f"the answers of prompt {prompt!r} start with the same token")
    return negative, positive


def load(base: str, adapter: str | None, device: str, chat_template: Path | None = None):
    tokenizer = AutoTokenizer.from_pretrained(base, revision=MODELS[base])
    if chat_template:
        tokenizer.chat_template = chat_template.read_text(encoding="utf-8")
    # Left padding keeps the last position of every row on its final prompt token.
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base, revision=MODELS[base], dtype=dtype)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.to(torch.device(device))
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def score_texts(tokenizer, model, texts: list[str], batch_size: int, prompt: str = SELECTED) -> list[dict]:
    negative, positive = answer_token_ids(tokenizer, prompt)
    # A prompt is a rendered training row cut off where the answer starts, so the
    # model is scored on exactly the text it was trained to continue.
    prompts = []
    for text in texts:
        row = tokenizer.apply_chat_template(
            [user_message(text, prompt), {"role": "assistant", "content": ANSWER_SLOT}], tokenize=False
        )
        prompts.append(row[: row.rindex(ANSWER_SLOT)])
    input_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    # Batching rows of similar length keeps padding, and so run time, small.
    order = sorted(range(len(texts)), key=lambda i: len(input_ids[i]))
    results: dict[int, dict] = {}
    for start in range(0, len(order), batch_size):
        batch_rows = order[start : start + batch_size]
        batch = tokenizer.pad({"input_ids": [input_ids[i] for i in batch_rows]}, return_tensors="pt").to(model.device)
        logits = model(**batch).logits[:, -1, :].float()
        scores = (logits[:, positive] - logits[:, negative]).tolist()
        top_tokens = tokenizer.convert_ids_to_tokens(logits.argmax(dim=-1).tolist())
        for i, score, top in zip(batch_rows, scores, top_tokens):
            results[i] = {"score": score, "top": top}
    return [results[i] for i in range(len(texts))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score labelled texts with the base or the fine-tuned model.")
    parser.add_argument("--data", type=Path, required=True, help="JSONL with text and label fields")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODELS), required=True, help="base model")
    parser.add_argument("--adapter", help="directory of a LoRA adapter trained on the base model")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default=SELECTED)
    parser.add_argument("--chat-template", type=Path, help="Jinja file to use instead of the model's own template")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, help="score only the first N rows")
    args = parser.parse_args()

    with args.data.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if args.limit:
        rows = rows[: args.limit]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load(args.model, args.adapter, device, args.chat_template)
    scored = score_texts(tokenizer, model, [r["text"] for r in rows], args.batch_size, args.prompt)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row, result in zip(rows, scored):
            f.write(json.dumps({"label": row["label"], **result}, ensure_ascii=False) + "\n")
    correct = sum((s["score"] > 0) == bool(r["label"]) for r, s in zip(rows, scored))
    print(f"{args.out}: {len(rows)} rows on {device}, accuracy at threshold 0: {correct / len(rows):.4f}")


if __name__ == "__main__":
    main()
