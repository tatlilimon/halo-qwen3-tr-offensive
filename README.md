# halo-qwen3-tr-offensive

A LoRA adapter for Qwen3-0.6B that answers whether a Turkish tweet is offensive, trained with [Halo](https://github.com/whitecircle/halo) on one Modal GPU.

The repo holds the whole pipeline: data preparation, the Halo config, the Modal app, the scoring code, a TF-IDF baseline, and the score files behind every number below. The dataset is OffensEval-TR 2020, the Turkish track of SemEval-2020 Task 12, and all test numbers are on its official test set.

## Results

Test split, 3,528 tweets, 20.3% offensive:

| System | Macro-F1 [95% interval] | Accuracy | Precision | Recall | ROC-AUC | Macro-F1, tuned threshold |
|---|---:|---:|---:|---:|---:|---:|
| Constant "not offensive" | 0.444 [0.439, 0.448] | 0.797 | 0.000 | 0.000 | n/a | n/a |
| TF-IDF character n-grams + logistic regression | 0.708 [0.688, 0.730] | 0.859 | 0.898 | 0.345 | 0.873 | 0.770 [0.752, 0.787] |
| Qwen3-0.6B zero-shot | 0.169 [0.159, 0.178] | 0.203 | 0.203 | 1.000 | 0.564 | 0.539 [0.522, 0.556] |
| Qwen3-0.6B + LoRA (this repo) | 0.758 [0.739, 0.776] | 0.863 | 0.738 | 0.503 | 0.872 | 0.768 [0.750, 0.785] |

Macro-F1 is the official metric of the task. Precision and recall are for the offensive class. All columns except the last use each system's default decision; the last column uses a decision threshold tuned on the validation split. Intervals are bootstrap intervals over the test tweets.

What the table says:

1. Before training, the model cannot do the task. Zero-shot Qwen3-0.6B answers "evet" (yes) for 3,527 of the 3,528 test tweets, which puts it below the constant baseline, and its scores barely rank the tweets (ROC-AUC 0.564). Tuning its decision threshold on the validation split only lifts it to macro-F1 0.539.
2. After two epochs of LoRA training the same model reaches macro-F1 0.758 and ROC-AUC 0.872.
3. The adapter is not better than a TF-IDF plus logistic regression baseline that trains in 21 seconds on a laptop CPU. Both rank the tweets equally well (ROC-AUC 0.872 and 0.873). The adapter is ahead at the default decision (0.758 against 0.708) because the linear model is more conservative (recall 0.345). With thresholds tuned on validation the two are tied (0.768 and 0.770).
4. For context, 46 teams submitted to this track in 2020. The best system scored 0.8258 and the majority baseline 0.4435 ([Zampieri et al., 2020](https://arxiv.org/abs/2006.07235), Table 11). The constant row above reproduces that baseline. The adapter's 0.7576 falls between the 21st (0.7578) and the 22nd (0.7553) submission.

Validation split, 1,756 tweets held out from the official training set, 19.1% offensive:

| System | Macro-F1 [95% interval] | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Constant "not offensive" | 0.447 [0.441, 0.453] | 0.809 | 0.000 | 0.000 | n/a |
| TF-IDF character n-grams + logistic regression | 0.667 [0.636, 0.698] | 0.850 | 0.793 | 0.287 | 0.845 |
| Qwen3-0.6B zero-shot | 0.163 [0.150, 0.176] | 0.192 | 0.191 | 1.000 | 0.571 |
| Qwen3-0.6B + LoRA (this repo) | 0.701 [0.671, 0.730] | 0.845 | 0.655 | 0.397 | 0.837 |

Both tables are printed by `report.py` from the score files in `results/`. `results/metrics.json` has the confusion matrices and the tuned thresholds.

## Data

OffensEval-TR 2020 ([Çöltekin, 2020](https://coltekin.github.io/offensive-turkish/troff.pdf)) is a set of tweets sampled from the Turkish Twitter stream and labeled offensive or not offensive by volunteers. User mentions are replaced by `@USER`. The license is CC BY 2.0.

`prepare_data.py` reads it from [`stefan-it/offenseval2020_tr`](https://huggingface.co/datasets/stefan-it/offenseval2020_tr) at a pinned revision. That version splits the official training set into 30,000 training and 1,756 development tweets and leaves the official test set unchanged.

| Split | Tweets | Offensive |
|---|---:|---:|
| train | 29,986 | 5,796 (19.3%) |
| validation | 1,756 | 335 (19.1%) |
| test | 3,528 | 716 (20.3%) |

The only cleaning step: training tweets whose text, ignoring case and whitespace, repeats a test, validation or earlier training tweet are dropped. That removes 14 of the 30,000. Validation and test are used as published.

## Prompt and scoring

Each tweet goes into one user message:

```
Aşağıdaki metin saldırgan mı? Saldırgan metin; hakaret, küfür ya da bir kişiye veya gruba yönelik açık ya da örtülü saldırı içerir. Yalnızca tek kelimeyle cevap ver: evet ya da hayır.

Metin: <tweet>
```

In English: "Is the following text offensive? An offensive text contains insults, profanity, or an open or veiled attack on a person or group. Answer with one word only: yes or no."

The assistant turn is `evet` (offensive) or `hayır` (not offensive). The stock Qwen3 chat template is used with thinking disabled, so a training row and a scoring prompt render to the same text up to the answer. `tests/test_smoke.py` checks this.

`score.py` does not generate text. It runs the model once per tweet and takes the logit of the first token of `evet` minus the logit of the first token of `hayır` at the answer position. A positive score predicts "offensive". This is greedy decoding restricted to the two answers, so there is no output to parse and the base model cannot lose points for formatting.

### Prompt wording

Three wordings were tried with the zero-shot base model, on the validation split only:

| Prompt wording | ROC-AUC | Macro-F1 | Accuracy |
|---|---:|---:|---:|
| `en` | 0.398 | 0.160 | 0.191 |
| `en_text_first` | 0.498 | 0.452 | 0.513 |
| `tr` | 0.571 | 0.163 | 0.192 |

`tr` is the Turkish wording above. `en` and `en_text_first` are English wordings with yes/no answers; all three are in `prompts.py`. None of them makes the base model useful. The adapter is trained and scored with `tr`, the wording with the highest ROC-AUC, so the zero-shot rows in the result tables show the base model with the best of the three.

## Training

The config is `configs/sft-qwen3-0.6b-lora.yaml`. `modal_app.py` runs it with `halo launch sft` inside the image `public.ecr.aws/whitecircle/halo:hopper-1.3.0rc1`, on one GPU requested from Modal as `H100`.

- LoRA with rank 16, alpha 32 and dropout 0.05 on all attention and MLP projections: 10,092,544 trainable parameters, 1.67% of the model.
- 2 epochs, learning rate 1e-4 with cosine decay and 5% warmup, seed 42.
- Halo packs the 29,986 rows into 1,814 blocks of 2,048 tokens and trains on 4 blocks per step, which gives 908 steps. The loss covers the answer turn only.
- The training loop took 261 seconds for 7.18 million tokens. The validation loss was 0.0396 after the first epoch and 0.0366 after the second. The full log is `results/halo_run.log`.

There was no hyperparameter search. The learning rate, scheduler and target modules are those of Halo's shipped Qwen3 LoRA recipe (`examples/sft/qwen3/qwen3-4b-ultrachat-lora.yaml`); rank, epochs and batch size were set once for this smaller model and dataset. All values were fixed before the test set was scored, and the final checkpoint is used.

The training command was run twice from scratch. Both runs wrote the same adapter file (SHA-256 `7050546b18b0d5dc4ea16ec84a7a1174b6c68fe0eaba01cdb8b08f613fff7e20`) and the same test scores, so on this image and GPU type the run is reproducible bit for bit.

## Reproduce

You need [uv](https://docs.astral.sh/uv/) and a Modal account with a payment method on file, which Modal requires for GPU functions.

```bash
uv sync
uv run pytest                      # CPU checks, about 15 seconds after the first model download
uv run python prepare_data.py      # writes data/
uv run python baseline_tfidf.py    # writes results/tfidf_*.jsonl

uv run modal token new             # once
uv run modal run modal_app.py::score --split validation --name base_validation_en --prompt en
uv run modal run modal_app.py::score --split validation --name base_validation_en_text_first --prompt en_text_first
uv run modal run modal_app.py::score --split validation --name base_validation_tr --prompt tr
uv run modal run modal_app.py::train
uv run modal run modal_app.py::score --split validation --name finetuned_validation \
    --adapter runs/sft-qwen3-0.6b-lora/checkpoint-908
uv run modal run modal_app.py::score --split test --name base_test
uv run modal run modal_app.py::score --split test --name finetuned_test \
    --adapter runs/sft-qwen3-0.6b-lora/checkpoint-908

uv run python report.py            # prints the tables, writes results/metrics.json
```

Measured times: the first use of the Halo image on Modal took 6 minutes 18 seconds to import, once. The training command took 5 minutes 19 seconds end to end and each scoring command 40 to 55 seconds. Together the GPU commands are about 10 minutes of wall time.

The trained adapter stays on the Modal volume. To copy it to `adapter/`:

```bash
mkdir -p adapter
uv run modal volume get halo-qwen3-tr-offensive runs/sft-qwen3-0.6b-lora/checkpoint-908/adapter_model.safetensors adapter/
uv run modal volume get halo-qwen3-tr-offensive runs/sft-qwen3-0.6b-lora/checkpoint-908/adapter_config.json adapter/
```

Local dependencies are pinned to the versions inside the Halo 1.3.0rc1 image (torch 2.11.0, transformers 5.20.2, peft 0.18.1, datasets 4.8.5), so the CPU checks and the GPU runs use the same library code.

## Use the adapter

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = (
    "Aşağıdaki metin saldırgan mı? Saldırgan metin; hakaret, küfür ya da bir kişiye veya gruba yönelik "
    "açık ya da örtülü saldırı içerir. Yalnızca tek kelimeyle cevap ver: evet ya da hayır.\n\nMetin: {text}"
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B"), "adapter")


def answer(text: str) -> str:
    messages = [{"role": "user", "content": PROMPT.format(text=text)}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False, return_dict=True, return_tensors="pt"
    )
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


print(answer("Günaydın, bugün hava çok güzel."))    # hayır
print(answer("Sen tam bir aptalsın, sus artık."))    # evet
```

The prompt has to match the training prompt exactly, and `enable_thinking=False` is required. To score a file of tweets the way the tables were made, use `score.py --adapter adapter`.

## Limitations and known failures

- The adapter misses half of the offensive tweets at its default decision: recall is 0.503 on the test split, at precision 0.738. The tuned threshold in `results/metrics.json` trades precision for recall.
- From reading its errors: missed tweets are mostly insults without explicit profanity, sarcasm, and profanity with obfuscated spelling. False alarms are mostly heated political or religious tweets that insult nobody, and tweets that mention violence.
- It does not beat a TF-IDF baseline in ranking quality. Use the baseline if you need a cheap classifier for this data.
- The numbers come from one seed and no hyperparameter search. Because a rerun with the same seed gives the same adapter, the variance across seeds was not measured. The intervals describe the sampling noise of the test split only.
- Only Qwen3-0.6B was trained. The base model was only measured with thinking disabled, the mode the adapter uses.
- The reported scores were computed in bf16 on the GPU. Scoring the first 96 test tweets in fp32 on a CPU gave the same 96 decisions with logit differences up to 0.36, so tweets near the threshold can flip between setups.
- The pipeline was only run on Modal with the `hopper` image.
- Most tweets in the dataset were labeled by a single volunteer annotator, and agreement on a doubly annotated sample was Cohen's kappa 0.761 (Zampieri et al., 2020, section 3). A model trained on these tweets inherits their label noise, topics and biases, and is not a moderation system on its own.

## Dataset choice

An earlier version of this repo trained on [`Toygar/turkish-offensive-language-detection`](https://huggingface.co/datasets/Toygar/turkish-offensive-language-detection), which merges OffensEval-TR with tweets from other sources. It was replaced because part of its test split can be solved from the data source instead of the content: 1,919 of its 8,851 test tweets (21.7%) contain a user handle that is not anonymized, and 99.7% of those are labeled offensive. The same holds for its 154 test tweets with a URL, all labeled offensive.

Both statistics can be checked with:

```python
import re
from datasets import load_dataset

test = load_dataset("Toygar/turkish-offensive-language-detection", revision="5aff92f9c824061b0781a5ff1bbf1e8246de5840", split="test")
handle = [bool(re.search(r"(?<!\w)@(?!USER\b)\w+", text)) for text in test["text"]]
with_handle = [label for label, has_handle in zip(test["label"], handle) if has_handle]
with_url = [label for label, text in zip(test["label"], test["text"]) if "http" in text]
print(len(with_handle), sum(with_handle) / len(with_handle), len(with_url), sum(with_url) / len(with_url))
```

## Licenses and references

- Code in this repo: Apache-2.0, see `LICENSE`.
- Base model: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), Apache-2.0.
- Dataset: OffensEval-TR 2020, CC BY 2.0. The score files in `results/` hold labels and scores but no tweet text.
- Training framework: [Halo](https://github.com/whitecircle/halo) 1.3.0rc1, under the Halo License.

Çağrı Çöltekin. 2020. A Corpus of Turkish Offensive Language on Social Media. In Proceedings of the 12th International Conference on Language Resources and Evaluation (LREC).

Marcos Zampieri, Preslav Nakov, Sara Rosenthal, Pepa Atanasova, Georgi Karadzhov, Hamdy Mubarak, Leon Derczynski, Zeses Pitenis and Çağrı Çöltekin. 2020. SemEval-2020 Task 12: Multilingual Offensive Language Identification in Social Media (OffensEval 2020). arXiv:2006.07235.
