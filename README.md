# halo-qwen3-tr-offensive

LoRA adapters for Qwen3-0.6B and Qwen3-1.7B that answer whether a Turkish tweet is offensive, trained with [Halo](https://github.com/whitecircle/halo) on one Modal GPU.

The repo holds the whole pipeline: data preparation, the Halo configs, the Modal app, the scoring code, a TF-IDF baseline, and the score files behind every number below. The dataset is OffensEval-TR 2020, the Turkish track of SemEval-2020 Task 12, and all test numbers are on its official test set.

## Results

Test split, 3,528 tweets, 20.3% offensive:

| System | Macro-F1 [95% interval] | Accuracy | Precision | Recall | ROC-AUC | Macro-F1, tuned threshold |
|---|---:|---:|---:|---:|---:|---:|
| Constant "not offensive" | 0.444 [0.439, 0.448] | 0.797 | 0.000 | 0.000 | n/a | n/a |
| TF-IDF character n-grams + logistic regression | 0.708 [0.688, 0.730] | 0.859 | 0.898 | 0.345 | 0.873 | 0.770 [0.752, 0.787] |
| Qwen3-0.6B zero-shot (`tr` wording) | 0.169 [0.159, 0.178] | 0.203 | 0.203 | 1.000 | 0.564 | 0.539 [0.522, 0.556] |
| Qwen3-0.6B + LoRA | 0.758 [0.739, 0.776] | 0.863 | 0.738 | 0.503 | 0.872 | 0.768 [0.750, 0.785] |
| Qwen3-1.7B zero-shot (`en_text_first` wording) | 0.389 [0.373, 0.404] | 0.389 | 0.240 | 0.929 | 0.733 | 0.640 [0.621, 0.658] |
| Qwen3-1.7B + LoRA | 0.788 [0.770, 0.806] | 0.875 | 0.750 | 0.578 | 0.897 | 0.792 [0.775, 0.807] |

Macro-F1 is the official metric of the task. Precision and recall are for the offensive class. All columns except the last use each system's default decision; the last column uses a decision threshold tuned on the validation split. Intervals are bootstrap intervals over the test tweets. Each zero-shot row uses the prompt wording that works best for that model, see [Prompt wording](#prompt-wording).

What the table says:

1. Before training, neither model does the task. Zero-shot Qwen3-0.6B answers "evet" (yes) for 3,527 of the 3,528 test tweets and its scores barely rank the tweets (ROC-AUC 0.564). Zero-shot Qwen3-1.7B ranks them better (ROC-AUC 0.733) but also answers yes far too often, and reaches macro-F1 0.640 only with a threshold tuned on the validation split.
2. After two epochs of LoRA training, Qwen3-0.6B reaches macro-F1 0.758 and Qwen3-1.7B 0.788, both at the default decision.
3. A TF-IDF plus logistic regression baseline that trains in 21 seconds on a laptop CPU is a serious competitor. The 0.6B adapter does not beat it: both rank the tweets equally well (ROC-AUC 0.872 and 0.873) and with tuned thresholds they are tied (0.768 and 0.770). The 1.7B adapter ranks better (ROC-AUC 0.897) and is ahead at the default decision (0.788 against 0.708), but with tuned thresholds its lead shrinks to 0.792 against 0.770 and the intervals overlap.
4. For context, 46 teams submitted to this track in 2020. The best system scored 0.8258 and the majority baseline 0.4435 ([Zampieri et al., 2020](https://arxiv.org/abs/2006.07235), Table 11). The constant row above reproduces that baseline. The 1.7B adapter's 0.7885 falls between the 6th (0.7933) and the 7th (0.7859) submission, the 0.6B adapter's 0.7576 between the 21st (0.7578) and the 22nd (0.7553).

Validation split, 1,756 tweets held out from the official training set, 19.1% offensive:

| System | Macro-F1 [95% interval] | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Constant "not offensive" | 0.447 [0.441, 0.453] | 0.809 | 0.000 | 0.000 | n/a |
| TF-IDF character n-grams + logistic regression | 0.667 [0.636, 0.698] | 0.850 | 0.793 | 0.287 | 0.845 |
| Qwen3-0.6B zero-shot (`tr` wording) | 0.163 [0.150, 0.176] | 0.192 | 0.191 | 1.000 | 0.571 |
| Qwen3-0.6B + LoRA | 0.701 [0.671, 0.730] | 0.845 | 0.655 | 0.397 | 0.837 |
| Qwen3-1.7B zero-shot (`en_text_first` wording) | 0.384 [0.363, 0.407] | 0.385 | 0.226 | 0.919 | 0.706 |
| Qwen3-1.7B + LoRA | 0.737 [0.709, 0.764] | 0.856 | 0.671 | 0.481 | 0.875 |

All tables are printed by `report.py` from the score files in `results/`. `results/metrics.json` has the confusion matrices and the tuned thresholds.

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

`score.py` does not generate text. It runs the model once per tweet and takes the logit of the first token of the "offensive" answer minus the logit of the first token of the "not offensive" answer at the answer position. A positive score predicts "offensive". This is greedy decoding restricted to the two answers, so there is no output to parse and a base model cannot lose points for formatting.

### Prompt wording

Three wordings were tried with the zero-shot base models, on the validation split only:

| Prompt wording | Qwen3-0.6B ROC-AUC | Qwen3-0.6B Macro-F1 | Qwen3-1.7B ROC-AUC | Qwen3-1.7B Macro-F1 |
|---|---:|---:|---:|---:|
| `en` | 0.398 | 0.160 | 0.705 | 0.443 |
| `en_text_first` | 0.498 | 0.452 | 0.706 | 0.384 |
| `tr` | 0.571 | 0.163 | 0.626 | 0.551 |

`tr` is the Turkish wording above. `en` and `en_text_first` are English wordings with yes/no answers; all three are in `prompts.py`.

Both adapters are trained and scored with `tr`. It was chosen with Qwen3-0.6B, for which it has the highest zero-shot ROC-AUC, before the larger model was trained. Zero-shot Qwen3-1.7B ranks better with the English wordings, so its zero-shot rows in the result tables use `en_text_first`. Each base model is therefore shown with its best wording of the three.

## Training

The configs are `configs/sft-qwen3-0.6b-lora.yaml` and `configs/sft-qwen3-1.7b-lora.yaml`; they differ only in the base model. `modal_app.py` runs them with `halo launch sft` inside the image `public.ecr.aws/whitecircle/halo:hopper-1.3.0rc1`, on one GPU requested from Modal as `H100`.

- LoRA with rank 16, alpha 32 and dropout 0.05 on all attention and MLP projections.
- 2 epochs, learning rate 1e-4 with cosine decay and 5% warmup, seed 42.
- Halo packs the 29,986 rows into 1,814 blocks of 2,048 tokens and trains on 4 blocks per step, which gives 908 steps over 7.18 million tokens. The loss covers the answer turn only.

| | Qwen3-0.6B | Qwen3-1.7B |
|---|---:|---:|
| Trainable parameters | 10,092,544 (1.67%) | 17,432,576 (1.00%) |
| Training loop | 261 s | 293 s |
| Validation loss after epoch 1 and 2 | 0.0396, 0.0366 | 0.0341, 0.0334 |

The full training logs are `results/qwen3-0.6b_halo_run.log` and `results/qwen3-1.7b_halo_run.log`.

There was no hyperparameter search. The learning rate, scheduler and target modules are those of Halo's shipped Qwen3 LoRA recipe (`examples/sft/qwen3/qwen3-4b-ultrachat-lora.yaml`); rank, epochs and batch size were set once. All values were fixed before the test set was scored, and the final checkpoint is used. Both models are reported; on the validation split the 1.7B adapter is the better one.

Each model was trained twice from scratch. Both runs wrote the same adapter file, byte for byte, and scoring the same adapter twice gave the same score file, so on this image and GPU type the results are reproducible exactly. SHA-256 of `adapter_model.safetensors`:

```
7050546b18b0d5dc4ea16ec84a7a1174b6c68fe0eaba01cdb8b08f613fff7e20  runs/sft-qwen3-0.6b-lora/checkpoint-908
7f75aeeeba3e559318d102f259c42c049d79a4bc4193f4f0eda1ae29daad7609  runs/sft-qwen3-1.7b-lora/checkpoint-908
```

## Reproduce

You need [uv](https://docs.astral.sh/uv/) and a Modal account with a payment method on file, which Modal requires for GPU functions.

```bash
uv run modal token new    # once
./reproduce.sh
```

`reproduce.sh` runs these steps:

```bash
uv sync
uv run pytest                      # CPU checks, about 15 seconds after the first model download
uv run python prepare_data.py      # writes data/
uv run python baseline_tfidf.py    # writes results/tfidf_*.jsonl

# for each model, here Qwen3-1.7B:
uv run modal run modal_app.py::score --split validation --name qwen3-1.7b_zeroshot_validation_tr \
    --model Qwen/Qwen3-1.7B --prompt tr          # and the same for en and en_text_first
uv run modal run modal_app.py::train --config sft-qwen3-1.7b-lora
uv run modal run modal_app.py::score --split validation --name qwen3-1.7b_lora_validation \
    --model Qwen/Qwen3-1.7B --adapter runs/sft-qwen3-1.7b-lora/checkpoint-908
uv run modal run modal_app.py::score --split test --name qwen3-1.7b_lora_test \
    --model Qwen/Qwen3-1.7B --adapter runs/sft-qwen3-1.7b-lora/checkpoint-908
uv run modal run modal_app.py::score --split test --name qwen3-1.7b_zeroshot_test_en_text_first \
    --model Qwen/Qwen3-1.7B --prompt en_text_first

uv run python report.py            # prints the tables, writes results/metrics.json
```

Measured times: the first use of the Halo image on Modal took 6 minutes 18 seconds to import, once. A training command took 5 minutes 19 seconds (0.6B) and 5 minutes 46 seconds (1.7B) end to end, and a scoring command 40 to 60 seconds.

The trained adapters stay on the Modal volume. To copy one to `adapter/`:

```bash
mkdir -p adapter
uv run modal volume get halo-qwen3-tr-offensive runs/sft-qwen3-1.7b-lora/checkpoint-908/adapter_model.safetensors adapter/
uv run modal volume get halo-qwen3-tr-offensive runs/sft-qwen3-1.7b-lora/checkpoint-908/adapter_config.json adapter/
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

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B"), "adapter")


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

The prompt has to match the training prompt exactly, and `enable_thinking=False` is required. This example was run on a CPU machine with 20 GB of memory; in fp32 the 1.7B model alone takes about 8 GB. To score a file of tweets the way the tables were made, use `score.py --model Qwen/Qwen3-1.7B --adapter adapter`.

## Limitations and known failures

- The 1.7B adapter misses four in ten offensive tweets at its default decision: recall is 0.578 on the test split, at precision 0.750. The 0.6B adapter misses half (recall 0.503). The tuned thresholds in `results/metrics.json` trade precision for recall.
- From reading a sample of the errors of both adapters: missed tweets are mostly insults without explicit profanity, sarcasm and political accusations. False alarms are mostly heated political or religious arguments.
- The 0.6B adapter does not beat the TF-IDF baseline, and the 1.7B adapter beats it clearly only at the default decision. Use the baseline if you need a cheap classifier for this data.
- The numbers come from one seed and no hyperparameter search. Because a rerun with the same seed gives the same adapter, the variance across seeds was not measured. The intervals describe the sampling noise of the split only.
- The base models were only measured with thinking disabled, the mode the adapters use.
- The reported scores were computed in bf16 on the GPU. For the 0.6B adapter, scoring the first 96 test tweets in fp32 on a CPU gave the same 96 decisions with logit differences up to 0.36, so tweets near the threshold can flip between setups. This was not measured for the 1.7B adapter.
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
- Base models: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) and [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), Apache-2.0.
- Dataset: OffensEval-TR 2020, CC BY 2.0. The score files in `results/` hold labels and scores but no tweet text.
- Training framework: [Halo](https://github.com/whitecircle/halo) 1.3.0rc1, under the Halo License.

Çağrı Çöltekin. 2020. A Corpus of Turkish Offensive Language on Social Media. In Proceedings of the 12th International Conference on Language Resources and Evaluation (LREC).

Marcos Zampieri, Preslav Nakov, Sara Rosenthal, Pepa Atanasova, Georgi Karadzhov, Hamdy Mubarak, Leon Derczynski, Zeses Pitenis and Çağrı Çöltekin. 2020. SemEval-2020 Task 12: Multilingual Offensive Language Identification in Social Media (OffensEval 2020). arXiv:2006.07235.
