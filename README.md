# halo-qwen3-tr-offensive

LoRA adapters for Qwen3-0.6B, Qwen3-1.7B and gpt-oss-20b that answer whether a Turkish tweet is offensive, trained with [Halo](https://github.com/whitecircle/halo) on one Modal GPU.

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
| gpt-oss-20b zero-shot (`en_text_first` wording) | 0.539 [0.520, 0.558] | 0.816 | 0.923 | 0.101 | 0.854 | 0.750 [0.732, 0.768] |
| gpt-oss-20b + LoRA | 0.825 [0.808, 0.841] | 0.895 | 0.796 | 0.648 | 0.930 | 0.827 [0.811, 0.843] |

Macro-F1 is the official metric of the task. Precision and recall are for the offensive class. All columns except the last use each system's default decision; the last column uses a decision threshold tuned on the validation split. Intervals are bootstrap intervals over the test tweets. Each zero-shot row uses the prompt wording that works best for that model, see [Prompt wording](#prompt-wording).

What the table says:

1. Before training, the two small models do not do the task. Zero-shot Qwen3-0.6B answers "evet" (yes) for 3,527 of the 3,528 test tweets and its scores barely rank the tweets (ROC-AUC 0.564). Zero-shot Qwen3-1.7B ranks them better (ROC-AUC 0.733) but answers yes far too often.
2. Zero-shot gpt-oss-20b already ranks the tweets about as well as the trained TF-IDF baseline (ROC-AUC 0.854 against 0.873), but its default decision almost never says "offensive" (recall 0.101). With a threshold tuned on validation it reaches macro-F1 0.750.
3. After two epochs of LoRA training, macro-F1 at the default decision is 0.758 for Qwen3-0.6B, 0.788 for Qwen3-1.7B and 0.825 for gpt-oss-20b.
4. A TF-IDF plus logistic regression baseline that trains in 21 seconds on a laptop CPU is a serious competitor for the small models. The 0.6B adapter does not beat it: both rank the tweets equally well (ROC-AUC 0.872 and 0.873) and with tuned thresholds they are tied (0.768 and 0.770). The 1.7B adapter ranks better (0.897), but with tuned thresholds its lead is 0.792 against 0.770 and the intervals overlap. The gpt-oss-20b adapter is clearly ahead on every column (ROC-AUC 0.930, tuned 0.827).
5. For context, 46 teams submitted to this track in 2020. The best system scored 0.8258, the second 0.8167, and the majority baseline 0.4435 ([Zampieri et al., 2020](https://arxiv.org/abs/2006.07235), Table 11). The constant row above reproduces that baseline. The gpt-oss-20b adapter's 0.8250 falls between the first and the second submission, the 1.7B adapter's 0.7885 between the 6th (0.7933) and the 7th (0.7859), and the 0.6B adapter's 0.7576 between the 21st (0.7578) and the 22nd (0.7553).

Validation split, 1,756 tweets held out from the official training set, 19.1% offensive:

| System | Macro-F1 [95% interval] | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Constant "not offensive" | 0.447 [0.441, 0.453] | 0.809 | 0.000 | 0.000 | n/a |
| TF-IDF character n-grams + logistic regression | 0.667 [0.636, 0.698] | 0.850 | 0.793 | 0.287 | 0.845 |
| Qwen3-0.6B zero-shot (`tr` wording) | 0.163 [0.150, 0.176] | 0.192 | 0.191 | 1.000 | 0.571 |
| Qwen3-0.6B + LoRA | 0.701 [0.671, 0.730] | 0.845 | 0.655 | 0.397 | 0.837 |
| Qwen3-1.7B zero-shot (`en_text_first` wording) | 0.384 [0.363, 0.407] | 0.385 | 0.226 | 0.919 | 0.706 |
| Qwen3-1.7B + LoRA | 0.737 [0.709, 0.764] | 0.856 | 0.671 | 0.481 | 0.875 |
| gpt-oss-20b zero-shot (`en_text_first` wording) | 0.514 [0.490, 0.539] | 0.822 | 0.958 | 0.069 | 0.835 |
| gpt-oss-20b + LoRA | 0.784 [0.757, 0.810] | 0.879 | 0.741 | 0.564 | 0.907 |

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

The assistant turn is `evet` (offensive) or `hayır` (not offensive). For the Qwen3 models the stock chat template renders it after an empty think block, which is the template's thinking-disabled form. gpt-oss-20b is trained with Halo's SFT template for that family, which puts the answer in the final channel with no reasoning before it.

`score.py` does not generate text. It renders a training row, cuts it off where the answer starts, runs the model once on that prompt and takes the logit of the first token of the "offensive" answer minus the logit of the first token of the "not offensive" answer. A positive score predicts "offensive". This is greedy decoding restricted to the two answers, so there is no output to parse and a base model cannot lose points for formatting. `tests/test_smoke.py` checks that this prompt equals the stock thinking-disabled prompt for Qwen3.

### Prompt wording

Three wordings were tried with the zero-shot base models, on the validation split only:

| Prompt wording | Qwen3-0.6B ROC-AUC | Qwen3-0.6B Macro-F1 | Qwen3-1.7B ROC-AUC | Qwen3-1.7B Macro-F1 | gpt-oss-20b ROC-AUC | gpt-oss-20b Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| `en` | 0.398 | 0.160 | 0.705 | 0.443 | 0.814 | 0.600 |
| `en_text_first` | 0.498 | 0.452 | 0.706 | 0.384 | 0.835 | 0.514 |
| `tr` | 0.571 | 0.163 | 0.626 | 0.551 | 0.758 | 0.478 |

`tr` is the Turkish wording above. `en` and `en_text_first` are English wordings with yes/no answers; all three are in `prompts.py`.

All adapters are trained and scored with `tr`. It was chosen with Qwen3-0.6B, for which it has the highest zero-shot ROC-AUC, before the larger models were trained. Zero-shot Qwen3-1.7B and gpt-oss-20b rank better with the English wordings, so their zero-shot rows in the result tables use `en_text_first`. Each base model is therefore shown with its best wording of the three.

## Training

There is one config per model in `configs/`. `modal_app.py` runs them with `halo launch sft` inside the image `public.ecr.aws/whitecircle/halo:hopper-1.3.0rc1`, on one GPU requested from Modal as `H100`.

All three use LoRA with rank 16, alpha 32 and dropout 0.05, 2 epochs, learning rate 1e-4 with cosine decay and 5% warmup, and seed 42. The loss covers the answer turn only.

- Qwen3-0.6B and Qwen3-1.7B: LoRA on all attention and MLP projections. Halo packs the 29,986 rows into 1,814 blocks of 2,048 tokens and trains on 4 blocks per step, which gives 908 steps.
- gpt-oss-20b: LoRA on the attention projections only, because its expert weights are fused tensors that stock PEFT cannot adapt. The pretrained attention sinks stay live and frozen (`reset_sinks: false`) so that the adapter fits the stock model, which on this image means eager attention and padded batches instead of packing: 64 rows per step, 938 steps. The base checkpoint is the BF16 mirror `unsloth/gpt-oss-20b-BF16`, as the Halo cookbook asks for.

| | Qwen3-0.6B | Qwen3-1.7B | gpt-oss-20b |
|---|---:|---:|---:|
| Trainable parameters | 10,092,544 (1.67%) | 17,432,576 (1.00%) | 7,962,624 (0.04%) |
| Training loop | 261 s | 293 s | 863 s |
| Validation loss after epoch 1 and 2 | 0.0396, 0.0366 | 0.0341, 0.0334 | 0.0359, 0.0359 |

The full training logs are in `results/`, one `*_halo_run.log` per model.

There was no hyperparameter search. The learning rate, scheduler and target modules of the Qwen3 runs are those of Halo's shipped Qwen3 LoRA recipe (`examples/sft/qwen3/qwen3-4b-ultrachat-lora.yaml`); rank, epochs and batch size were set once. All values were fixed before the test set was scored, and the final checkpoint is used. All three models are reported; their order on the validation split is the same as on the test split.

Each Qwen3 model was trained twice from scratch. Both runs wrote the same adapter file, byte for byte, and scoring the same adapter twice gave the same score file, so on this image and GPU type those results are reproducible exactly. gpt-oss-20b was trained once. SHA-256 of `adapter_model.safetensors`:

```
7050546b18b0d5dc4ea16ec84a7a1174b6c68fe0eaba01cdb8b08f613fff7e20  runs/sft-qwen3-0.6b-lora/checkpoint-908
7f75aeeeba3e559318d102f259c42c049d79a4bc4193f4f0eda1ae29daad7609  runs/sft-qwen3-1.7b-lora/checkpoint-908
c38406431a0e1d51c0de195972f7baa8b5c95a4ed16272c4c5b7e27499619ef0  runs/sft-gpt-oss-20b-lora/checkpoint-938
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

# for each Qwen3 model, here Qwen3-1.7B; gpt-oss-20b also takes --chat-template, see reproduce.sh:
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

Run from a fresh clone with the two Qwen3 models, `reproduce.sh` took 28 minutes and left every file in `results/` unchanged, byte for byte. The gpt-oss-20b steps were added to the script afterwards and were run one by one, not through the script. The first use of the Halo image on Modal took another 6 minutes 18 seconds to import, once. A Qwen3 training command takes 5 to 6 minutes end to end and a scoring command 40 to 60 seconds; the gpt-oss-20b training command took 15 minutes 53 seconds, after a first download of the 42 GB checkpoint.

The trained adapters stay on the Modal volume. To copy one to `adapter/`:

```bash
mkdir -p adapter
uv run modal volume get halo-qwen3-tr-offensive runs/sft-qwen3-1.7b-lora/checkpoint-908/adapter_model.safetensors adapter/
uv run modal volume get halo-qwen3-tr-offensive runs/sft-qwen3-1.7b-lora/checkpoint-908/adapter_config.json adapter/
```

Local dependencies are pinned to the versions inside the Halo 1.3.0rc1 image (torch 2.11.0, transformers 5.20.2, peft 0.18.1, datasets 4.8.5), so the CPU checks and the GPU runs use the same library code.

## Use the adapter

Two adapters are on the Hugging Face Hub: [`tatlilimon/qwen3-1.7b-tr-offensive-lora`](https://huggingface.co/tatlilimon/qwen3-1.7b-tr-offensive-lora) and [`tatlilimon/gpt-oss-20b-tr-offensive-lora`](https://huggingface.co/tatlilimon/gpt-oss-20b-tr-offensive-lora). Their model cards are in `model_cards/`. The example below is for the Qwen3-1.7B adapter; the gpt-oss-20b card has its own, which builds the prompt in that model's format.

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = (
    "Aşağıdaki metin saldırgan mı? Saldırgan metin; hakaret, küfür ya da bir kişiye veya gruba yönelik "
    "açık ya da örtülü saldırı içerir. Yalnızca tek kelimeyle cevap ver: evet ya da hayır.\n\nMetin: {text}"
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B"), "tatlilimon/qwen3-1.7b-tr-offensive-lora"
)


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

The prompt has to match the training prompt exactly, and `enable_thinking=False` is required. This example was run on a CPU machine with 20 GB of memory; in fp32 the 1.7B model alone takes about 8 GB. To score a file of tweets the way the tables were made, copy the adapter to `adapter/` as shown above and use `score.py --model Qwen/Qwen3-1.7B --adapter adapter`.

## Limitations and known failures

- Every adapter misses many offensive tweets at its default decision: recall on the test split is 0.648 for gpt-oss-20b, 0.578 for Qwen3-1.7B and 0.503 for Qwen3-0.6B. The tuned thresholds in `results/metrics.json` trade precision for recall.
- From reading a sample of the errors of the two Qwen3 adapters: missed tweets are mostly insults without explicit profanity, sarcasm and political accusations. False alarms are mostly heated political or religious arguments.
- The 0.6B adapter does not beat the TF-IDF baseline, and the 1.7B adapter beats it clearly only at the default decision. Use the baseline if you need a cheap classifier for this data.
- The gpt-oss-20b adapter needs a GPU with about 45 GB of memory and was trained once, so its run-to-run variation was not checked.
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
- Base models: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) and [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) through its BF16 mirror [unsloth/gpt-oss-20b-BF16](https://huggingface.co/unsloth/gpt-oss-20b-BF16), all Apache-2.0.
- Dataset: OffensEval-TR 2020, CC BY 2.0. The score files in `results/` hold labels and scores but no tweet text.
- Training framework: [Halo](https://github.com/whitecircle/halo) 1.3.0rc1, under the Halo License.

Çağrı Çöltekin. 2020. A Corpus of Turkish Offensive Language on Social Media. In Proceedings of the 12th International Conference on Language Resources and Evaluation (LREC).

Marcos Zampieri, Preslav Nakov, Sara Rosenthal, Pepa Atanasova, Georgi Karadzhov, Hamdy Mubarak, Leon Derczynski, Zeses Pitenis and Çağrı Çöltekin. 2020. SemEval-2020 Task 12: Multilingual Offensive Language Identification in Social Media (OffensEval 2020). arXiv:2006.07235.
