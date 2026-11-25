---
base_model: unsloth/gpt-oss-20b-BF16
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
language:
- tr
datasets:
- stefan-it/offenseval2020_tr
tags:
- lora
- halo
- turkish
- offensive-language-detection
---

# gpt-oss-20b LoRA for Turkish offensive-language detection

A LoRA adapter for gpt-oss-20b that answers `evet` (yes) or `hayır` (no) to the question whether a Turkish tweet is offensive. It was trained with [Halo](https://github.com/whitecircle/halo) on OffensEval-TR 2020, on the BF16 mirror [unsloth/gpt-oss-20b-BF16](https://huggingface.co/unsloth/gpt-oss-20b-BF16) of [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b).

The code, the Halo config, the training log and the score files behind every number are in [tatlilimon/halo-qwen3-tr-offensive](https://github.com/tatlilimon/halo-qwen3-tr-offensive).

## Results

Official OffensEval-TR 2020 test set, 3,528 tweets, 20.3% offensive:

| System | Macro-F1 [95% interval] | Accuracy | Precision | Recall | ROC-AUC | Macro-F1, tuned threshold |
|---|---:|---:|---:|---:|---:|---:|
| Constant "not offensive" | 0.444 [0.439, 0.448] | 0.797 | 0.000 | 0.000 | n/a | n/a |
| TF-IDF character n-grams + logistic regression | 0.708 [0.688, 0.730] | 0.859 | 0.898 | 0.345 | 0.873 | 0.770 [0.752, 0.787] |
| gpt-oss-20b zero-shot (best of three prompt wordings) | 0.539 [0.520, 0.558] | 0.816 | 0.923 | 0.101 | 0.854 | 0.750 [0.732, 0.768] |
| gpt-oss-20b + this adapter | 0.825 [0.808, 0.841] | 0.895 | 0.796 | 0.648 | 0.930 | 0.827 [0.811, 0.843] |

Macro-F1 is the official metric of the task. Precision and recall are for the offensive class. All columns except the last use the default decision, which for the model is greedy decoding between its two answers; the last column uses a decision threshold tuned on the validation split. Intervals are bootstrap intervals over the test tweets.

For context, 46 teams submitted to this track of SemEval-2020 Task 12. The best system scored 0.8258, the second 0.8167 and the majority baseline 0.4435 (Zampieri et al., 2020, Table 11). This adapter's 0.8250 falls between the first and the second submission.

## Use

The adapter was trained with Halo's SFT chat template for this model family, which puts the answer in the final channel with no reasoning before it. The example builds that prompt directly.

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

QUESTION = (
    "Aşağıdaki metin saldırgan mı? Saldırgan metin; hakaret, küfür ya da bir kişiye veya gruba yönelik "
    "açık ya da örtülü saldırı içerir. Yalnızca tek kelimeyle cevap ver: evet ya da hayır.\n\nMetin: {text}"
)
PROMPT = (
    "<|start|>user<|message|>{question}<|end|>"
    "<|start|>assistant<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"
)

tokenizer = AutoTokenizer.from_pretrained("unsloth/gpt-oss-20b-BF16")
model = AutoModelForCausalLM.from_pretrained("unsloth/gpt-oss-20b-BF16", dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(model, "tatlilimon/gpt-oss-20b-tr-offensive-lora")


def answer(text: str) -> str:
    prompt = PROMPT.format(question=QUESTION.format(text=text))
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


print(answer("Günaydın, bugün hava çok güzel."))    # hayır
print(answer("Sen tam bir aptalsın, sus artık."))    # evet
```

The prompt has to match the training prompt exactly. The model needs a GPU with about 45 GB of memory in bf16. In English the question reads: "Is the following text offensive? An offensive text contains insults, profanity, or an open or veiled attack on a person or group. Answer with one word only: yes or no."

## Training

- Data: 29,986 training tweets of OffensEval-TR 2020 (Çöltekin, 2020), read from [`stefan-it/offenseval2020_tr`](https://huggingface.co/datasets/stefan-it/offenseval2020_tr), 19.3% offensive. User mentions are replaced by `@USER` in the dataset.
- Halo 1.3.0rc1, `halo launch sft`, on one GPU requested from Modal as `H100`.
- LoRA with rank 16, alpha 32 and dropout 0.05 on the attention projections: 7,962,624 trainable parameters, 0.04% of the model. The expert weights are not adapted.
- The pretrained attention sinks were kept live and frozen, so the adapter fits the stock model.
- 2 epochs, 938 steps of 64 tweets, learning rate 1e-4 with cosine decay and 5% warmup, seed 42. The training loop took 863 seconds.
- No hyperparameter search, one training run. SHA-256 of `adapter_model.safetensors`: `c38406431a0e1d51c0de195972f7baa8b5c95a4ed16272c4c5b7e27499619ef0`.

## Limitations

- The adapter misses about a third of the offensive tweets at its default decision: recall is 0.648 on the test split, at precision 0.796.
- One seed, one run, no hyperparameter search. The intervals describe the sampling noise of the test split only.
- The reported scores were computed in bf16 on a GPU. Other precisions were not measured, and tweets near the threshold may flip.
- Most tweets in the dataset were labeled by a single volunteer annotator, and agreement on a doubly annotated sample was Cohen's kappa 0.761 (Zampieri et al., 2020, section 3). The adapter inherits the label noise, topics and biases of these tweets and is not a moderation system on its own.

## License and references

The adapter is released under Apache-2.0, the license of the base model. The training data is OffensEval-TR 2020, CC BY 2.0.

Çağrı Çöltekin. 2020. A Corpus of Turkish Offensive Language on Social Media. In Proceedings of the 12th International Conference on Language Resources and Evaluation (LREC).

Marcos Zampieri, Preslav Nakov, Sara Rosenthal, Pepa Atanasova, Georgi Karadzhov, Hamdy Mubarak, Leon Derczynski, Zeses Pitenis and Çağrı Çöltekin. 2020. SemEval-2020 Task 12: Multilingual Offensive Language Identification in Social Media (OffensEval 2020). arXiv:2006.07235.
