---
base_model: Qwen/Qwen3-1.7B
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

# Qwen3-1.7B LoRA for Turkish offensive-language detection

A LoRA adapter for [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) that answers `evet` (yes) or `hayır` (no) to the question whether a Turkish tweet is offensive. It was trained with [Halo](https://github.com/whitecircle/halo) on OffensEval-TR 2020.

The code, the Halo config, the training log and the score files behind every number are in [tatlilimon/halo-qwen3-tr-offensive](https://github.com/tatlilimon/halo-qwen3-tr-offensive).

## Results

Official OffensEval-TR 2020 test set, 3,528 tweets, 20.3% offensive:

| System | Macro-F1 [95% interval] | Accuracy | Precision | Recall | ROC-AUC | Macro-F1, tuned threshold |
|---|---:|---:|---:|---:|---:|---:|
| Constant "not offensive" | 0.444 [0.439, 0.448] | 0.797 | 0.000 | 0.000 | n/a | n/a |
| TF-IDF character n-grams + logistic regression | 0.708 [0.688, 0.730] | 0.859 | 0.898 | 0.345 | 0.873 | 0.770 [0.752, 0.787] |
| Qwen3-1.7B zero-shot (best of three prompt wordings) | 0.389 [0.373, 0.404] | 0.389 | 0.240 | 0.929 | 0.733 | 0.640 [0.621, 0.658] |
| Qwen3-1.7B + this adapter | 0.788 [0.770, 0.806] | 0.875 | 0.750 | 0.578 | 0.897 | 0.792 [0.775, 0.807] |

Macro-F1 is the official metric of the task. Precision and recall are for the offensive class. All columns except the last use the default decision, which for the model is greedy decoding between its two answers; the last column uses a decision threshold tuned on the validation split. Intervals are bootstrap intervals over the test tweets.

For context, 46 teams submitted to this track of SemEval-2020 Task 12. The best system scored 0.8258 and the majority baseline 0.4435 (Zampieri et al., 2020, Table 11). This adapter's 0.7885 falls between the 6th (0.7933) and the 7th (0.7859) submission.

With thresholds tuned on validation the adapter's lead over the TF-IDF baseline is small (0.792 against 0.770) and the intervals overlap.

## Use

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

The prompt has to match the training prompt exactly, and `enable_thinking=False` is required. In fp32 the model takes about 8 GB of memory. In English the prompt reads: "Is the following text offensive? An offensive text contains insults, profanity, or an open or veiled attack on a person or group. Answer with one word only: yes or no."

## Training

- Data: 29,986 training tweets of OffensEval-TR 2020 (Çöltekin, 2020), read from [`stefan-it/offenseval2020_tr`](https://huggingface.co/datasets/stefan-it/offenseval2020_tr), 19.3% offensive. User mentions are replaced by `@USER` in the dataset.
- Halo 1.3.0rc1, `halo launch sft`, on one GPU requested from Modal as `H100`.
- LoRA with rank 16, alpha 32 and dropout 0.05 on all attention and MLP projections: 17,432,576 trainable parameters, 1.00% of the model.
- 2 epochs, 908 steps, learning rate 1e-4 with cosine decay and 5% warmup, seed 42. The training loop took 293 seconds.
- No hyperparameter search. Training twice from scratch gave the same adapter file, byte for byte. SHA-256 of `adapter_model.safetensors`: `7f75aeeeba3e559318d102f259c42c049d79a4bc4193f4f0eda1ae29daad7609`.

## Limitations

- The adapter misses four in ten offensive tweets at its default decision: recall is 0.578 on the test split, at precision 0.750.
- From reading a sample of its errors: missed tweets are mostly insults without explicit profanity, sarcasm and political accusations. False alarms are mostly heated political or religious arguments.
- It beats a TF-IDF plus logistic regression baseline clearly only at the default decision.
- One seed, no hyperparameter search. The intervals describe the sampling noise of the test split only.
- The reported scores were computed in bf16 on a GPU. Other precisions were not measured for this adapter, and tweets near the threshold may flip.
- Most tweets in the dataset were labeled by a single volunteer annotator, and agreement on a doubly annotated sample was Cohen's kappa 0.761 (Zampieri et al., 2020, section 3). The adapter inherits the label noise, topics and biases of these tweets and is not a moderation system on its own.

## License and references

The adapter is released under Apache-2.0, the license of the base model. The training data is OffensEval-TR 2020, CC BY 2.0.

Çağrı Çöltekin. 2020. A Corpus of Turkish Offensive Language on Social Media. In Proceedings of the 12th International Conference on Language Resources and Evaluation (LREC).

Marcos Zampieri, Preslav Nakov, Sara Rosenthal, Pepa Atanasova, Georgi Karadzhov, Hamdy Mubarak, Leon Derczynski, Zeses Pitenis and Çağrı Çöltekin. 2020. SemEval-2020 Task 12: Multilingual Offensive Language Identification in Social Media (OffensEval 2020). arXiv:2006.07235.
