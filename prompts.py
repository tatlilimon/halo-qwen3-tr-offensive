"""Prompt wordings and answer words shared by data preparation and scoring.

Training rows and scoring prompts have to render to the same text, so both
sides build their messages here.

There are three wordings so that the zero-shot base model can be measured with
each of them. SELECTED is the wording with the best zero-shot ROC-AUC on the
validation split; the adapter is trained and scored with it.
"""

from typing import NamedTuple


class Prompt(NamedTuple):
    template: str
    # (not offensive, offensive): the index is the dataset label.
    answers: tuple[str, str]


# The definition of "offensive" follows the label description in the dataset card.
PROMPTS = {
    "en": Prompt(
        "Is the following Turkish text offensive? Offensive means it contains offensive language, profanity, "
        "or a direct or veiled offense against a person or group. Answer with exactly one word: yes or no.\n\n"
        "Text: {text}",
        ("no", "yes"),
    ),
    "en_text_first": Prompt(
        "Text: {text}\n\n"
        "Is the Turkish text above offensive? Offensive means it contains offensive language, profanity, "
        "or a direct or veiled offense against a person or group. Answer with exactly one word: yes or no.",
        ("no", "yes"),
    ),
    "tr": Prompt(
        "Aşağıdaki metin saldırgan mı? Saldırgan metin; hakaret, küfür ya da bir kişiye veya gruba yönelik "
        "açık ya da örtülü saldırı içerir. Yalnızca tek kelimeyle cevap ver: evet ya da hayır.\n\n"
        "Metin: {text}",
        ("hayır", "evet"),
    ),
}
SELECTED = "tr"


def user_message(text: str, prompt: str = SELECTED) -> dict[str, str]:
    return {"role": "user", "content": PROMPTS[prompt].template.format(text=text)}
