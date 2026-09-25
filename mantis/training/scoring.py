"""
Answer scoring shared by Stage 3 rewards and the evaluation harness.

A lexical proxy for correctness that does not credit negated or abstained
answers: a response counts as correct when the normalized reference appears
in it as a whole phrase with no negation word in the response, or when the
word-level F1 between response and reference reaches the threshold.
Abstentions are never scored as correct or wrong; callers handle them
explicitly.
"""

import re
import string
from typing import Optional

from mantis.inference.prompting import word_f1

NEGATIONS = {"not", "n't", "never", "no", "none", "neither", "nor", "false", "incorrect", "wrong", "isn't",
             "aren't", "wasn't", "weren't", "doesn't", "don't", "didn't", "cannot", "can't", "won't"}
ARTICLES = {"a", "an", "the"}


def normalize_answer(text: str) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace (SQuAD normalization)."""
    text = text.lower()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    words = [w for w in text.split() if w not in ARTICLES]
    return " ".join(words)


def has_negation(text: str) -> bool:
    words = set(re.findall(r"[a-z']+", text.lower()))
    return bool(words & NEGATIONS)


def answer_correct(response: str, reference: str, f1_threshold: float = 0.5) -> bool:
    """
    Lexical correctness of `response` against `reference`.

    True when the normalized reference appears as a whole phrase in a
    response that contains no negation word, or when word F1 >= threshold.
    "Paris is NOT the capital of France" is wrong for reference "Paris".
    """
    response_n, reference_n = normalize_answer(response), normalize_answer(reference)
    if not reference_n:
        return False
    # A negated assertion must not become correct through lexical overlap.
    if has_negation(response):
        return False
    phrase = re.search(rf"(?<!\S){re.escape(reference_n)}(?!\S)", response_n) is not None
    if phrase:
        return True
    return word_f1(response_n, reference_n) >= f1_threshold


def score_answer(response: str, reference: str, abstained: bool, f1_threshold: float = 0.5) -> Optional[bool]:
    """True = correct, False = wrong, None = abstained."""
    if abstained:
        return None
    return answer_correct(response, reference, f1_threshold)
