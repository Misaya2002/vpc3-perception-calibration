"""Conservative, label-free object phrases for open-vocabulary counters."""

from __future__ import annotations

import re


_LEADING_DETERMINERS = re.compile(r"^(?:(?:of)\s+)?(?:(?:the|a|an|these|those|some)\s+)+", re.I)
_AUXILIARY = re.compile(r"\b(?:are|is|was|were|do|does|did|can|could|will|would|have|has|had)\b", re.I)
_TRAILING_QUERY = re.compile(
    r"\s+(?:in|from|of)\s+(?:this|the|a|an)\s+(?:picture|photo|image|scene)\s*$", re.I
)
_TRAILING_VIEW = re.compile(r"\s+(?:do|does|did)\s+you\s+see\s*$", re.I)
_TRAILING_THERE = re.compile(r"\s+(?:are|is|was|were)\s+there\s*$", re.I)
_TOKEN = re.compile(r"[a-z]+(?:[-'][a-z]+)*|\d+", re.I)
_NON_OBJECT_ONLY = {
    "this", "that", "it", "they", "we", "you", "i", "there", "here",
    "many", "much", "number", "amount", "proof", "time", "weather",
}


def count_query_from_question(question: str) -> str | None:
    """Return a compact object phrase or abstain; never consult route/labels.

    This is intentionally not a general semantic parser. It supports the common
    ``How many ...?`` form and removes leading determiners plus common question
    scaffolding. Predicates that help define the counted subset (e.g. ``wearing
    glasses`` or ``full of beer``) are retained as phrase modifiers.
    """
    text = " ".join(str(question or "").strip().split())
    match = re.match(r"^how\s+many\s+(.+?)\s*[?!.]*$", text, re.I)
    if not match:
        return None

    phrase = match.group(1).strip(" \t?!.:,;")
    phrase = _LEADING_DETERMINERS.sub("", phrase)
    phrase = _TRAILING_THERE.sub("", phrase)
    phrase = _TRAILING_VIEW.sub("", phrase)
    phrase = re.sub(r"\b(?:can|could)\s+be\s+seen\b", "", phrase, flags=re.I)
    phrase = re.sub(r"\b(?:have|has|had)\b", "with", phrase, flags=re.I)
    # Remove auxiliary verbs while keeping useful subset descriptors:
    # "people are wearing glasses" -> "people wearing glasses".
    phrase = _AUXILIARY.sub(" ", phrase)
    phrase = re.sub(r"\b(?:you|we|i)\b", " ", phrase, flags=re.I)
    phrase = _TRAILING_QUERY.sub("", phrase)
    phrase = re.sub(r"\s+", " ", phrase).strip(" \t?!.:,;")
    tokens = _TOKEN.findall(phrase)
    if not tokens or len(tokens) > 14:
        return None
    if len(tokens) == 1 and tokens[0].casefold() in _NON_OBJECT_ONLY:
        return None
    if any(token.casefold() in {"equal", "plus", "minus", "weather"} for token in tokens):
        return None
    return phrase
