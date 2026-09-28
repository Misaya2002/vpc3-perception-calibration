"""Question-only answer-space and evidence routing for native VPC.

This parser deliberately does not accept a dataset category, a model answer, or
ground truth.  Unclear questions are marked ambiguous rather than coerced into
yes/no or an unsupported tool branch.
"""

from __future__ import annotations

import re
from typing import Any


SPATIAL = re.compile(
    r"\b(?:to the left of|to the right of|on the left of|on the right of|"
    r"left of|right of|in front of|behind|above|below|under|beneath|"
    r"next to|beside|inside|outside|on top of|near)\b", re.I
)
OCR_EXPLICIT = re.compile(
    r"\b(?:read|reads|written|printed|spelled|spell|word|letter|letters|"
    r"text|caption|title|logo|brand|name of the book|name on|date shown|"
    r"number on|number is the .*page|sign say|sign says|label say|"
    r"label says|paragraph|photographer)\b", re.I
)
COMPLEX_PRESENCE = re.compile(r"\b(?:both|or|not|neither|either|without|except)\b", re.I)
COUNT_CONSTRAINT = re.compile(
    r"\b(?:who|which|that|whose|wearing|looking|holding|with|without|"
    r"mostly|full of|not|white|black|red|blue|green|yellow|left|right)\b", re.I
)


def route_question(question: str) -> dict[str, Any]:
    """Return a label-free route and a conservative evidence-eligibility flag."""
    q = " ".join(question.strip().split())
    lower = q.lower()
    if not q:
        return _route("unknown", "ambiguous", "empty_question", "", "", False)

    if re.match(r"^how many\b", lower):
        body = re.sub(r"^how many\s+", "", q, flags=re.I).rstrip(" ?.")
        body = re.sub(r"\b(?:are|is) there\b.*$", "", body, flags=re.I).strip()
        constrained = bool(COUNT_CONSTRAINT.search(body))
        return _route("integer", "count", "qualified_count" if constrained else "simple_count",
                      body, "attribute_filter_required" if constrained else "", not constrained)

    if ("color" in lower or "colour" in lower) and re.match(r"^(?:what|which)\b", lower):
        match = re.search(r"\b(?:color|colour)\s+(?:of|is|does)\s+(.+)", q, re.I)
        target = match.group(1).rstrip(" ?.") if match else q
        nested = len(SPATIAL.findall(q)) > 0 or " that " in lower
        return _route("color", "color", "qualified_target" if nested else "direct_target",
                      target, "target_grounding_required" if nested else "", True)

    if re.match(r"^(?:is there|are there)\b", lower):
        match = re.match(r"^(?:is there|are there)\s+(.+?)(?:\s+in (?:the|this)\s+"
                         r"(?:image|picture|photo|photograph))?\??$", q, re.I)
        target = match.group(1).rstrip(" ?.") if match else q
        complex_logic = bool(COMPLEX_PRESENCE.search(target))
        return _route("boolean", "object_presence",
                      "logical_expression" if complex_logic else "single_target",
                      target, "logical_composition_required" if complex_logic else "",
                      not complex_logic)

    if OCR_EXPLICIT.search(q):
        return _route("open_text", "ocr", "explicit_text_cue", q, "", True)

    relations = SPATIAL.findall(q)
    if relations:
        complex_relation = len(relations) > 1
        return _route("entity", "spatial_relation",
                      "nested_relation" if complex_relation else "direct_relation",
                      q, "multiple_relations" if complex_relation else relations[0].lower(),
                      not complex_relation)

    # OCR questions need not contain words like 'text' or 'read'.  An open
    # question without an explicit cue remains unresolved for later multimodal
    # inspection; it is never silently labelled as spatial or binary.
    return _route("open_text", "ambiguous", "no_explicit_cue", q, "", False)


def _route(answer_type: str, task: str, parse_status: str, target: str,
           constraint: str, tool_eligible: bool) -> dict[str, Any]:
    return {
        "answer_type": answer_type,
        "perceptual_task": task,
        "parse_status": parse_status,
        "target_phrase": target,
        "constraint": constraint,
        "tool_eligible": tool_eligible,
        "source": "question_text_only",
    }
