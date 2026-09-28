"""Improved question-only front gate; never uses answer or dataset labels."""

from __future__ import annotations

import re
from typing import Any

from .native_router import _route, route_question as prior_route


COLOR = re.compile(r"\bcolou?r\b", re.I)
PRESENCE_MODIFIER = re.compile(
    r"\b(?:not|without|small|large|brown|white|black|red|blue|green|"
    r"yellow|orange|pink|gray|grey|wet|full|open|closed|safety)\b", re.I
)


def route_question(question: str) -> dict[str, Any]:
    q = " ".join(question.strip().split())
    lower = q.lower()
    if not q:
        return prior_route(q)

    # Color queries need not begin with 'what/which': 'The boots are what
    # color?' and 'Of what color...' are native color questions too.
    if COLOR.search(q) and not lower.startswith(("is there", "are there")):
        target = re.sub(r"\b(?:what|which|of|the|is|are|does|do|has|have|color|colour)\b",
                        " ", q, flags=re.I)
        target = " ".join(target.strip(" ?.!").split()) or q
        qualified = any(token in lower for token in (" that ", " left ", " right ", " under ", " near "))
        return _route("color", "color", "qualified_target" if qualified else "direct_target",
                      target, "target_grounding_required" if qualified else "", True)

    # 'How many proof' asks for a number read from a label, not a count of
    # distinct objects.  More generally, a mass/measurement unit without a
    # countable object should not be forced into the instance-count branch.
    if re.match(r"^how many\s+(?:proof|degrees|percent|percentage)\b", lower):
        return _route("open_text", "ambiguous", "measurement_not_object_count",
                      q, "needs_image_or_text_evidence", False)

    if lower.startswith(("is there", "are there")):
        result = prior_route(q)
        target = result["target_phrase"]
        operator = "and" if re.search(r"\bboth\b", target, re.I) else "or" if re.search(
            r"\b(?:or|either)\b", target, re.I) else "single"
        target = re.sub(r"^(?:both|either|any|a|an)\s+", "", target, flags=re.I)
        operands = [part.strip() for part in re.split(r"\b(?:and|or)\b", target, flags=re.I)]
        operands = [part for part in operands if part]
        modifier = bool(PRESENCE_MODIFIER.search(target) or " that " in lower)
        # A two-target expression can be evaluated from two separately
        # grounded detections.  Attribute/negation modifiers require an
        # additional verifier, so routing is recorded but changing is barred.
        result.update({"logic_operator": operator, "logic_operands": operands,
                       "parse_status": "qualified_logical_expression" if modifier else
                                       "plain_logical_expression" if len(operands) > 1 else "single_target",
                       "constraint": "attribute_or_negation" if modifier else "",
                       "tool_eligible": not modifier and 1 <= len(operands) <= 2})
        return result

    return prior_route(q)
