"""Label-free expert proposals in the question's native answer space.

Every proposal is optional. A specialist score is retained as evidence, never
treated as a calibrated probability or as authority to overwrite Qwen.
"""

from __future__ import annotations

import re
import math
from typing import Any

from .evidence import labels_match
from .native_count_query import count_query_from_question


PALETTE = {"black", "white", "gray", "red", "orange", "yellow", "green",
           "blue", "purple", "pink", "brown"}


def normalize_answer(value: Any, answer_type: str) -> str | None:
    text = " ".join(str(value or "").strip().split())
    if not text:
        return None
    low = text.lower()
    if answer_type == "boolean":
        return low if low in {"yes", "no"} else None
    if answer_type == "integer":
        return str(int(low)) if re.fullmatch(r"\d{1,3}", low) else None
    if answer_type == "color":
        low = low.replace("grey", "gray")
        return low if low in PALETTE else None
    if answer_type in {"entity", "open_text"}:
        if low in {"yes", "no"}:
            return None
        return text if len(text) <= 120 and not re.search(r"[\n\r]", text) else None
    return None


def proposal(expert: str, answer: Any, answer_type: str, score: float,
             semantics: str, details: dict[str, Any]) -> dict[str, Any] | None:
    normalized = normalize_answer(answer, answer_type)
    if normalized is None or not 0.0 <= score <= 1.0:
        return None
    return {"expert": expert, "answer": normalized, "answer_type": answer_type,
            "score": round(score, 6), "score_semantics": semantics,
            "details": details}


def _objects(evidence: Any) -> list[dict[str, Any]]:
    return [row for row in (evidence or {}).get("objects", []) if isinstance(row, dict)]


def presence_proposals(route: dict[str, Any], raw_answer: str,
                       evidence: Any) -> list[dict[str, Any]]:
    if (route.get("perceptual_task") != "object_presence" or
            route.get("answer_type") != "boolean" or
            not route.get("tool_eligible") or raw_answer.lower() != "no"):
        return []
    operands = route.get("logic_operands") or [route.get("target_phrase", "")]
    if not 1 <= len(operands) <= 2 or not all(operands):
        return []
    matched_objects = []
    scores = []
    for target in operands:
        matches = [row for row in _objects(evidence)
                   if labels_match(target, row.get("label"))]
        best_match = max(matches, key=lambda row: float(row.get("score", 0.0)), default=None)
        scores.append(float(best_match.get("score", 0.0)) if best_match else 0.0)
        if best_match:
            matched_objects.append({
                "target": target, "label": best_match.get("label"),
                "score": float(best_match.get("score", 0.0)),
                "bbox": best_match.get("bbox"),
            })
    operator = route.get("logic_operator", "single")
    if operator == "and" and len(scores) == 2:
        supported, score = all(x >= 0.45 for x in scores), min(scores)
    elif operator == "or" and len(scores) == 2:
        supported, score = any(x >= 0.45 for x in scores), max(scores)
    elif operator == "single" and len(scores) == 1:
        supported, score = scores[0] >= 0.45, scores[0]
    else:
        return []
    if not supported:
        return []
    result = proposal("sam3_presence", "yes", "boolean", score,
                      "SAM3 matched-object score, not probability",
                      {"operands": operands, "scores": scores,
                       "matched_objects": matched_objects, "logic": operator})
    return [result] if result else []


def color_proposals(route: dict[str, Any], raw_answer: str,
                    evidence: Any) -> list[dict[str, Any]]:
    if (route.get("perceptual_task") != "color" or
            route.get("answer_type") != "color" or not route.get("tool_eligible") or
            normalize_answer(raw_answer, "color") is None):
        return []
    rows = [row for row in (evidence or {}).get("colors", []) if isinstance(row, dict)]
    if not rows:
        return []
    top = max(rows, key=lambda row: float(row.get("confidence", 0.0)))
    value = normalize_answer(top.get("color"), "color")
    score = float(top.get("confidence", 0.0))
    if value is None or value == normalize_answer(raw_answer, "color") or score < 0.45:
        return []
    result = proposal("sam3_mask_color", value, "color", score,
                      "mask color fraction, not correctness probability",
                      {"target": top.get("target"), "bbox": top.get("bbox"),
                       "crop_box": top.get("crop_box"),
                       "mask_observation_available": bool(top.get("mask_path"))})
    return [result] if result else []


def siglip2_color_proposals(route: dict[str, Any], raw_answer: str,
                            evidence: Any, question: str) -> list[dict[str, Any]]:
    if (route.get("perceptual_task") != "color" or route.get("answer_type") != "color" or
            not isinstance(evidence, dict) or evidence.get("question") != question):
        return []
    raw = normalize_answer(raw_answer, "color")
    result = []
    for row in evidence.get("candidates", []):
        if not isinstance(row, dict):
            continue
        answer = normalize_answer(row.get("answer"), "color")
        score = float(row.get("score", 0.0))
        margin = float(row.get("margin", 0.0))
        if answer is None or answer == raw or score < 0.25 or margin < 0.02:
            continue
        candidate = proposal("siglip2_sam3_crop_color", answer, "color", score,
                             "SigLIP2 zero-shot color ranking, not correctness probability",
                             {"bbox": row.get("bbox"), "crop_box": row.get("crop_box"),
                              "margin": margin, "sam3_score": row.get("sam3_score"),
                              "target": row.get("target")})
        if candidate and not any(item["answer"] == answer for item in result):
            result.append(candidate)
    return result


def count_anything_proposals(route: dict[str, Any], question: str,
                             raw_answer: str, evidence: Any) -> list[dict[str, Any]]:
    if route.get("answer_type") != "integer" or route.get("perceptual_task") != "count":
        return []
    if not isinstance(evidence, dict) or evidence.get("question") != question:
        return []
    if evidence.get("status") != "complete" or evidence.get("query_matches_question_target") is not True:
        return []
    if str(evidence.get("query", "")).strip().lower() != str(route.get("target_phrase", "")).strip().lower():
        return []
    value = normalize_answer(evidence.get("count"), "integer")
    if value is None or value == normalize_answer(raw_answer, "integer"):
        return []
    points = evidence.get("points")
    if not isinstance(points, list) or len(points) != int(value):
        return []
    score = float(evidence.get("score", 0.0))
    result = proposal("count_anything", value, "integer", score,
                      "instance-point score, not correctness probability",
                      {"query": evidence.get("query"), "point_count": len(points),
                       "points": points})
    return [result] if result else []


def countgd_proposals(route: dict[str, Any], question: dict[str, Any],
                      raw_answer: str, evidence: Any) -> list[dict[str, Any]]:
    """Turn released CountGD detections into a native integer candidate.

    The mean detector score is evidence strength, not a probability that the
    proposed count is correct. The query is derived from the frozen question.
    """
    if route.get("answer_type") != "integer" or route.get("perceptual_task") != "count":
        return []
    if not isinstance(evidence, dict):
        return []
    target = count_query_from_question(str(question.get("question", "")))
    query = " ".join(str(evidence.get("query", "")).split())
    if (evidence.get("status") != "complete" or not target or
            evidence.get("question") != question.get("question") or
            evidence.get("image") != question.get("image") or
            query.casefold() != target.casefold() or
            evidence.get("query_source") != "question_only_conservative_v1"):
        return []
    points = evidence.get("points")
    count = normalize_answer(evidence.get("count"), "integer")
    if not isinstance(points, list) or count is None or len(points) != int(count):
        return []
    raw = normalize_answer(raw_answer, "integer")
    if count == raw:
        return []
    candidate = proposal(
        "countgd", count, "integer", float(evidence.get("score", 0.0)),
        "mean retained CountGD detection score; not count correctness probability",
        {"query": query, "point_count": len(points), "points": points,
         "confidence_threshold": evidence.get("confidence_threshold")},
    )
    return [candidate] if candidate else []


def ocr_proposals(route: dict[str, Any], raw_answer: str,
                  evidence: Any) -> list[dict[str, Any]]:
    if route.get("answer_type") != "open_text" or route.get("perceptual_task") != "ocr":
        return []
    rows = [row for row in (evidence or {}).get("ocr", [])
            if isinstance(row, dict) and normalize_answer(row.get("text"), "open_text")
            and float(row.get("confidence", 0.0)) >= 0.90]
    # A single high-confidence line is a narrowly supported answer candidate.
    # Multiple lines require question-conditioned retrieval, not top-score guesswork.
    if len(rows) != 1:
        return []
    row = rows[0]
    if row["text"].strip().lower() == raw_answer.strip().lower():
        return []
    result = proposal("ppocrv6_single_line", row["text"], "open_text",
                      float(row["confidence"]), "OCR recognition score, not answer probability",
                      {"bbox": row.get("bbox"), "selection": "only_high_confidence_line"})
    return [result] if result else []


def ocr_vlm_verified_proposals(route: dict[str, Any], raw_answer: str,
                               ocr_evidence: Any, vlm_evidence: Any,
                               question: str, expert_name: str = "ppocrv6_qwen3_verified") -> list[dict[str, Any]]:
    """Ask OCR to verify a *previously generated* VLM answer, not invent text."""
    if route.get("answer_type") != "open_text" or route.get("perceptual_task") != "ocr":
        return []
    if not isinstance(vlm_evidence, dict) or vlm_evidence.get("question") != question:
        return []
    if (expert_name == "ppocrv6_qwen3_32b_verified" and
            vlm_evidence.get("model_name") != "Qwen3-VL-32B-Instruct-FP8"):
        return []
    if vlm_evidence.get("status") != "complete":
        return []
    answer = normalize_answer(vlm_evidence.get("answer"), "open_text")
    if answer is None or answer.casefold() == raw_answer.strip().casefold():
        return []
    compact = " ".join(re.findall(r"\w+", answer.casefold()))
    if len(compact) < 3:
        return []
    matching = [row for row in (ocr_evidence or {}).get("ocr", [])
                if isinstance(row, dict)
                and " ".join(re.findall(r"\w+", str(row.get("text", "")).casefold())) == compact
                and float(row.get("confidence", 0.0)) >= 0.75]
    if not matching:
        return []
    best = max(matching, key=lambda row: float(row["confidence"]))
    result = proposal(expert_name, answer, "open_text",
                      float(best["confidence"]),
                      "independent OCR exact-line agreement, not correctness probability",
                      {"ocr_text": best["text"], "bbox": best.get("bbox"),
                       "model_revision": vlm_evidence.get("model_revision")})
    return [result] if result else []


def ocr_vlm_phrase_proposals(route: dict[str, Any], raw_answer: str,
                             ocr_evidence: Any, vlm_evidence: Any,
                             question: str) -> list[dict[str, Any]]:
    """Independent OCR support for a VLM phrase, including a misrouted OCR task.

    This deliberately creates *candidates*, not automatic corrections. The
    answer must occur as complete contiguous OCR tokens; character substrings
    (e.g. ``cat`` in ``catalog``) and ungrounded VLM guesses are rejected.
    """
    if route.get("answer_type") != "open_text":
        return []
    if (not isinstance(vlm_evidence, dict) or
            vlm_evidence.get("question") != question or
            vlm_evidence.get("status") != "complete"):
        return []
    answer = normalize_answer(vlm_evidence.get("answer"), "open_text")
    if answer is None or answer.casefold() == raw_answer.strip().casefold():
        return []
    tokens = re.findall(r"\w+", answer.casefold())
    if (not tokens or len(tokens) > 6 or
            not any(len(token) >= 4 for token in tokens) or
            tokens == ["none"]):
        return []
    phrase = " ".join(tokens)
    matches = []
    if not isinstance(ocr_evidence, dict):
        return []
    for row in ocr_evidence.get("ocr", []):
        if not isinstance(row, dict):
            continue
        score = float(row.get("confidence", 0.0))
        if not math.isfinite(score) or score < 0.85 or score > 1.0:
            continue
        line = " ".join(re.findall(r"\w+", str(row.get("text", "")).casefold()))
        if (line == phrase and route.get("perceptual_task") == "ocr") or not re.search(
                r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", line):
            continue
        matches.append(row)
    if not matches:
        return []
    best = max(matches, key=lambda row: float(row["confidence"]))
    expert = ("ppocrv6_qwen3_phrase_verified" if route.get("perceptual_task") == "ocr"
              else "ppocrv6_qwen3_crossroute_verified")
    candidate = proposal(
        expert, answer, "open_text", float(best["confidence"]),
        "independent OCR contains complete VLM-answer phrase; not correctness probability",
        {"ocr_text": best["text"], "bbox": best.get("bbox"),
         "route_perceptual_task": route.get("perceptual_task"),
         "model_revision": vlm_evidence.get("model_revision")},
    )
    return [candidate] if candidate else []


def spatial_proposals(route: dict[str, Any], raw_answer: str,
                      evidence: Any, question_id: str) -> list[dict[str, Any]]:
    if route.get("answer_type") != "entity" or route.get("perceptual_task") != "spatial_relation":
        return []
    candidates = (evidence or {}).get("candidates", []) if isinstance(evidence, dict) else []
    result = []
    for row in candidates:
        if (not isinstance(row, dict) or str(row.get("question_id")) != question_id or
                row.get("status") != "geometry_verified" or
                row.get("syntax_verified") is not True or
                row.get("relation") not in {"left_of", "right_of", "above", "below"}):
            continue
        anchor = row.get("anchor_box")
        candidate_box = row.get("candidate_box")
        if not _relation_holds(candidate_box, anchor, row["relation"]):
            continue
        answer = normalize_answer(row.get("answer"), "entity")
        if answer is None or answer.lower() == raw_answer.strip().lower():
            continue
        candidate = proposal("grounded_2d_geometry", answer, "entity",
                             float(row.get("score", 0.0)),
                             "grounding score with verified geometry, not answer probability",
                             {"relation": row["relation"], "anchor_box": row.get("anchor_box"),
                              "candidate_box": row.get("candidate_box")})
        if candidate:
            result.append(candidate)
    return result


def qwen3_vl_proposals(route: dict[str, Any], raw_answer: str,
                       evidence: Any, question: str, expert_name: str = "qwen3_vl_8b") -> list[dict[str, Any]]:
    """Independent image-question candidate; never use it as a ground-truth label.

    This is a competing VLM, not a specialized detector. Its fixed score is an
    uncalibrated placeholder and cannot itself trigger acceptance.
    """
    if not isinstance(evidence, dict) or evidence.get("question") != question:
        return []
    if expert_name == "qwen3_vl_32b_fp8" and evidence.get("model_name") != "Qwen3-VL-32B-Instruct-FP8":
        return []
    if evidence.get("status") != "complete":
        return []
    answer_type = str(route.get("answer_type", ""))
    answer = normalize_answer(evidence.get("answer"), answer_type)
    if answer is None or answer.lower() == raw_answer.strip().lower():
        return []
    result = proposal(expert_name, answer, answer_type, 0.5,
                      "fixed uncalibrated competing-model score",
                      {"model_revision": evidence.get("model_revision")})
    return [result] if result else []


def blip_vqa_proposals(route: dict[str, Any], raw_answer: str,
                       evidence: Any, question: str) -> list[dict[str, Any]]:
    if (not isinstance(evidence, dict) or evidence.get("question") != question or
            evidence.get("model_name") != "Salesforce/blip-vqa-base" or
            evidence.get("status") != "complete"):
        return []
    answer_type = str(route.get("answer_type", ""))
    answer = normalize_answer(evidence.get("answer"), answer_type)
    if answer is None or answer.casefold() == raw_answer.strip().casefold():
        return []
    result = proposal("blip_vqa_base", answer, answer_type, 0.5,
                      "fixed uncalibrated released VQA-model candidate",
                      {"model_name": "Salesforce/blip-vqa-base"})
    return [result] if result else []


def _relation_holds(anchor: Any, candidate: Any, relation: str) -> bool:
    """Verify external grounding geometry; relation describes first to second."""
    if not (isinstance(anchor, list) and isinstance(candidate, list)
            and len(anchor) == len(candidate) == 4):
        return False
    try:
        ax1, ay1, ax2, ay2 = map(float, anchor)
        bx1, by1, bx2, by2 = map(float, candidate)
    except (TypeError, ValueError):
        return False
    if (not all(math.isfinite(x) for x in (ax1, ay1, ax2, ay2, bx1, by1, bx2, by2))
            or ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1):
        return False
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
    xmargin = 0.1 * max(ax2 - ax1, bx2 - bx1)
    ymargin = 0.1 * max(ay2 - ay1, by2 - by1)
    return {"left_of": acx + xmargin < bcx,
            "right_of": acx - xmargin > bcx,
            "above": acy + ymargin < bcy,
            "below": acy - ymargin > bcy}[relation]


def collect_proposals(question: dict[str, Any], baseline: dict[str, Any],
                      route: dict[str, Any], evidence: dict[str, Any],
                      include_ocr_phrase: bool = False) -> list[dict[str, Any]]:
    image = question["image"]
    raw = str(baseline["raw_answer"])
    proposals = []
    proposals.extend(presence_proposals(route, raw, evidence.get("presence", {}).get(image)))
    proposals.extend(color_proposals(route, raw, evidence.get("color", {}).get(image)))
    proposals.extend(siglip2_color_proposals(route, raw,
                                             evidence.get("siglip2_color", {}).get(image),
                                             question["question"]))
    proposals.extend(count_anything_proposals(route, question["question"], raw,
                                               evidence.get("count_anything", {}).get(image)))
    proposals.extend(countgd_proposals(
        route, question, raw,
        evidence.get("countgd", {}).get(str(question["question_id"]))))
    proposals.extend(ocr_proposals(route, raw, evidence.get("ocr", {}).get(image)))
    proposals.extend(ocr_vlm_verified_proposals(
        route, raw, evidence.get("ocr", {}).get(image),
        evidence.get("qwen3_vl", {}).get(str(question["question_id"])), question["question"]))
    if include_ocr_phrase:
        proposals.extend(ocr_vlm_phrase_proposals(
            route, raw, evidence.get("ocr", {}).get(image),
            evidence.get("qwen3_vl", {}).get(str(question["question_id"])), question["question"]))
    proposals.extend(ocr_vlm_verified_proposals(
        route, raw, evidence.get("ocr", {}).get(image),
        evidence.get("qwen3_vl_32b", {}).get(str(question["question_id"])), question["question"],
        expert_name="ppocrv6_qwen3_32b_verified"))
    proposals.extend(spatial_proposals(route, raw, evidence.get("spatial", {}).get(image),
                                       str(question["question_id"])))
    proposals.extend(qwen3_vl_proposals(route, raw,
                                        evidence.get("qwen3_vl", {}).get(str(question["question_id"])),
                                        question["question"]))
    proposals.extend(qwen3_vl_proposals(route, raw,
                                        evidence.get("qwen3_vl_32b", {}).get(str(question["question_id"])),
                                        question["question"], expert_name="qwen3_vl_32b_fp8"))
    proposals.extend(blip_vqa_proposals(route, raw,
                                        evidence.get("blip_vqa", {}).get(str(question["question_id"])),
                                        question["question"]))
    answer_type = route.get("answer_type")
    return [item for item in proposals if item["answer_type"] == answer_type]
