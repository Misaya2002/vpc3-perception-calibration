import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .categories import (
    CATEGORY_COLOR,
    CATEGORY_COUNT,
    CATEGORY_OBJECT,
    CATEGORY_OCR,
    CATEGORY_SPATIAL,
)


LABEL_ALIASES = {
    "automobile": "car",
    "bike": "bicycle",
    "cycle": "bicycle",
    "table": "dining table",
    "ski": "skis",
}


def normalize_label(value: Any) -> str:
    text = str(value or "").lower().replace("_", " ").replace("-", " ").strip()
    text = " ".join(text.split())
    return LABEL_ALIASES.get(text, text)


def labels_match(first: Any, second: Any) -> bool:
    left = normalize_label(first)
    right = normalize_label(second)
    if not left or not right:
        return False
    if left == right:
        return True
    return (left.endswith("s") and left[:-1] == right) or (
        right.endswith("s") and right[:-1] == left
    )


def tool_evidence_for_claims(
    claims: Iterable[Dict[str, Any]],
    observations: Iterable[Dict[str, Any]],
    count_dedup_iou: Optional[float] = None,
) -> List[Dict[str, Any]]:
    observation_list = list(observations)
    objects = _object_rows(observation_list)
    colors = _typed_rows(observation_list, {"color", "color_analysis"})
    ocr_rows = _typed_rows(observation_list, {"ocr"})
    relations = _typed_rows(observation_list, {"spatial_relation", "geometry"})

    result: List[Dict[str, Any]] = []
    for claim in claims:
        category = claim.get("category")
        if category == CATEGORY_OBJECT:
            items = [
                _object_evidence(claim, rows, source=source)
                for source, rows in _rows_by_source(objects).items()
            ]
        elif category == CATEGORY_COUNT:
            items = [
                _count_evidence(
                    claim, rows, source=source, dedup_iou=count_dedup_iou
                )
                for source, rows in _rows_by_source(objects).items()
            ]
        elif category == CATEGORY_COLOR:
            items = [
                _color_evidence(claim, rows, source=source)
                for source, rows in _rows_by_source(colors, default="color").items()
            ]
        elif category == CATEGORY_OCR:
            items = [_ocr_evidence(claim, ocr_rows)]
        elif category == CATEGORY_SPATIAL:
            items = [_spatial_evidence(claim, relations, objects)]
        else:
            items = []
        result.extend(item for item in items if item)
    return result


def _base_evidence(
    claim: Dict[str, Any], source: str, answer: str, probability: float, explanation: str
) -> Dict[str, Any]:
    probability = max(0.0, min(1.0, float(probability)))
    return {
        "claim_id": claim["claim_id"],
        "question_id": f"{claim['claim_id']}_{source}",
        "question": f"Local {source} check for {claim.get('target', '')}",
        "answer": answer,
        "confidence": round(probability if answer == "yes" else 1.0 - probability, 4),
        "yes_probability": round(probability, 4),
        "evidence": explanation,
        "source": source,
        "source_family": source,
        "raw": "",
    }


def _object_evidence(
    claim: Dict[str, Any], rows: List[Dict[str, Any]], source: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    matched = [row for row in rows if labels_match(claim.get("target"), row.get("label"))]
    source = source or _object_source(matched or rows)
    if matched:
        best = max(float(row.get("score", row.get("confidence", 0.5))) for row in matched)
        probability = max(0.70, min(0.98, best))
        item = _base_evidence(
            claim,
            source,
            "yes",
            probability,
            f"{source} found {len(matched)} instance(s) of {claim.get('target')} (best score {best:.3f}).",
        )
        item["detector_score"] = best
        item["matched_count"] = len(matched)
        return item
    labels = sorted({normalize_label(row.get("label")) for row in rows if row.get("label")})
    return _base_evidence(
        claim,
        source,
        "no",
        0.40,
        f"{source} did not find {claim.get('target')}; absence is treated as weak evidence. Labels: {', '.join(labels[:12])}.",
    )


def _count_evidence(
    claim: Dict[str, Any],
    rows: List[Dict[str, Any]],
    source: str = "count",
    dedup_iou: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    expected = claim.get("expected_value")
    try:
        expected = int(expected)
    except (TypeError, ValueError):
        return None
    matched = [row for row in rows if labels_match(claim.get("target"), row.get("label"))]
    if not rows:
        return None
    raw_count = len(matched)
    if dedup_iou is not None:
        matched = _deduplicate_boxes(matched, dedup_iou)
    observed = len(matched)
    answer = "yes" if observed == expected else "no"
    probability = 0.88 if answer == "yes" else 0.16
    item = _base_evidence(
        claim,
        source,
        answer,
        probability,
        f"Instance evidence counted {observed} {claim.get('target')}; expected {expected}.",
    )
    item.update(
        {
            "observed_count": observed,
            "raw_count": raw_count,
            "expected_count": expected,
            "instance_dedup_iou": dedup_iou,
        }
    )
    return item


def _deduplicate_boxes(rows: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    """Suppress duplicate detections within one source; preserve rows lacking geometry."""
    if not 0.0 < float(threshold) <= 1.0:
        raise ValueError("count_dedup_iou must be in (0, 1]")
    ordered = sorted(
        rows,
        key=lambda row: float(row.get("score", row.get("confidence", 0.0))),
        reverse=True,
    )
    kept: List[Dict[str, Any]] = []
    for row in ordered:
        box = _valid_box(row.get("bbox"))
        if box is not None and any(
            _box_iou(box, prior_box) >= threshold
            for prior in kept
            for prior_box in [_valid_box(prior.get("bbox"))]
            if prior_box is not None
        ):
            continue
        kept.append(row)
    return kept


def _valid_box(value: Any) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        left, top, right, bottom = (float(part) for part in value)
    except (TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _box_iou(first: Tuple[float, ...], second: Tuple[float, ...]) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _color_evidence(
    claim: Dict[str, Any], rows: List[Dict[str, Any]], source: str = "color"
) -> Optional[Dict[str, Any]]:
    target_rows = [row for row in rows if labels_match(claim.get("target"), row.get("target") or row.get("label"))]
    if not target_rows:
        return None
    expected_colors = _expected_colors(claim.get("expected_value"))
    if not expected_colors:
        return None
    best = max(target_rows, key=lambda row: _color_match_score(row, expected_colors))
    observed = normalize_label(best.get("color") or best.get("value"))
    score = float(best.get("confidence", best.get("score", 0.75)))
    distribution = best.get("distribution", {})
    proportions = [float(distribution.get(color, 0.0)) for color in expected_colors]
    thresholds = [0.16 if color in {"black", "white", "gray"} else 0.08 for color in expected_colors]
    if distribution:
        matched = all(value >= threshold for value, threshold in zip(proportions, thresholds))
        strength = min(
            [value / max(threshold, 1e-6) for value, threshold in zip(proportions, thresholds)]
            or [0.0]
        )
        probability = min(0.96, 0.58 + 0.16 * strength) if matched else max(0.08, 0.40 - 0.18 * strength)
    else:
        matched = len(expected_colors) == 1 and observed == expected_colors[0]
        probability = max(0.55, min(0.97, score)) if matched else min(0.35, 1.0 - score)
    item = _base_evidence(
        claim,
        source,
        "yes" if matched else "no",
        probability,
        f"Masked color analysis found {observed or 'unknown'} for {claim.get('target')}; expected {' and '.join(expected_colors)}.",
    )
    item.update(
        {
            "observed_color": observed,
            "expected_color": "|".join(expected_colors),
            "color_distribution": distribution,
        }
    )
    return item


def _expected_colors(value: Any) -> List[str]:
    text = str(value or "").lower().replace("grey", "gray")
    return [color for color in ("red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray") if re.search(rf"\b{color}\b", text)]


def _color_match_score(row: Dict[str, Any], expected: List[str]) -> float:
    distribution = row.get("distribution", {})
    if distribution:
        return min(float(distribution.get(color, 0.0)) for color in expected)
    observed = normalize_label(row.get("color") or row.get("value"))
    return float(row.get("confidence", row.get("score", 0.5))) if observed in expected else 0.0


def _ocr_evidence(claim: Dict[str, Any], rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    texts = []
    confidences = []
    for row in rows:
        text = str(row.get("text", "")).strip()
        if text:
            texts.append(text)
            confidences.append(float(row.get("confidence", row.get("score", 0.5))))
    if not texts:
        return None
    expected = _normalize_text(claim.get("expected_value"))
    observed = _normalize_text(" ".join(texts))
    matched = bool(expected and expected in observed)
    best = max(confidences or [0.5])
    probability = max(0.60, min(0.98, best)) if matched else max(0.10, min(0.40, 1.0 - best))
    item = _base_evidence(
        claim,
        "ocr",
        "yes" if matched else "no",
        probability,
        f"OCR read: {' | '.join(texts[:8])}; expected text: {claim.get('expected_value') or 'unspecified'}.",
    )
    item["ocr_texts"] = texts
    item["ocr_matched"] = matched
    item["ocr_best_confidence"] = round(best, 4)
    item["ocr_expected_text"] = expected
    return item


def _spatial_evidence(
    claim: Dict[str, Any], relations: List[Dict[str, Any]], objects: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    target = claim.get("target")
    reference = claim.get("reference_target")
    expected = str(claim.get("relation") or "").lower()
    for row in relations:
        if (
            labels_match(target, row.get("subject") or row.get("target"))
            and labels_match(reference, row.get("object") or row.get("reference_target"))
            and str(row.get("relation", "")).lower() == expected
        ):
            score = float(row.get("confidence", row.get("score", 0.85)))
            return _base_evidence(
                claim,
                "geometry",
                "yes",
                max(0.65, min(0.98, score)),
                f"Geometry evidence supports {target} {expected} {reference}.",
            )

    target_box = _best_box(objects, target)
    reference_boxes = _matching_boxes(objects, reference)
    reference_box = _union_boxes(reference_boxes) if expected == "middle_of" else (
        reference_boxes[0] if reference_boxes else None
    )
    if not target_box or not reference_box or not expected:
        return None
    observed = _relation_holds(target_box, reference_box, expected)
    if observed is None:
        return None
    probability = 0.88 if observed else 0.12
    return _base_evidence(
        claim,
        "geometry",
        "yes" if observed else "no",
        probability,
        f"Box geometry {'supports' if observed else 'contradicts'} {target} {expected} {reference}.",
    )


def _relation_holds(first: List[float], second: List[float], relation: str) -> Optional[bool]:
    ax, ay = (first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0
    bx, by = (second[0] + second[2]) / 2.0, (second[1] + second[3]) / 2.0
    if relation == "left_of":
        return ax < bx
    if relation == "right_of":
        return ax > bx
    if relation == "above":
        return ay < by
    if relation == "below":
        return ay > by
    if relation == "inside":
        return first[0] >= second[0] and first[1] >= second[1] and first[2] <= second[2] and first[3] <= second[3]
    if relation == "outside":
        return not (
            first[0] >= second[0]
            and first[1] >= second[1]
            and first[2] <= second[2]
            and first[3] <= second[3]
        )
    if relation == "middle_of":
        return abs(ax - bx) <= 0.22 * max(1.0, second[2] - second[0])
    if relation == "overlapping":
        return min(first[2], second[2]) > max(first[0], second[0]) and min(first[3], second[3]) > max(first[1], second[1])
    if relation == "next_to":
        width = max(first[2], second[2]) - min(first[0], second[0])
        return abs(ax - bx) < 0.45 * max(width, 1.0)
    # Front/behind and unknown relations need depth or VLM evidence.
    return None


def _best_box(rows: List[Dict[str, Any]], label: Any) -> Optional[List[float]]:
    if normalize_label(label) in {"image", "picture", "photo"}:
        for row in rows:
            size = row.get("image_size")
            if isinstance(size, list) and len(size) == 2:
                return [0.0, 0.0, float(size[0]), float(size[1])]
    matched = [row for row in rows if labels_match(label, row.get("label")) and row.get("bbox")]
    if not matched:
        return None
    best = max(matched, key=lambda row: float(row.get("score", row.get("confidence", 0.0))))
    box = best.get("bbox")
    if isinstance(box, list) and len(box) == 4:
        return [float(value) for value in box]
    return None


def _matching_boxes(rows: List[Dict[str, Any]], label: Any) -> List[List[float]]:
    if normalize_label(label) in {"image", "picture", "photo"}:
        image_box = _best_box(rows, label)
        return [image_box] if image_box else []
    matched = [
        row
        for row in rows
        if labels_match(label, row.get("label")) and isinstance(row.get("bbox"), list)
    ]
    matched.sort(
        key=lambda row: float(row.get("score", row.get("confidence", 0.0))), reverse=True
    )
    return [
        [float(value) for value in row["bbox"]]
        for row in matched
        if len(row["bbox"]) == 4
    ]


def _union_boxes(boxes: List[List[float]]) -> Optional[List[float]]:
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _object_rows(observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for observation in observations:
        if observation.get("type") not in {"object_detection", "segmentation", "objects"}:
            continue
        content = observation.get("content", {})
        if isinstance(content, list):
            candidates = content
        else:
            candidates = content.get("detections") or content.get("objects") or content.get("segments") or []
        for row in candidates:
            if isinstance(row, dict):
                enriched = dict(row)
                enriched.setdefault("source", observation.get("tool", "object_detector"))
                rows.append(enriched)
    return rows


def _typed_rows(observations: List[Dict[str, Any]], types: set) -> List[Dict[str, Any]]:
    rows = []
    for observation in observations:
        if observation.get("type") not in types:
            continue
        content = observation.get("content", [])
        if isinstance(content, dict):
            content = content.get("items") or content.get("results") or content.get("texts") or [content]
        rows.extend(row for row in content if isinstance(row, dict))
    return rows


def _object_source(rows: List[Dict[str, Any]]) -> str:
    sources = {str(row.get("source", "")).lower() for row in rows}
    if any("sam" in source for source in sources):
        return "sam3"
    if any("grounding" in source for source in sources):
        return "grounding_dino"
    return "object_detector"


def _rows_by_source(
    rows: List[Dict[str, Any]], default: str = "object_detector"
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        raw = str(row.get("source") or default).lower()
        if "sam" in raw:
            source = "sam3_color" if "color" in raw else "sam3"
        elif "grounding" in raw:
            source = "grounding_dino_color" if "color" in raw else "grounding_dino"
        else:
            source = raw or default
        grouped.setdefault(source, []).append(row)
    return grouped


def _normalize_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").lower())
