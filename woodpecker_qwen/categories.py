import re
from typing import Any, Dict


CATEGORY_OBJECT = "object_presence"
CATEGORY_COUNT = "count"
CATEGORY_COLOR = "color"
CATEGORY_SPATIAL = "spatial_relation"
CATEGORY_OCR = "ocr"

# Legacy names remain accepted when reading old experiment files.
CATEGORY_ATTRIBUTE = "attribute"
CATEGORY_FACTUAL = "factual_consistency"

VALID_CATEGORIES = {
    CATEGORY_OBJECT,
    CATEGORY_COUNT,
    CATEGORY_COLOR,
    CATEGORY_SPATIAL,
    CATEGORY_OCR,
}


def _rule(
    low: float,
    high: float,
    min_yes_votes: int,
    min_confidence: float,
    negative_votes: int,
    negative_confidence: float,
    description: str,
) -> Dict[str, Any]:
    return {
        "change_to_no_threshold": low,
        "change_to_yes_threshold": high,
        "baseline_prior_strength": 0.65,
        "min_yes_votes": min_yes_votes,
        "min_avg_confidence": min_confidence,
        "negative_veto_votes": negative_votes,
        "negative_min_avg_confidence": negative_confidence,
        "source_weights": {
            "baseline": 0.70,
            "qwen_full": 0.85,
            "qwen_crop": 1.00,
            "qwen_highlight": 0.95,
            "object_detector": 1.10,
            "grounding_dino": 1.25,
            "sam3": 1.35,
            "sam3_color": 1.35,
            "grounding_dino_color": 1.15,
            "count": 1.30,
            "color": 1.35,
            "geometry": 1.35,
            "ocr": 1.45,
            "depth": 1.15,
        },
        "description": description,
    }


DEFAULT_RULES: Dict[str, Dict[str, Any]] = {
    CATEGORY_OBJECT: _rule(
        0.16,
        0.82,
        2,
        0.55,
        2,
        0.82,
        "Object existence uses detector or mask evidence plus independent visual views.",
    ),
    CATEGORY_COUNT: _rule(
        0.18,
        0.84,
        2,
        0.60,
        2,
        0.80,
        "Object count uses deduplicated instances and crop-level visual verification.",
    ),
    CATEGORY_COLOR: _rule(
        0.20,
        0.82,
        2,
        0.60,
        2,
        0.78,
        "Color uses foreground masks and deterministic color statistics.",
    ),
    CATEGORY_SPATIAL: _rule(
        0.18,
        0.84,
        2,
        0.62,
        2,
        0.80,
        "Spatial relations use mask or box geometry and optional depth evidence.",
    ),
    CATEGORY_OCR: _rule(
        0.20,
        0.80,
        1,
        0.60,
        1,
        0.78,
        "Scene text uses OCR recognition confidence plus a text-region visual view.",
    ),
}


def infer_category(question: str, answer: str = "") -> str:
    text = f"{question} {answer}".lower()
    ocr_markers = [
        "text", "word", "letter", "sign say", "read", "written", "spell",
        "logo say", "number on", "文字", "写着", "读出", "标志", "牌子",
    ]
    count_markers = [
        "how many", "number of", "count", "多少", "几个", "数量", "一只", "两个", "三辆",
    ]
    color_markers = [
        "color", "colour", "red", "blue", "green", "yellow", "black",
        "white", "orange", "purple", "pink", "gray", "grey", "brown",
        "颜色", "红色", "蓝色", "绿色", "黄色", "黑色", "白色",
    ]
    spatial_markers = [
        "left", "right", "above", "below", "under", "behind", "front of",
        "next to", "near", "inside", "overlap", "on top", "位置", "左边",
        "右边", "上方", "下方", "前面", "后面", "里面", "旁边",
    ]
    object_markers = [
        "is there", "are there", "contain", "visible", "can you see",
        "does the image have", "是否有", "有没有", "是否存在", "图中有",
    ]

    if any(marker in text for marker in ocr_markers):
        return CATEGORY_OCR
    numbered_existence = re.search(
        r"\bare there\s+(?:exactly\s+)?(?:\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\b",
        text,
    )
    if any(marker in text for marker in count_markers) or numbered_existence:
        return CATEGORY_COUNT
    if any(marker in text for marker in color_markers):
        return CATEGORY_COLOR
    if any(marker in text for marker in spatial_markers):
        return CATEGORY_SPATIAL
    if any(marker in text for marker in object_markers):
        return CATEGORY_OBJECT
    return CATEGORY_OBJECT


def normalize_category(category: str, question: str = "", answer: str = "") -> str:
    value = (category or "").strip().lower()
    aliases = {
        "object": CATEGORY_OBJECT,
        "object_existence": CATEGORY_OBJECT,
        "existence": CATEGORY_OBJECT,
        "object_count": CATEGORY_COUNT,
        "counting": CATEGORY_COUNT,
        "attribute_color": CATEGORY_COLOR,
        "colour": CATEGORY_COLOR,
        "text": CATEGORY_OCR,
        "scene_text": CATEGORY_OCR,
        "spatial": CATEGORY_SPATIAL,
        "position": CATEGORY_SPATIAL,
        "relation": CATEGORY_SPATIAL,
    }
    value = aliases.get(value, value)
    if value in {CATEGORY_ATTRIBUTE, CATEGORY_FACTUAL}:
        return infer_category(question, answer)
    if value in VALID_CATEGORIES:
        return value
    return infer_category(question, answer)
