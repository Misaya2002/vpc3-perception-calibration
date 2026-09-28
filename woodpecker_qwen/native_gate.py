"""Cost-sensitive native-answer learning-to-defer gate.

Training labels never enter proposal generation or model application. The gate
scores *incremental utility* of each expert answer versus the original answer;
this supports non-binary tasks where both answers may be wrong.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

EXPERTS = ("sam3_presence", "sam3_mask_color", "siglip2_sam3_crop_color", "count_anything", "countgd",
           "grounded_2d_geometry", "ppocrv6_single_line", "qwen3_vl_8b",
           "ppocrv6_qwen3_verified", "qwen3_vl_32b_fp8",
           "ppocrv6_qwen3_32b_verified", "blip_vqa_base")
ANSWER_TYPES = ("boolean", "color", "entity", "integer", "open_text")
FEATURE_NAMES = ("intercept", *(f"expert_{name}" for name in EXPERTS),
                 *(f"answer_type_{name}" for name in ANSWER_TYPES),
                 *(f"pair_{expert}_{answer_type}" for expert in EXPERTS
                   for answer_type in ANSWER_TYPES),
                 "expert_score", "raw_length", "proposal_length",
                 "qualified_route", "has_bbox", "has_point_count",
                 "independent_agreement", "independent_conflict")
PUBLIC_EVIDENCE_KEYS = frozenset({
    "target", "bbox", "crop_box", "margin", "sam3_score", "query",
    "point_count", "points", "confidence_threshold", "ocr_text",
    "relation", "anchor_box", "candidate_box", "selection", "operands",
    "scores", "matched_objects", "logic", "model_revision", "route_perceptual_task",
    "mask_observation_available",
})


def public_observations(details: dict[str, Any] | None) -> dict[str, Any]:
    """Keep useful visual observations while excluding local file paths."""
    if not isinstance(details, dict):
        return {}
    return {key: value for key, value in details.items() if key in PUBLIC_EVIDENCE_KEYS}


def compact_evidence(evidence: dict[str, Any] | None) -> dict[str, Any]:
    """Convert internal gate traces to the small, stable VPC3 public schema."""
    evidence = evidence if isinstance(evidence, dict) else {}
    observations = evidence.get("observations")
    observations = observations if isinstance(observations, dict) else {}

    if observations.get("point_count") is not None:
        target = observations.get("query") or observations.get("target") or "目标"
        observation = f"{target}: {observations['point_count']} 个检测点"
    elif observations.get("relation"):
        observation = str(observations["relation"])
    elif observations.get("ocr_text"):
        observation = str(observations["ocr_text"]).strip()[:160]
    elif observations.get("matched_objects"):
        found = observations["matched_objects"]
        if isinstance(found, list):
            names = [str(item.get("label", item)) if isinstance(item, dict) else str(item)
                     for item in found[:5]]
            observation = "检测到: " + ", ".join(names)
        else:
            observation = str(found)[:160]
    elif observations.get("target") or observations.get("bbox"):
        observation = json.dumps(
            {key: observations[key] for key in ("target", "bbox", "candidate_box")
             if key in observations}, ensure_ascii=False, separators=(",", ":"))[:200]
    elif set(observations) == {"model_revision"}:
        observation = "独立视觉语言模型的候选回答；未提供检测框或文字定位"
    elif observations:
        observation = json.dumps(observations, ensure_ascii=False,
                                 separators=(",", ":"))[:200]
    else:
        observation = "无可用结构化观测"
    observation = str(observation)[:200]

    reason = evidence.get("reason")
    reason_map = {
        "highest_scoring_candidate_above_threshold": "candidate_passed_gate",
        "below_acceptance_threshold": "candidate_below_gate",
        "lower_score_than_another_candidate": "another_candidate_scored_higher",
        "no_expert_disagreement": "no_alternative_candidate",
    }
    return {
        "source": evidence.get("source"),
        "observation": observation,
        "candidate_answer": evidence.get("candidate_answer"),
        "reason": reason_map.get(reason, reason or "no_candidate_selected"),
    }


def features(row: dict[str, Any], candidate: dict[str, Any]) -> np.ndarray:
    expert = candidate["expert"]
    answer_type = row["answer_type"]
    if expert not in EXPERTS or answer_type not in ANSWER_TYPES:
        raise ValueError("Unknown native expert/answer type")
    if candidate["answer_type"] != answer_type:
        raise ValueError("Expert proposal changes the native answer type")
    details = candidate.get("details") or {}
    other = [item for item in row.get("proposals", [])
             if item is not candidate and item.get("expert") != expert
             and item.get("answer_type") == answer_type]
    agreement = sum(str(item.get("answer", "")).strip().lower() ==
                    str(candidate.get("answer", "")).strip().lower() for item in other)
    return np.asarray([
        1.0, *(float(expert == name) for name in EXPERTS),
        *(float(answer_type == name) for name in ANSWER_TYPES),
        *(float(expert == name and answer_type == kind)
          for name in EXPERTS for kind in ANSWER_TYPES),
        float(candidate["score"]),
        min(len(str(row["raw_answer"])), 40) / 40.0,
        min(len(str(candidate["answer"])), 40) / 40.0,
        float("qualified" in str(row.get("parse_status", ""))),
        float(bool(details.get("bbox") or details.get("candidate_box"))),
        float("point_count" in details),
        min(agreement, 2) / 2.0,
        min(len(other) - agreement, 2) / 2.0,
    ], dtype=float)


def fit_ridge(x: np.ndarray, utility: np.ndarray, penalty: float = 10.0) -> np.ndarray:
    if x.ndim != 2 or len(x) != len(utility) or len(x) < 20:
        raise ValueError("Need at least 20 aligned development proposals")
    regularizer = np.eye(x.shape[1]) * penalty
    regularizer[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + regularizer, x.T @ utility)


def select(row: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    if tuple(model["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Gate feature schema mismatch")
    weights = np.asarray(model["weights"], dtype=float)
    if weights.shape != (len(FEATURE_NAMES),):
        raise ValueError("Gate model shape mismatch")
    threshold = float(model["utility_threshold"])
    minimum = int(model["minimum_expert_support"])
    support = model["expert_training_support"]
    raw = str(row["raw_answer"])
    best = {"answer": raw, "expert": "qwen_baseline", "expected_utility": 0.0,
            "decision": "retain_baseline", "evidence": {
                "status": "no_alternative_selected", "source": None,
                "observations": {},
            }}
    strongest_rejected: dict[str, Any] | None = None
    for candidate in row.get("proposals", []):
        if candidate.get("answer_type") != row.get("answer_type"):
            continue
        if candidate.get("answer", "").strip().lower() == raw.strip().lower():
            continue
        name = candidate.get("expert")
        support_key = f"{name}|{row['answer_type']}"
        if name not in EXPERTS or int(support.get(support_key, 0)) < minimum:
            continue
        score = float(features(row, candidate) @ weights)
        evidence = {
            "status": "candidate_not_selected",
            "source": name,
            "candidate_answer": candidate["answer"],
            "source_score": float(candidate["score"]),
            "source_score_semantics": candidate.get("score_semantics"),
            "observations": public_observations(candidate.get("details")),
            "gate_expected_utility": score,
            "gate_score_semantics": "learned incremental utility; not a correctness probability",
            "threshold": threshold,
            "reason": "below_acceptance_threshold" if score <= threshold else "lower_score_than_another_candidate",
        }
        if strongest_rejected is None or score > strongest_rejected["score"]:
            strongest_rejected = {"score": score, "evidence": evidence}
        if score > threshold and score > best["expected_utility"]:
            best = {"answer": candidate["answer"], "expert": name,
                    "expected_utility": round(score, 6), "decision": "accept_expert",
                    "evidence": {
                        **evidence,
                        "status": "used_for_correction",
                        "reason": "highest_scoring_candidate_above_threshold",
                    }}
    if best["decision"] == "retain_baseline" and strongest_rejected is not None:
        best["evidence"] = strongest_rejected["evidence"]
    return best


def image_fold(image: str, folds: int) -> int:
    return int(hashlib.sha256(str(image).encode()).hexdigest(), 16) % folds
