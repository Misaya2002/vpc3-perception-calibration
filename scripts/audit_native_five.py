"""Audit native-answer correction only after predictions and inputs are frozen.

The composite's OCR correctness is a predeclared consensus rule, *not* the
official TextVQA leaderboard score.  A ten-annotator VQA-style soft score is
also reported separately for the OCR subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


CATEGORIES = ("object_presence", "color", "spatial_relation", "count", "ocr")
NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).lower().strip()
    text = text.replace("grey", "gray")
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [token for token in text.split() if token not in {"a", "an", "the"}]
    text = " ".join(tokens)
    return NUMBER_WORDS.get(text, text)


def correctness(answer: str, truth: dict) -> tuple[bool, float | None]:
    value = normalize(answer)
    references = [normalize(reference) for reference in truth["references"]]
    if truth["category"] != "ocr":
        return value != "" and value in references, None
    counts = Counter(references)
    # If annotators never agree three times, accept only the most-supported
    # answer(s).  This avoids making any low-agreement question impossible.
    threshold = min(3, max(counts.values()))
    votes = counts[value]
    return value != "" and votes >= threshold, min(1.0, votes / 3.0)


def exact_mcnemar(success: int, harm: int) -> float:
    from math import comb

    discordant = success + harm
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, k) for k in range(min(success, harm) + 1))
    return min(1.0, 2 * tail / (2 ** discordant))


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    baseline_correct = sum(row["before_correct"] for row in rows)
    calibrated_correct = sum(row["after_correct"] for row in rows)
    success = sum(not row["before_correct"] and row["after_correct"] for row in rows)
    harm = sum(row["before_correct"] and not row["after_correct"] for row in rows)
    baseline_accuracy = baseline_correct / n if n else None
    calibrated_accuracy = calibrated_correct / n if n else None
    accuracy_gain = (calibrated_accuracy - baseline_accuracy) if n else None
    return {
        "n": n,
        "images": len({row["image"] for row in rows}),
        "baseline_correct": baseline_correct,
        "calibrated_correct": calibrated_correct,
        "baseline_accuracy": baseline_accuracy,
        "calibrated_accuracy": calibrated_accuracy,
        "accuracy_gain": accuracy_gain,
        "accuracy_gain_percentage_points": 100.0 * accuracy_gain if accuracy_gain is not None else None,
        "success": success,
        "harm": harm,
        "net": success - harm,
        "mcnemar_exact_p": exact_mcnemar(success, harm),
    }


def bootstrap_accuracy_gain_pp(rows: list[dict], replicates: int = 5000) -> list[float] | None:
    """Paired image-group bootstrap interval for absolute accuracy gain, in pp."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in rows:
        groups[item["image"]].append(item)
    keys = sorted(groups)
    if not keys:
        return None
    rng = random.Random(20260928)
    gains = []
    for _ in range(replicates):
        selected = [item for _ in keys for item in groups[keys[rng.randrange(len(keys))]]]
        gains.append(100.0 * sum(int(item["after_correct"]) - int(item["before_correct"])
                                 for item in selected) / len(selected))
    gains.sort()
    return [gains[int(0.025 * (len(gains) - 1))],
            gains[int(0.975 * (len(gains) - 1))]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_summary = json.loads((args.manifest / "manifest_summary.json").read_text())
    freeze = json.loads((args.run / "PREDICTIONS_FROZEN.json").read_text())
    question_path = args.manifest / "question_only.jsonl"
    prediction_path = args.run / "predictions.jsonl"
    if (digest(question_path) != manifest_summary["artifact_sha256"]["question_only.jsonl"]
            or freeze.get("question_only_sha256") != digest(question_path)
            or freeze.get("predictions_sha256") != digest(prediction_path)):
        raise ValueError("Input/prediction freeze hash mismatch; labels remain sealed")
    questions = load_jsonl(question_path)
    predictions = load_jsonl(prediction_path)
    if len(questions) != manifest_summary["questions"] or len(predictions) != len(questions):
        raise ValueError("Incomplete frozen predictions; labels remain sealed")
    expected = {str(row["question_id"]): row for row in questions}
    if len(expected) != len(questions):
        raise ValueError("Duplicate question ID")
    output_schema = freeze.get("output_schema")
    compact_fields = {"request_id", "original_answer", "calibrated_answer", "changed", "evidence"}
    if output_schema == "vpc3.answer-evidence.v2":
        if any(set(row) != compact_fields for row in predictions):
            raise ValueError("VPC3 compact output schema mismatch; labels remain sealed")
    for row in predictions:
        if set(row) & {"label", "references", "ground_truth", "gold_answer"}:
            raise ValueError("Prediction file contains answer labels")
        request_id = str(row.get("request_id", row.get("question_id", "")))
        question = expected.pop(request_id, None)
        if question is None:
            raise ValueError("Prediction/request ID mismatch")
        if output_schema != "vpc3.answer-evidence.v2" and (
                (row.get("image"), row.get("question")) !=
                (question["image"], question["question"])):
            raise ValueError("Legacy prediction/question mismatch")
        if not ("original_answer" in row or "raw_answer" in row) or "calibrated_answer" not in row:
            raise ValueError("Missing paired answer")
        before_answer = row.get("original_answer", row.get("raw_answer"))
        expected_changed = str(before_answer).strip().casefold() != str(
            row["calibrated_answer"]).strip().casefold()
        if "changed" in row and row["changed"] is not expected_changed:
            raise ValueError("Changed flag does not match paired answers")
        evidence = row.get("evidence")
        if output_schema == "vpc3.answer-evidence.v2" and (
                not isinstance(evidence, dict) or set(evidence) !=
                {"source", "observation", "candidate_answer", "reason"}):
            raise ValueError("VPC3 compact evidence schema mismatch")
    if expected:
        raise ValueError("Missing question predictions")

    # Only after all checks above may the sealed labels be opened.
    label_path = args.manifest / "sealed_labels.jsonl"
    if digest(label_path) != manifest_summary["artifact_sha256"]["sealed_labels.jsonl"]:
        raise ValueError("Sealed label hash mismatch")
    labels = {str(row["question_id"]): row for row in load_jsonl(label_path)}
    if len(labels) != len(predictions):
        raise ValueError("Sealed label cardinality mismatch")
    question_by_id = {str(item["question_id"]): item for item in questions}
    scored = []
    for row in predictions:
        request_id = str(row.get("request_id", row.get("question_id", "")))
        truth = labels[request_id]
        question = question_by_id[request_id]
        if ((question["image"], question["question"]) != (truth["image"], truth["question"])
                or truth["category"] not in CATEGORIES):
            raise ValueError("Sealed label/question/category mismatch")
        before_answer = row.get("original_answer", row.get("raw_answer"))
        before, before_soft = correctness(before_answer, truth)
        after, after_soft = correctness(row["calibrated_answer"], truth)
        scored.append({"image": truth["image"], "category": truth["category"],
                       "before_correct": before, "after_correct": after,
                       "before_soft": before_soft, "after_soft": after_soft})
    result = {
        "role": "vpc3_native_answer_five_category_composite_audit",
        "primary_metric": "absolute_accuracy_gain_percentage_points",
        "scoring": "NFKC lowercase, punctuation/articles removed, 0-20 number words normalized; OCR consensus threshold min(3,max annotator vote)",
        "qualification": "Multi-source composite; GQA val, TallyQA test, TextVQA val; free-text OCR answers use a local consensus rule, not the official TextVQA leaderboard metric.",
        "prediction_sha256": digest(prediction_path),
        "sealed_label_sha256": digest(label_path),
        "overall": summarize(scored),
        "by_category": {},
    }
    result["overall"]["image_group_bootstrap_accuracy_gain_pp_95ci"] = bootstrap_accuracy_gain_pp(scored)
    for category in CATEGORIES:
        subset = [row for row in scored if row["category"] == category]
        part = summarize(subset)
        part["image_group_bootstrap_accuracy_gain_pp_95ci"] = bootstrap_accuracy_gain_pp(subset)
        if category == "ocr":
            part["vqa_style_soft_accuracy_before"] = sum(row["before_soft"] for row in subset) / len(subset)
            part["vqa_style_soft_accuracy_after"] = sum(row["after_soft"] for row in subset) / len(subset)
        result["by_category"][category] = part
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "primary_metric": result["primary_metric"],
        "overall": {key: result["overall"][key] for key in (
            "baseline_accuracy", "calibrated_accuracy", "accuracy_gain_percentage_points",
            "success", "harm", "net", "mcnemar_exact_p")},
        "by_category_accuracy": {key: {
            "baseline_accuracy": val["baseline_accuracy"],
            "calibrated_accuracy": val["calibrated_accuracy"],
            "accuracy_gain_percentage_points": val["accuracy_gain_percentage_points"]}
            for key, val in result["by_category"].items()},
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
