"""Nested image-group audit of off-the-shelf selective-correction regressors.

All candidate answers are frozen before this script opens development labels.
The outer five-fold predictions are diagnostic, not confirmation results.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_native_five import correctness, digest, load_jsonl
from woodpecker_qwen.native_gate import FEATURE_NAMES, features, image_fold

PALETTE = ("black", "white", "gray", "red", "orange", "yellow", "green",
           "blue", "purple", "pink", "brown")
THRESHOLDS = (0.0, 0.025, 0.05, 0.1, 0.2)
TYPE_THRESHOLDS = THRESHOLDS + (0.3, 0.4)
MODELS = ("ridge_1", "ridge_10", "extra_trees", "hist_gradient_boosting", "catboost")
EXTRA_FEATURE_NAMES = ("raw_yes", "raw_no", "candidate_yes", "candidate_no",
                       "raw_number", "candidate_number", "number_gap",
                       *(f"raw_color_{color}" for color in PALETTE),
                       *(f"candidate_color_{color}" for color in PALETTE),
                       "evidence_margin", "sam3_score", "candidate_count",
                       "question_length", "qwen_confidence_available",
                       "qwen_first_token_logprob", "qwen_mean_six_logprob",
                       "qwen_yes_probability", "qwen_rerun_matches")
ALL_FEATURE_NAMES = FEATURE_NAMES + EXTRA_FEATURE_NAMES


def feature_vector(row: dict, candidate: dict, confidence: dict | None = None) -> np.ndarray:
    raw, answer = str(row["raw_answer"]).casefold(), str(candidate["answer"]).casefold()
    details = candidate.get("details") or {}
    raw_int = int(raw) if raw.isdecimal() else -1
    answer_int = int(answer) if answer.isdecimal() else -1
    confidence = confidence or {}
    first_logprob = confidence.get("first_token_logprob")
    mean_logprob = confidence.get("mean_first_six_logprob")
    yes_probability = confidence.get("yes_probability")
    confidence_available = first_logprob is not None
    extra = [float(raw == "yes"), float(raw == "no"),
             float(answer == "yes"), float(answer == "no"),
             max(raw_int, 0) / 20, max(answer_int, 0) / 20,
             abs(raw_int - answer_int) / 20 if raw_int >= 0 and answer_int >= 0 else 0,
             *(float(raw == color) for color in PALETTE),
             *(float(answer == color) for color in PALETTE),
             min(max(float(details.get("margin") or 0), 0), 1),
             min(max(float(details.get("sam3_score") or 0), 0), 1),
             min(len(row.get("proposals", [])), 5) / 5,
             min(len(str(row["question"])), 160) / 160,
             float(confidence_available),
             max(-20.0, min(0.0, float(first_logprob))) / 20 if confidence_available else 0.0,
             max(-20.0, min(0.0, float(mean_logprob))) / 20 if mean_logprob is not None else 0.0,
             float(yes_probability) if yes_probability is not None else 0.5,
             float(confidence.get("matches_frozen_answer", False))]
    return np.concatenate((features(row, candidate), np.asarray(extra, dtype=float)))


def make_model(name: str):
    if name == "ridge_1":
        return Ridge(alpha=1.0)
    if name == "ridge_10":
        return Ridge(alpha=10.0)
    if name == "extra_trees":
        return ExtraTreesRegressor(n_estimators=160, max_depth=5,
                                   min_samples_leaf=3, max_features=0.8,
                                   random_state=20260927, n_jobs=1)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=5,
                                             min_samples_leaf=10, learning_rate=0.05,
                                             l2_regularization=10, random_state=20260927)
    if name == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(iterations=180, depth=4, learning_rate=0.05,
                                 l2_leaf_reg=5, loss_function="RMSE", random_seed=20260927,
                                 thread_count=2, verbose=False, allow_writing_files=False)
    raise ValueError(name)


def examples_for(rows: list[dict], labels: dict, harm_cost: float,
                 confidence: dict[str, dict] | None = None):
    vectors, utilities, row_indices, candidate_indices = [], [], [], []
    for row_index, row in enumerate(rows):
        before = correctness(row["raw_answer"], labels[row["question_id"]])[0]
        for candidate_index, candidate in enumerate(row.get("proposals", [])):
            if str(candidate["answer"]).casefold() == str(row["raw_answer"]).casefold():
                continue
            after = correctness(candidate["answer"], labels[row["question_id"]])[0]
            vectors.append(feature_vector(row, candidate,
                                          (confidence or {}).get(row["question_id"])))
            utilities.append(1.0 if after and not before else
                             -harm_cost if before and not after else 0.0)
            row_indices.append(row_index)
            candidate_indices.append(candidate_index)
    if not vectors:
        raise ValueError("No candidate disagreements")
    return (np.stack(vectors), np.asarray(utilities),
            np.asarray(row_indices), np.asarray(candidate_indices))


def decisions(rows: list[dict], row_indices: np.ndarray, candidate_indices: np.ndarray,
              scores: np.ndarray, allowed: set[int], threshold: float | dict) -> dict[int, dict]:
    selected: dict[int, dict] = {}
    for index, (row_index, candidate_index, score) in enumerate(
            zip(row_indices, candidate_indices, scores)):
        row_index = int(row_index)
        row_threshold = (threshold[rows[row_index]["answer_type"]]
                         if isinstance(threshold, dict) else threshold)
        if row_index not in allowed or score <= row_threshold:
            continue
        if row_index not in selected or score > selected[row_index]["score"]:
            selected[row_index] = {"candidate_index": int(candidate_index),
                                   "score": float(score), "example_index": index}
    return selected


def realized_utility(rows: list[dict], labels: dict, selected: dict[int, dict],
                     allowed: set[int], harm_cost: float) -> tuple[float, int, int]:
    success = harm = 0
    for row_index in allowed:
        choice = selected.get(row_index)
        if choice is None:
            continue
        row = rows[row_index]
        before = correctness(row["raw_answer"], labels[row["question_id"]])[0]
        after = correctness(row["proposals"][choice["candidate_index"]]["answer"],
                            labels[row["question_id"]])[0]
        success += int(after and not before)
        harm += int(before and not after)
    return success - harm_cost * harm, success, harm


def choose_model(rows: list[dict], labels: dict, vectors: np.ndarray,
                 utilities: np.ndarray, row_indices: np.ndarray,
                 candidate_indices: np.ndarray, eligible: set[int],
                 harm_cost: float, by_answer_type: bool = False) -> tuple[str, float | dict, dict]:
    totals = defaultdict(lambda: [0.0, 0, 0])
    type_totals = defaultdict(lambda: [0.0, 0, 0])
    for fold in range(4):
        heldout = {i for i in eligible
                   if image_fold("gate-inner:" + rows[i]["image"], 4) == fold}
        train_mask = np.asarray([int(index) in eligible and int(index) not in heldout
                                 for index in row_indices])
        heldout_mask = np.asarray([int(index) in heldout for index in row_indices])
        if train_mask.sum() < 30 or heldout_mask.sum() == 0:
            continue
        for name in MODELS:
            model = make_model(name).fit(vectors[train_mask], utilities[train_mask])
            scores = np.full(len(vectors), -np.inf)
            scores[heldout_mask] = model.predict(vectors[heldout_mask])
            for threshold in THRESHOLDS:
                selected = decisions(rows, row_indices, candidate_indices,
                                     scores, heldout, threshold)
                utility, success, harm = realized_utility(rows, labels, selected,
                                                           heldout, harm_cost)
                result = totals[(name, threshold)]
                result[0] += utility
                result[1] += success
                result[2] += harm
            if by_answer_type:
                for answer_type in ("boolean", "color", "entity", "integer", "open_text"):
                    typed = {i for i in heldout if rows[i]["answer_type"] == answer_type}
                    for threshold in TYPE_THRESHOLDS:
                        selected = decisions(rows, row_indices, candidate_indices,
                                             scores, typed, threshold)
                        utility, success, harm = realized_utility(
                            rows, labels, selected, typed, harm_cost)
                        result = type_totals[(name, answer_type, threshold)]
                        result[0] += utility
                        result[1] += success
                        result[2] += harm
    best = max(((name, threshold) for name in MODELS for threshold in THRESHOLDS),
               key=lambda pair: (totals[pair][0], -totals[pair][2],
                                 pair[1], -MODELS.index(pair[0])))
    report = {f"{name}|{threshold}": totals[(name, threshold)]
              for name in MODELS for threshold in THRESHOLDS}
    if not by_answer_type:
        return best[0], best[1], report
    # Category-specific thresholds must earn at least two units of *inner-fold*
    # utility over the pooled threshold; otherwise retain the pooled choice.
    # This is fixed before outer-fold predictions, never fit on the held-out fold.
    chosen = {}
    for answer_type in ("boolean", "color", "entity", "integer", "open_text"):
        default = type_totals[(best[0], answer_type, best[1])][0]
        candidate = max(TYPE_THRESHOLDS,
                        key=lambda threshold: (type_totals[(best[0], answer_type, threshold)][0],
                                               -type_totals[(best[0], answer_type, threshold)][2],
                                               threshold))
        gain = type_totals[(best[0], answer_type, candidate)][0] - default
        chosen[answer_type] = candidate if gain >= 2.0 else best[1]
    report["by_answer_type"] = {
        "pooled": {"model": best[0], "threshold": best[1]},
        "selected": chosen,
        "inner_type_scores": {
            f"{kind}|{threshold}": type_totals[(best[0], kind, threshold)]
            for kind in chosen for threshold in TYPE_THRESHOLDS},
    }
    return best[0], chosen, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--proposals", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--confidence", type=Path,
                        help="Optional frozen label-free Qwen baseline token confidence")
    parser.add_argument("--harm-cost", type=float, default=1.25)
    parser.add_argument("--by-answer-type", action="store_true",
                        help="Nested, shrinkage-protected per-answer-type thresholds")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Refusing to overwrite gate run")
    if args.manifest.name != "development" or args.harm_cost < 1:
        raise ValueError("This script is development-only and harm-averse")
    qpath = args.manifest / "question_only.jsonl"
    lpath = args.manifest / "sealed_labels.jsonl"
    ppath = args.proposals / "proposals.jsonl"
    mf = json.loads((args.manifest / "manifest_summary.json").read_text())
    pf = json.loads((args.proposals / "PROPOSALS_FROZEN.json").read_text())
    if (digest(qpath) != mf["artifact_sha256"]["question_only.jsonl"] or
            digest(lpath) != mf["artifact_sha256"]["sealed_labels.jsonl"] or
            pf["question_only_sha256"] != digest(qpath) or
            pf["proposals_sha256"] != digest(ppath)):
        raise ValueError("Frozen inputs changed")
    confidence: dict[str, dict] = {}
    if args.confidence:
        cf = json.loads((args.confidence.parent / "QWEN_CONFIDENCE_FROZEN.json").read_text())
        if (digest(args.confidence) != cf["evidence_sha256"] or
                cf["question_only_sha256"] != digest(qpath) or
                cf["baseline_sha256"] != pf["baseline_sha256"] or
                cf["labels_opened"] is not False or cf["rows"] != 500):
            raise ValueError("Frozen confidence evidence mismatch")
        confidence = json.loads(args.confidence.read_text())
    questions, rows = load_jsonl(qpath), load_jsonl(ppath)
    if len(questions) != 500 or len(rows) != 500:
        raise ValueError("Expected 500 aligned development rows")
    for question, row in zip(questions, rows):
        if any(question[key] != row[key] for key in ("question_id", "image", "question")):
            raise ValueError("Question/proposal mismatch")
        if set(row) & {"label", "references", "ground_truth", "gold_answer", "category"}:
            raise ValueError("Label field leaked into proposals")
    labels = {row["question_id"]: row for row in load_jsonl(lpath)}
    if len(labels) != 500:
        raise ValueError("Development label mismatch")
    if confidence and set(confidence) != {row["question_id"] for row in rows}:
        raise ValueError("Confidence/question IDs differ")
    vectors, utilities, row_indices, candidate_indices = examples_for(
        rows, labels, args.harm_cost, confidence)
    oof, outer = [], []
    for fold in range(5):
        train_rows = {i for i, row in enumerate(rows) if image_fold(row["image"], 5) != fold}
        test_rows = set(range(len(rows))) - train_rows
        name, threshold, inner = choose_model(rows, labels, vectors, utilities,
                                               row_indices, candidate_indices,
                                               train_rows, args.harm_cost,
                                               args.by_answer_type)
        train_mask = np.asarray([int(index) in train_rows for index in row_indices])
        test_mask = np.asarray([int(index) in test_rows for index in row_indices])
        model = make_model(name).fit(vectors[train_mask], utilities[train_mask])
        scores = np.full(len(vectors), -np.inf)
        if test_mask.any():
            scores[test_mask] = model.predict(vectors[test_mask])
        selected = decisions(rows, row_indices, candidate_indices,
                             scores, test_rows, threshold)
        outer.append({"fold": fold, "model": name, "threshold": threshold,
                      "train_proposals": int(train_mask.sum()),
                      "heldout_questions": len(test_rows), "inner_scores": inner})
        for row_index in sorted(test_rows):
            row = rows[row_index]
            choice = selected.get(row_index)
            candidate = row["proposals"][choice["candidate_index"]] if choice else None
            oof.append({"question_id": row["question_id"], "image": row["image"],
                        "question": row["question"], "raw_answer": row["raw_answer"],
                        "calibrated_answer": candidate["answer"] if candidate else row["raw_answer"],
                        "decision_route": candidate["expert"] if candidate else "qwen_baseline",
                        "gate_expected_utility": choice["score"] if choice else 0,
                        "gate_fold": fold, "candidate_count": len(row["proposals"])})
    order = {row["question_id"]: index for index, row in enumerate(rows)}
    oof.sort(key=lambda row: order[row["question_id"]])
    if len(oof) != 500:
        raise ValueError("Incomplete outer-fold predictions")
    final_name, final_threshold, final_inner = choose_model(
        rows, labels, vectors, utilities, row_indices, candidate_indices,
        set(range(len(rows))), args.harm_cost, args.by_answer_type)
    final_model = make_model(final_name).fit(vectors, utilities)
    args.output.mkdir(parents=True, exist_ok=True)
    pred_path = args.output / "predictions.jsonl"
    with pred_path.open("x", encoding="utf-8") as stream:
        for row in oof:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    model_path = args.output / "gate_model.joblib"
    joblib.dump(final_model, model_path)
    spec_path = args.output / "gate_model.json"
    spec_path.write_text(json.dumps({
        "role": "native_selective_gate_development_only",
        "model": final_name, "threshold": final_threshold,
        "model_file_sha256": digest(model_path),
        "feature_names": ALL_FEATURE_NAMES,
        "by_answer_type": args.by_answer_type,
        "harm_cost": args.harm_cost, "outer_folds": outer,
        "final_inner_scores": final_inner,
        "question_only_sha256": digest(qpath), "development_labels_sha256": digest(lpath),
        "proposals_sha256": digest(ppath),
        "confidence_sha256": digest(args.confidence) if args.confidence else None,
        "warning": "No confirmation labels used; freeze only after development review.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "PREDICTIONS_FROZEN.json").write_text(json.dumps({
        "role": "development_image_group_out_of_fold_sklearn_gate",
        "question_only_sha256": digest(qpath), "proposals_sha256": digest(ppath),
        "development_labels_sha256": digest(lpath),
        "predictions_sha256": digest(pred_path), "gate_model_sha256": digest(spec_path),
        "folds": 5, "labels_opened_only_after_proposal_freeze": True,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"questions": len(oof), "proposals": len(vectors),
                      "accepted": sum(row["decision_route"] != "qwen_baseline" for row in oof),
                      "final_model": final_name, "final_threshold": final_threshold}), flush=True)


if __name__ == "__main__":
    main()
