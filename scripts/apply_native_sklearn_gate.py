"""Apply a frozen native-answer gate without reading any reference answers."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_native_sklearn_gate import ALL_FEATURE_NAMES, feature_vector
from woodpecker_qwen.native_gate import compact_evidence, public_observations


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True, type=Path)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--confidence", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite predictions")
    ppath = args.proposals / "proposals.jsonl"
    pf = json.loads((args.proposals / "PROPOSALS_FROZEN.json").read_text())
    if digest(ppath) != pf["proposals_sha256"] or pf["labels_opened"] is not False:
        raise ValueError("Proposals are not frozen and label-free")
    spec_path = args.gate / "gate_model.json"
    spec = json.loads(spec_path.read_text())
    model_path = args.gate / "gate_model.joblib"
    runtime_names = list(ALL_FEATURE_NAMES)
    trained_names = spec.get("feature_names")
    if (not isinstance(trained_names, list) or
            len(set(trained_names)) != len(trained_names) or
            set(trained_names) - set(runtime_names) or
            digest(model_path) != spec["model_file_sha256"]):
        raise ValueError("Gate artifact or feature schema changed")
    feature_indices = [runtime_names.index(name) for name in trained_names]
    trained_experts = {name.removeprefix("expert_") for name in trained_names
                       if name.startswith("expert_") and name != "expert_score"}
    manifest_path = args.gate / "artifact_manifest.json"
    if manifest_path.is_file():
        artifact = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (artifact["sha256"]["gate_model.json"] != digest(spec_path) or
                artifact["sha256"]["gate_model.joblib"] != digest(model_path)):
            raise ValueError("Packaged gate artifact hash mismatch")
        if "proposal_support" in artifact:
            trained_experts &= {name for name, count in artifact["proposal_support"].items()
                                if int(count) > 0}
    confidence = {}
    if args.confidence:
        cf = json.loads((args.confidence.parent / "QWEN_CONFIDENCE_FROZEN.json").read_text())
        if (digest(args.confidence) != cf["evidence_sha256"] or
                cf["question_only_sha256"] != pf["question_only_sha256"] or
                cf["baseline_sha256"] != pf["baseline_sha256"] or
                cf["labels_opened"] is not False):
            raise ValueError("Confidence evidence changed or leaks labels")
        confidence = json.loads(args.confidence.read_text())
    if bool(spec["confidence_sha256"]) != bool(args.confidence):
        raise ValueError("Confidence input must match gate training schema")
    model = joblib.load(model_path)
    rows = jsonl(ppath)
    output = []
    accepted = 0
    for row in rows:
        scoring_row = {**row, "proposals": [candidate for candidate in row.get("proposals", [])
                                            if candidate["expert"] in trained_experts]}
        candidates = [candidate for candidate in scoring_row["proposals"]
                      if (str(candidate["answer"]).casefold() != str(row["raw_answer"]).casefold()
                          and candidate["expert"] in trained_experts)]
        chosen = None
        chosen_score = float(spec["threshold"])
        scored_candidates = []
        if candidates:
            vectors = np.stack([feature_vector(scoring_row, candidate,
                                               confidence.get(row["question_id"]))[feature_indices]
                                for candidate in candidates])
            scores = model.predict(vectors)
            for candidate, score in zip(candidates, scores):
                scored_candidates.append((candidate, float(score)))
                if float(score) > chosen_score:
                    chosen, chosen_score = candidate, float(score)
        if chosen is not None:
            selected_evidence = {
                "status": "used_for_correction",
                "source": chosen["expert"],
                "source_score": float(chosen["score"]),
                "source_score_semantics": chosen.get("score_semantics"),
                "candidate_answer": chosen["answer"],
                "observations": public_observations(chosen.get("details")),
                "gate_expected_utility": float(chosen_score),
                "gate_score_semantics": "learned incremental utility; not a correctness probability",
                "threshold": float(spec["threshold"]),
                "reason": "highest_scoring_candidate_above_threshold",
            }
        elif scored_candidates:
            rejected, rejected_score = max(scored_candidates, key=lambda item: item[1])
            selected_evidence = {
                "status": "candidate_not_selected",
                "source": rejected["expert"],
                "source_score": float(rejected["score"]),
                "source_score_semantics": rejected.get("score_semantics"),
                "candidate_answer": rejected["answer"],
                "observations": public_observations(rejected.get("details")),
                "gate_expected_utility": rejected_score,
                "gate_score_semantics": "learned incremental utility; not a correctness probability",
                "threshold": float(spec["threshold"]),
                "reason": "below_acceptance_threshold",
            }
        else:
            selected_evidence = {"status": "no_alternative_candidate", "source": None,
                                 "observations": {}, "reason": "no_expert_disagreement"}
        final_answer = chosen["answer"] if chosen else row["raw_answer"]
        accepted += int(chosen is not None)
        output.append({"request_id": row["question_id"],
                       "original_answer": row["raw_answer"],
                       "calibrated_answer": final_answer,
                       "changed": str(final_answer).strip().casefold() != str(row["raw_answer"]).strip().casefold(),
                       "evidence": compact_evidence(selected_evidence)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.output_dir / "predictions.jsonl"
    with pred_path.open("x", encoding="utf-8") as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (args.output_dir / "PREDICTIONS_FROZEN.json").write_text(json.dumps({
        "role": "vpc3_native_answer_gate_application",
        "output_schema": "vpc3.answer-evidence.v2",
        "method_version": "VPC3-native-v1",
        "question_only_sha256": pf["question_only_sha256"],
        "proposals_sha256": digest(ppath), "gate_model_sha256": digest(spec_path),
        "predictions_sha256": digest(pred_path), "questions": len(rows),
        "accepted": accepted,
        "labels_opened": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"questions": len(rows), "accepted": accepted}), flush=True)


if __name__ == "__main__":
    main()
