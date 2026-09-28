"""Apply a frozen native-answer multi-expert gate without reading labels."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_native_five import digest, load_jsonl
from woodpecker_qwen.native_gate import compact_evidence, select


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--proposals", required=True, type=Path)
    parser.add_argument("--gate-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Refusing to overwrite native gate output")
    qpath = args.manifest / "question_only.jsonl"
    ppath = args.proposals / "proposals.jsonl"
    mf = json.loads((args.manifest / "manifest_summary.json").read_text(encoding="utf-8"))
    pf = json.loads((args.proposals / "PROPOSALS_FROZEN.json").read_text(encoding="utf-8"))
    if (digest(qpath) != mf["artifact_sha256"]["question_only.jsonl"] or
            digest(qpath) != pf["question_only_sha256"] or
            digest(ppath) != pf["proposals_sha256"]):
        raise ValueError("Manifest/proposal freeze mismatch; labels remain sealed")
    model = json.loads(args.gate_model.read_text(encoding="utf-8"))
    if model.get("role") != "native_answer_development_gate_for_future_freeze":
        raise ValueError("Unexpected gate model artifact")
    questions, proposals = load_jsonl(qpath), load_jsonl(ppath)
    if len(questions) != mf["questions"] or len(proposals) != len(questions):
        raise ValueError("Question/proposal cardinality mismatch")
    counts = Counter()
    results = []
    for question, row in zip(questions, proposals):
        if any(question[key] != row[key] for key in ("question_id", "image", "question")):
            raise ValueError("Question/proposal order mismatch")
        if set(row) & {"label", "references", "ground_truth", "gold_answer", "category"}:
            raise ValueError("Proposal has forbidden evaluation/source label")
        chosen = select(row, model)
        counts[chosen["expert"]] += 1
        results.append({
            "request_id": row["question_id"],
            "original_answer": row["raw_answer"],
            "calibrated_answer": chosen["answer"],
            "changed": chosen["answer"].strip().casefold() != row["raw_answer"].strip().casefold(),
            "evidence": compact_evidence(chosen.get("evidence")),
        })
    args.output.mkdir(parents=True, exist_ok=True)
    pred_path = args.output / "predictions.jsonl"
    with pred_path.open("x", encoding="utf-8") as stream:
        for item in results:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    freeze = {
        "role": "vpc3_native_answer_frozen_gate_application",
        "output_schema": "vpc3.answer-evidence.v2",
        "method_version": "VPC3-native-v1",
        "question_only_sha256": digest(qpath), "proposals_sha256": digest(ppath),
        "gate_model_sha256": digest(args.gate_model),
        "predictions_sha256": digest(pred_path),
        "code_sha256": {"apply": digest(Path(__file__)),
                         "native_gate": digest(ROOT / "woodpecker_qwen/native_gate.py")},
        "rows": len(results), "decision_counts": dict(counts),
        "evaluation_labels_opened": False,
    }
    (args.output / "PREDICTIONS_FROZEN.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(results), "decision_counts": dict(counts),
                      "predictions_sha256": freeze["predictions_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
