"""Build native-answer expert proposals from frozen, label-free inputs.

This stage does not make a correction decision or open sealed labels. Missing
expert evidence results in no proposal, never a fabricated prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from woodpecker_qwen.native_experts import collect_proposals

FORBIDDEN = {"label", "references", "ground_truth", "gold_answer", "category"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def evidence_index(path: Path | None) -> dict:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Evidence must be image-keyed JSON object: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--routes", required=True, type=Path)
    for name in ("presence", "color", "siglip2_color", "count_anything", "countgd", "spatial", "ocr", "qwen3_vl", "qwen3_vl_32b", "blip_vqa"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path)
    parser.add_argument("--ocr-phrase-exploratory", action="store_true",
                        help="Failed development-only OCR phrase branch; never enabled by default")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    freeze_path = args.output.with_name("PROPOSALS_FROZEN.json")
    if freeze_path.exists():
        raise FileExistsError(f"Refusing to overwrite {freeze_path}")
    qpath = args.manifest / "question_only.jsonl"
    bpath = args.baseline / "baseline.jsonl"
    rpath = args.routes / "routes.jsonl"
    mf = json.loads((args.manifest / "manifest_summary.json").read_text(encoding="utf-8"))
    bf = json.loads((args.baseline / "BASELINE_FROZEN.json").read_text(encoding="utf-8"))
    rf = json.loads((args.routes / "ROUTES_FROZEN.json").read_text(encoding="utf-8"))
    if (digest(qpath) != mf["artifact_sha256"]["question_only.jsonl"] or
            bf["question_only_sha256"] != digest(qpath) or
            bf["baseline_sha256"] != digest(bpath) or
            rf["question_only_sha256"] != digest(qpath) or
            rf["routes_sha256"] != digest(rpath)):
        raise ValueError("Frozen manifest/baseline/routes hash mismatch")
    questions, baselines, routes = jsonl(qpath), jsonl(bpath), jsonl(rpath)
    if len(questions) != mf["questions"] or len(baselines) != len(questions) or len(routes) != len(questions):
        raise ValueError("Input cardinality mismatch")
    evidence_paths = {name: getattr(args, name) for name in
                      ("presence", "color", "siglip2_color", "count_anything", "countgd", "spatial", "ocr", "qwen3_vl", "qwen3_vl_32b", "blip_vqa")}
    for name, path in evidence_paths.items():
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"Missing {name} evidence: {path}")
    evidence = {name: evidence_index(path) for name, path in evidence_paths.items()}
    counts = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for question, baseline, route in zip(questions, baselines, routes):
            for item in (question, baseline, route):
                if FORBIDDEN & item.keys():
                    raise ValueError("Forbidden label/source category field in predictor input")
                if any(item.get(key) != question.get(key) for key in ("question_id", "image", "question")):
                    raise ValueError("Question/baseline/route mismatch")
            candidates = collect_proposals(question, baseline, route, evidence,
                                           include_ocr_phrase=args.ocr_phrase_exploratory)
            counts["questions"] += 1
            counts["with_proposal"] += bool(candidates)
            counts[f"task_{route['perceptual_task']}"] += 1
            for candidate in candidates:
                counts[f"expert_{candidate['expert']}"] += 1
            stream.write(json.dumps({
                "question_id": question["question_id"], "image": question["image"],
                "question": question["question"], "raw_answer": baseline["raw_answer"],
                "answer_type": route["answer_type"], "perceptual_task": route["perceptual_task"],
                "parse_status": route["parse_status"], "proposals": candidates,
            }, ensure_ascii=False, sort_keys=True) + "\n")
    freeze_path.write_text(json.dumps({
        "role": "label_free_native_expert_proposals",
        "question_only_sha256": digest(qpath), "baseline_sha256": digest(bpath),
        "routes_sha256": digest(rpath),
        "evidence_sha256": {name: digest(path) if path else None
                            for name, path in evidence_paths.items()},
        "code_sha256": {"builder": digest(Path(__file__)),
                         "native_experts": digest(ROOT / "woodpecker_qwen/native_experts.py")},
        "proposals_sha256": digest(args.output),
        "counts": dict(counts), "labels_opened": False,
        "no_evidence_no_proposal": True,
        "ocr_phrase_exploratory": args.ocr_phrase_exploratory,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": dict(counts), "sha256": digest(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
