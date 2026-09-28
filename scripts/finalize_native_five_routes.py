"""Freeze answer-type-safe routes after question and image evidence routing.

The image model may choose which tool to consult, but it may not convert the
question's answer space.  Incompatible tool routes abstain.  No source labels
or candidate answers are inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--question-router", required=True, type=Path)
    parser.add_argument("--image-router", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "routes.jsonl"
    freeze_path = args.output / "ROUTES_FROZEN.json"
    if destination.exists() or freeze_path.exists():
        raise FileExistsError("Final routes already exist")
    question_path = args.manifest / "question_only.jsonl"
    manifest = json.loads((args.manifest / "manifest_summary.json").read_text())
    question_hash = digest(question_path)
    if question_hash != manifest["artifact_sha256"]["question_only.jsonl"]:
        raise ValueError("Question manifest changed")
    parent_paths = [args.question_router / "routes.jsonl", args.image_router / "routes.jsonl"]
    for directory, path in zip((args.question_router, args.image_router), parent_paths):
        freeze = json.loads((directory / "ROUTES_FROZEN.json").read_text())
        if freeze["question_only_sha256"] != question_hash or freeze["routes_sha256"] != digest(path):
            raise ValueError("Input route changed")
    questions = rows(question_path)
    question_routes, image_routes = map(rows, parent_paths)
    if len(questions) != manifest["questions"] or len(questions) != len(question_routes) or len(questions) != len(image_routes):
        raise ValueError("Route cardinality mismatch")
    with destination.open("x", encoding="utf-8") as stream:
        for question, basic, visual in zip(questions, question_routes, image_routes):
            if any(question[key] != basic[key] or question[key] != visual[key]
                   for key in ("question_id", "image", "question")):
                raise ValueError("Route/question alignment mismatch")
            result = dict(visual)
            result["answer_type"] = basic["answer_type"]
            result["answer_type_source"] = "question_text_only"
            result["evidence_route_source"] = visual["source"]
            if basic["answer_type"] == "open_text" and visual["perceptual_task"] in {"count", "object_presence"}:
                result["tool_eligible"] = False
                result["parse_status"] = "image_route_incompatible_with_open_answer"
                result["constraint"] = "preserve_native_open_answer_space"
            stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    freeze_path.write_text(json.dumps({
        "role": "native_answer_type_safe_evidence_routes",
        "questions": len(questions),
        "question_only_sha256": question_hash,
        "question_routes_sha256": digest(parent_paths[0]),
        "image_routes_sha256": digest(parent_paths[1]),
        "routes_sha256": digest(destination),
        "script_sha256": digest(Path(__file__)),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"questions": len(questions), "routes_sha256": digest(destination)}))


if __name__ == "__main__":
    main()
