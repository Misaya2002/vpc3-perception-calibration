"""Freeze label-free native-answer routes before any category audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--router-version", choices=("v1", "v2"), default="v1")
    args = parser.parse_args()
    module_name = "native_router" if args.router_version == "v1" else "native_router_v2"
    route_question = importlib.import_module(f"woodpecker_qwen.{module_name}").route_question
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "routes.jsonl"
    freeze_path = args.output / "ROUTES_FROZEN.json"
    if destination.exists() or freeze_path.exists():
        raise FileExistsError("Refusing to overwrite existing route artifacts")
    question_path = args.manifest / "question_only.jsonl"
    manifest = json.loads((args.manifest / "manifest_summary.json").read_text())
    if digest(question_path) != manifest["artifact_sha256"]["question_only.jsonl"]:
        raise ValueError("Manifest question hash mismatch")
    questions = [json.loads(line) for line in question_path.read_text().splitlines() if line]
    if len(questions) != manifest["questions"] or len({q["question_id"] for q in questions}) != len(questions):
        raise ValueError("Question count or uniqueness mismatch")
    module_path = Path(__file__).resolve().parents[1] / "woodpecker_qwen" / f"{module_name}.py"
    with destination.open("x", encoding="utf-8") as stream:
        for row in questions:
            result = {"question_id": row["question_id"], "image": row["image"],
                      "question": row["question"], **route_question(row["question"])}
            if set(result) & {"label", "references", "ground_truth", "gold_answer", "category"}:
                raise ValueError("Route unexpectedly contains label or source category")
            stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    freeze_path.write_text(json.dumps({
        "role": "native_question_only_router",
        "router_version": args.router_version,
        "questions": len(questions),
        "question_only_sha256": digest(question_path),
        "routes_sha256": digest(destination),
        "router_code_sha256": digest(module_path),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"questions": len(questions), "routes_sha256": digest(destination)}))


if __name__ == "__main__":
    main()
