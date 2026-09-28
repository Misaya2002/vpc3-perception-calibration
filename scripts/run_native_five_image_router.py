"""Resolve ambiguous native evidence routes using image+question, never labels.

The model is asked only which evidence source is needed, not to answer or
correct the original question.  A closed vocabulary and conservative fallback
prevent an unparseable response from silently choosing a tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from woodpecker_qwen.backends import OpenAICompatibleBackend


PROMPT = (
    "You are selecting evidence tools, not answering the question. Inspect the "
    "image and question. Which single visual evidence type is most necessary "
    "to answer reliably? Reply with exactly one token from this list: "
    "OCR, SPATIAL, COLOR, COUNT, PRESENCE, UNCLEAR. "
    "OCR means reading words, printed numbers, names, brands, signs or labels "
    "in the image, even if the question does not say 'text'. SPATIAL means "
    "locating objects or their geometric relations. COLOR means identifying "
    "an object's color. COUNT means counting instances. PRESENCE means deciding "
    "whether an object exists. If the required evidence is unclear, say "
    "UNCLEAR. Do not give the answer to the question.\nQuestion: {question}"
)
TASKS = {
    "OCR": ("ocr", "open_text"),
    "SPATIAL": ("spatial_relation", "entity"),
    "COLOR": ("color", "color"),
    "COUNT": ("count", "integer"),
    "PRESENCE": ("object_presence", "boolean"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def parse_choice(text: str) -> str:
    value = text.strip().upper().strip(" .!?`\n\r\t")
    return value if value in TASKS or value == "UNCLEAR" else "UNCLEAR"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--question-router", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-name", default="Qwen2.5-VL-7B-Instruct")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "routes.jsonl"
    freeze_path = args.output / "ROUTES_FROZEN.json"
    if freeze_path.exists():
        raise FileExistsError("Image routes already frozen")
    qpath = args.manifest / "question_only.jsonl"
    qsummary = json.loads((args.manifest / "manifest_summary.json").read_text())
    parent_freeze = json.loads((args.question_router / "ROUTES_FROZEN.json").read_text())
    parent_path = args.question_router / "routes.jsonl"
    if (digest(qpath) != qsummary["artifact_sha256"]["question_only.jsonl"]
            or parent_freeze["question_only_sha256"] != digest(qpath)
            or parent_freeze["routes_sha256"] != digest(parent_path)):
        raise ValueError("Parent route or question hash mismatch")
    questions = load_jsonl(qpath)
    parents = load_jsonl(parent_path)
    if len(questions) != len(parents) or len(questions) != qsummary["questions"]:
        raise ValueError("Route cardinality mismatch")
    for question, parent in zip(questions, parents):
        if any(question[key] != parent[key] for key in ("question_id", "image", "question")):
            raise ValueError("Parent route/question alignment mismatch")
    completed = load_jsonl(destination) if destination.exists() else []
    if len(completed) > len(parents):
        raise ValueError("Resumed output exceeds input")
    for result, parent in zip(completed, parents):
        if any(result[key] != parent[key] for key in ("question_id", "image", "question")):
            raise ValueError("Resumed route/input mismatch")
    backend = OpenAICompatibleBackend(args.base_url, "EMPTY", args.model_name)
    with destination.open("a", encoding="utf-8") as stream:
        for index, parent in enumerate(parents[len(completed):], len(completed) + 1):
            result = dict(parent)
            result["image_router_choice"] = "SKIPPED"
            if parent["perceptual_task"] == "ambiguous":
                image = args.manifest / parent["image"]
                if not image.is_file():
                    raise FileNotFoundError(image)
                text = backend.generate(str(image), PROMPT.format(question=parent["question"]), max_new_tokens=16)
                choice = parse_choice(text)
                result["image_router_choice"] = choice
                result["image_router_raw"] = text
                if choice in TASKS:
                    task, answer_type = TASKS[choice]
                    result.update(perceptual_task=task, answer_type=answer_type,
                                  parse_status="image_route_" + task,
                                  tool_eligible=True, source="question_and_image_only")
            stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            if index % 25 == 0:
                print(json.dumps({"visited": index, "total": len(parents)}), flush=True)
    final = load_jsonl(destination)
    if len(final) != len(parents):
        raise ValueError("Incomplete image routes")
    freeze_path.write_text(json.dumps({
        "role": "native_question_plus_image_router",
        "questions": len(final),
        "question_only_sha256": digest(qpath),
        "parent_routes_sha256": digest(parent_path),
        "routes_sha256": digest(destination),
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "script_sha256": digest(Path(__file__)),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"questions": len(final), "routes_sha256": digest(destination)}), flush=True)


if __name__ == "__main__":
    main()
