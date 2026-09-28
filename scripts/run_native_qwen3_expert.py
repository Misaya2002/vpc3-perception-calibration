"""Generate label-free native-answer Qwen3-VL proposals with resumable output.

The model sees only an image and its question. It never sees Qwen2.5's answer,
dataset category, reference answers, or the sealed confirmation labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from woodpecker_qwen.backends import OpenAICompatibleBackend
from woodpecker_qwen.native_experts import normalize_answer

PROMPTS = {
    "boolean": "Answer the image question with exactly Yes or No. No explanation.\nQuestion: {question}",
    "integer": "Count what the image question asks. Answer with exactly one nonnegative integer. No explanation.\nQuestion: {question}",
    "color": "Answer the image question with exactly one basic color word. No explanation.\nQuestion: {question}",
    "entity": "Answer the image question with the shortest object or person name. No explanation.\nQuestion: {question}",
    "open_text": "Read the image and answer the question with only the requested visible text. No explanation.\nQuestion: {question}",
}
FORBIDDEN = {"label", "references", "ground_truth", "gold_answer", "category"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    args = parser.parse_args()
    final = args.output_dir / "qwen3_vl_evidence.json"
    frozen = args.output_dir / "QWEN3_EXPERT_FROZEN.json"
    partial = args.output_dir / "qwen3_vl_partial.jsonl"
    if final.exists() or frozen.exists():
        raise FileExistsError("Expert evidence is already frozen")
    qpath = args.manifest / "question_only.jsonl"
    rpath = args.routes / "routes.jsonl"
    manifest = json.loads((args.manifest / "manifest_summary.json").read_text(encoding="utf-8"))
    routes_frozen = json.loads((args.routes / "ROUTES_FROZEN.json").read_text(encoding="utf-8"))
    if (digest(qpath) != manifest["artifact_sha256"]["question_only.jsonl"] or
            digest(qpath) != routes_frozen["question_only_sha256"] or
            digest(rpath) != routes_frozen["routes_sha256"]):
        raise ValueError("Frozen manifest/routes hashes mismatch")
    questions, routes = read_jsonl(qpath), read_jsonl(rpath)
    if len(questions) != manifest["questions"] or len(questions) != len(routes):
        raise ValueError("Input cardinality mismatch")
    expected = {}
    for question, route in zip(questions, routes):
        if FORBIDDEN & (question.keys() | route.keys()):
            raise ValueError("Label or source category leaked into inference inputs")
        if any(question.get(key) != route.get(key) for key in ("question_id", "image", "question")):
            raise ValueError("Question and route mismatch")
        if route["answer_type"] not in PROMPTS:
            raise ValueError("Unknown answer type")
        key = str(question["question_id"])
        if key in expected:
            raise ValueError("Duplicate question id")
        expected[key] = (question, route)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed = {}
    if partial.exists():
        for row in read_jsonl(partial):
            key = str(row["question_id"])
            if key in completed or key not in expected:
                raise ValueError("Unexpected or duplicated resumed row")
            question, route = expected[key]
            if (row["question"], row["image"], row["answer_type"], row["model_revision"]) != (
                question["question"], question["image"], route["answer_type"], args.model_revision
            ):
                raise ValueError("Resumed row does not match frozen inputs")
            completed[key] = row
    backend = OpenAICompatibleBackend(args.base_url, "EMPTY", args.model_name)
    with partial.open("a", encoding="utf-8") as stream:
        for index, question in enumerate(questions, 1):
            key = str(question["question_id"])
            if key in completed:
                continue
            _, route = expected[key]
            image = args.manifest / question["image"]
            if not image.is_file():
                raise FileNotFoundError(image)
            answer_type = route["answer_type"]
            response = backend.generate(str(image), PROMPTS[answer_type].format(question=question["question"]),
                                        max_new_tokens=48)
            # Reject invalid type instead of smuggling an answer through a permissive parser.
            first = response.strip().splitlines()[0].strip().strip(" .") if response.strip() else ""
            answer = normalize_answer(first, answer_type)
            row = {"question_id": key, "image": question["image"], "question": question["question"],
                   "answer_type": answer_type, "raw_text": response, "answer": answer,
                   "status": "complete" if answer is not None else "invalid_output",
                   "model_revision": args.model_revision, "model_name": args.model_name}
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            if index % 25 == 0:
                print(json.dumps({"visited": index, "total": len(questions)}), flush=True)
    rows = read_jsonl(partial)
    if len(rows) != len(questions) or {str(r["question_id"]) for r in rows} != set(expected):
        raise ValueError("Incomplete expert inference; refusing to freeze")
    indexed = {str(row["question_id"]): row for row in rows}
    final.write_text(json.dumps(indexed, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    freeze = {"role": "label_free_competing_vlm_native_proposals",
              "model_name": args.model_name, "model_revision": args.model_revision,
              "question_only_sha256": digest(qpath), "routes_sha256": digest(rpath),
              "prompt_sha256": hashlib.sha256(json.dumps(PROMPTS, sort_keys=True).encode()).hexdigest(),
              "partial_sha256": digest(partial), "evidence_sha256": digest(final),
              "rows": len(rows), "valid": sum(row["answer"] is not None for row in rows),
              "labels_opened": False}
    frozen.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(freeze, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
