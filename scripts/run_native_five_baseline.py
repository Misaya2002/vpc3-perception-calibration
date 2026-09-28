"""Generate label-free native short answers using the fixed Qwen backend.

This is a baseline generator, not a correction method.  It never opens source
labels or category metadata and does not convert questions to yes/no.
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


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def short_answer(text: str) -> str:
    first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
    first_line = re.sub(r"^(?:answer|final answer)\s*:\s*", "", first_line, flags=re.I)
    return first_line.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-name", default="Qwen2.5-VL-7B-Instruct")
    args = parser.parse_args()
    if args.output.name != "baseline.jsonl":
        raise ValueError("Output must be named baseline.jsonl")
    if (args.output.parent / "BASELINE_FROZEN.json").exists():
        raise FileExistsError("Baseline already frozen")
    summary = json.loads((args.manifest / "manifest_summary.json").read_text())
    question_path = args.manifest / "question_only.jsonl"
    question_hash = digest(question_path)
    if question_hash != summary["artifact_sha256"]["question_only.jsonl"]:
        raise ValueError("Frozen question-only manifest changed")
    questions = [json.loads(line) for line in question_path.read_text(encoding="utf-8").splitlines() if line]
    if len(questions) != summary["questions"] or len({row["question_id"] for row in questions}) != len(questions):
        raise ValueError("Question count or uniqueness mismatch")
    completed = {}
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            key = row["question_id"]
            if key in completed:
                raise ValueError(f"Duplicate baseline result: {key}")
            completed[key] = row
    expected = {row["question_id"]: row for row in questions}
    for key, row in completed.items():
        question = expected.get(key)
        if question is None or (row["image"], row["question"]) != (question["image"], question["question"]):
            raise ValueError(f"Resume record does not match frozen input: {key}")
    backend = OpenAICompatibleBackend(args.base_url, "EMPTY", args.model_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as stream:
        for index, row in enumerate(questions, 1):
            if row["question_id"] in completed:
                continue
            image = args.manifest / row["image"]
            if not image.is_file():
                raise FileNotFoundError(image)
            prompt = ("Answer the question about the image. Give only the shortest direct "
                      "answer, without explanation or a full sentence.\nQuestion: " + row["question"])
            raw_text = backend.generate(str(image), prompt, max_new_tokens=48)
            result = {"question_id": row["question_id"], "image": row["image"],
                      "question": row["question"], "raw_text": raw_text,
                      "raw_answer": short_answer(raw_text)}
            stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            if index % 25 == 0:
                print(json.dumps({"visited": index, "total": len(questions)}), flush=True)
    final = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line]
    if len(final) != len(questions) or len({row["question_id"] for row in final}) != len(questions):
        raise ValueError("Incomplete baseline; no freeze written")
    freeze = {"question_only_sha256": question_hash, "baseline_sha256": digest(args.output),
              "rows": len(final), "role": "label_free_native_answer_baseline"}
    (args.output.parent / "BASELINE_FROZEN.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(freeze), flush=True)


if __name__ == "__main__":
    main()
