"""Fair-format native-answer Qwen baseline; development before confirmation.

Unlike the exploratory V1 prompt, this asks for an explicit yes/no answer to
yes/no questions while keeping count, color, relation and OCR answers native.
No category, dataset source or ground-truth answer enters the prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from run_native_five_baseline import digest, short_answer
from woodpecker_qwen.backends import OpenAICompatibleBackend


PROMPT = (
    "Answer the question about the image. If it asks for a yes/no judgment, "
    "answer exactly Yes or No. Otherwise give only the shortest direct answer "
    "(such as a number, color, object name, or text), without explanation.\n"
    "Question: {question}"
)


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
            if row["question_id"] in completed:
                raise ValueError("Duplicate resume answer")
            completed[row["question_id"]] = row
    expected = {row["question_id"]: row for row in questions}
    for key, row in completed.items():
        source = expected.get(key)
        if source is None or (row["image"], row["question"]) != (source["image"], source["question"]):
            raise ValueError("Resume answer differs from frozen question")
    backend = OpenAICompatibleBackend(args.base_url, "EMPTY", args.model_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as stream:
        for index, row in enumerate(questions, 1):
            if row["question_id"] in completed:
                continue
            image = args.manifest / row["image"]
            if not image.is_file():
                raise FileNotFoundError(image)
            raw_text = backend.generate(str(image), PROMPT.format(question=row["question"]), max_new_tokens=48)
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
              "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
              "rows": len(final), "role": "label_free_native_answer_baseline_fair_format_v2"}
    (args.output.parent / "BASELINE_FROZEN.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(freeze), flush=True)


if __name__ == "__main__":
    main()
