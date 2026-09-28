"""Freeze native-answer five-category development and confirmation manifests.

This builder never reads model predictions.  Questions retain their source
wording and answer space; in particular, counting and OCR are not converted to
yes/no verification.  Inference must read only question_only.jsonl, never the
sealed labels.  Image overlap auditing remains a separate mandatory gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path


SPATIAL = re.compile(
    r"\b(?:left|right|above|below|behind|in front of|next to|on top of|"
    r"under|near|beside|inside|outside|over|across from)\b", re.I
)
COLOR = re.compile(r"\bcolou?r\b", re.I)
COUNT_IMAGE = re.compile(r"VG_100K(?:_2)?/[0-9]+\.jpg\Z")
PROTOCOL = "native-five-v1-sha256-ranked-2026-09-26"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rank(category: str, identifier: str) -> bytes:
    return hashlib.sha256(f"{PROTOCOL}:{category}:{identifier}".encode()).digest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def exclusion_ids(path: Path, failed_audits: list[Path]) -> dict[str, set[str]]:
    excluded: dict[str, set[str]] = {"gqa": set(), "tallyqa": set(), "textvqa": set()}
    for row in read_jsonl(path):
        source = row["source"]
        image_id = str(row["image_id"])
        if source == "gqa":
            excluded["gqa"].add(image_id)
        elif source == "tallyqa":
            excluded["tallyqa"].add(image_id)
        elif source == "textocr":
            # TextOCR and TextVQA both use Open Images image IDs.
            excluded["textvqa"].add(image_id)
    for audit_path in failed_audits:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("pass") is not False:
            raise ValueError(f"Only failed pre-inference pixel audits may exclude candidates: {audit_path}")
        images = set(audit.get("historical_matches", {})) | set(audit.get("within_candidate_matches", {}))
        for image in images:
            if image.startswith("images/gqa/") and image.endswith(".jpg"):
                excluded["gqa"].add(Path(image).stem)
            elif image.startswith("images/tallyqa/"):
                excluded["tallyqa"].add(image.removeprefix("images/tallyqa/"))
            elif image.startswith("images/textvqa/") and image.endswith(".jpg"):
                excluded["textvqa"].add(Path(image).stem)
            else:
                raise ValueError(f"Unrecognized candidate image in audit: {image}")
    return excluded


def gqa_pools(path: Path, excluded: set[str]) -> dict[str, list[dict]]:
    import pyarrow.parquet as pq

    rows = pq.read_table(
        path, columns=["id", "imageId", "question", "answer", "types", "semanticStr"]
    ).to_pylist()
    pools: dict[str, list[dict]] = {name: [] for name in ("object_presence", "color", "spatial_relation")}
    for row in rows:
        image_id = str(row["imageId"])
        if image_id in excluded:
            continue
        typ = row["types"]
        answer = str(row["answer"]).strip()
        question = str(row["question"]).strip()
        if not answer or not question:
            continue
        category = None
        if (typ["semantic"] == "obj" and typ["detailed"].startswith("exist")
                and answer.lower() in {"yes", "no"}
                and question.lower().startswith(("is there", "are there"))):
            category = "object_presence"
        elif (typ["semantic"] == "attr" and typ["structural"] == "query"
              and COLOR.search(question) and answer.lower() not in {"yes", "no"}):
            category = "color"
        elif (typ["semantic"] == "rel" and typ["structural"] == "query"
              and SPATIAL.search(question) and answer.lower() not in {"yes", "no"}):
            category = "spatial_relation"
        if category:
            pools[category].append({
                "question_id": f"gqa-val:{row['id']}",
                "image": f"images/gqa/{image_id}.jpg",
                "image_id": image_id,
                "question": question,
                "references": [answer],
                "category": category,
                "source": "GQA-val-balanced-native",
            })
    return pools


def select_unique(pool: list[dict], count: int, category: str, used: set[str]) -> list[dict]:
    selected = []
    for row in sorted(pool, key=lambda item: rank(category, item["question_id"])):
        if row["image_id"] in used:
            continue
        used.add(row["image_id"])
        selected.append(row)
        if len(selected) == count:
            return selected
    raise ValueError(f"{category}: only {len(selected)} unique eligible images, need {count}")


def tally_pool(path: Path, excluded: set[str]) -> list[dict]:
    with zipfile.ZipFile(path) as archive:
        rows = json.loads(archive.read("test.json"))
    result = []
    for row in rows:
        image = str(row.get("image", ""))
        answer = row.get("answer")
        if (not COUNT_IMAGE.fullmatch(image) or image in excluded
                or type(answer) is not int or answer < 0
                or not str(row.get("question", "")).strip()):
            continue
        result.append({
            "question_id": f"tallyqa-test:{row['question_id']}",
            "image": f"images/tallyqa/{image}",
            "image_id": image,
            "question": str(row["question"]).strip(),
            "references": [str(answer)],
            "category": "count",
            "source": "TallyQA-test-native",
            "issimple": row.get("issimple"),
        })
    return result


def textvqa_pool(path: Path, excluded: set[str]) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))["data"]
    result = []
    for row in rows:
        image_id = str(row["image_id"])
        answers = [str(answer).strip() for answer in row.get("answers", [])]
        if image_id in excluded or not answers or not str(row.get("question", "")).strip():
            continue
        result.append({
            "question_id": f"textvqa-val:{row['question_id']}",
            "image": f"images/textvqa/{image_id}.jpg",
            "image_id": image_id,
            "question": str(row["question"]).strip(),
            "references": answers,
            "category": "ocr",
            "source": "TextVQA-val-native",
        })
    return result


def image_request(row: dict) -> dict:
    request = {"image": row["image"], "image_id": row["image_id"]}
    if row["source"].startswith("GQA"):
        request["source"] = "gqa"
    elif row["source"].startswith("TallyQA"):
        request.update(source="tallyqa", url=f"https://cs.stanford.edu/people/rak248/{row['image_id']}")
    else:
        request.update(source="textvqa", url=f"https://open-images-dataset.s3.amazonaws.com/train/{row['image_id']}.jpg")
    return request


def freeze_split(root: Path, name: str, rows: list[dict], source_hashes: dict[str, str],
                 exclusions_hash: str, failed_audit_hashes: dict[str, str]) -> None:
    split = root / name
    split.mkdir()
    question_only = [{key: row[key] for key in ("question_id", "image", "question")} for row in rows]
    requests = {row["image"]: image_request(row) for row in rows}
    write_jsonl(split / "question_only.jsonl", question_only)
    write_jsonl(split / "sealed_labels.jsonl", rows)
    write_jsonl(split / "image_requests.jsonl", [requests[key] for key in sorted(requests)])
    summary = {
        "protocol": PROTOCOL,
        "status": "manifest_frozen_no_model_predictions_or_image_overlap_audit_yet",
        "split": name,
        "questions": len(rows),
        "images": len(requests),
        "categories": dict(Counter(row["category"] for row in rows)),
        "answer_spaces": {
            "object_presence": "native yes/no",
            "color": "native open color name",
            "spatial_relation": "native open object named by spatial relation",
            "count": "native integer",
            "ocr": "native TextVQA free-text answer",
        },
        "source_sha256": source_hashes,
        "excluded_binary_candidate_requests_sha256": exclusions_hash,
        "pre_inference_failed_pixel_audit_sha256": failed_audit_hashes,
        "artifact_sha256": {filename: digest(split / filename) for filename in
                            ("question_only.jsonl", "sealed_labels.jsonl", "image_requests.jsonl")},
        "limitations": [
            "Predeclared multi-source composite, not one official benchmark.",
            "GQA val images have been used in other misaya projects; not globally untouched.",
            "TextVQA official soft accuracy must be reported separately from the composite's fixed exact/consensus correctness.",
            "Image byte/perceptual overlap gate and actual image availability are mandatory before inference.",
        ],
    }
    (split / "manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gqa", type=Path, required=True)
    parser.add_argument("--tallyqa", type=Path, required=True)
    parser.add_argument("--textvqa", type=Path, required=True)
    parser.add_argument("--exclude-image-requests", type=Path, required=True)
    parser.add_argument("--exclude-audit", type=Path, action="append", default=[],
                        help="Failed pre-inference pixel audit used only for deterministic image replacement")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-per-category", type=int, default=100)
    parser.add_argument("--confirmation-per-category", type=int, default=300)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if min(args.development_per_category, args.confirmation_per_category) < 1:
        raise ValueError("Each split must have at least one question per category")
    total = args.development_per_category + args.confirmation_per_category
    exclusions = exclusion_ids(args.exclude_image_requests, args.exclude_audit)
    gqa = gqa_pools(args.gqa, exclusions["gqa"])
    used_gqa: set[str] = set()
    presence = []
    if total % 2:
        raise ValueError("Presence total must be even for a balanced yes/no selection")
    for label in ("yes", "no"):
        pool = [row for row in gqa["object_presence"] if row["references"][0].lower() == label]
        presence.extend(select_unique(pool, total // 2, f"object_presence/{label}", used_gqa))
    presence.sort(key=lambda row: rank("object_presence/split", row["question_id"]))
    pools = {
        "object_presence": presence,
        "color": select_unique(gqa["color"], total, "color", used_gqa),
        "spatial_relation": select_unique(gqa["spatial_relation"], total, "spatial_relation", used_gqa),
        "count": select_unique(tally_pool(args.tallyqa, exclusions["tallyqa"]), total, "count", set()),
        "ocr": select_unique(textvqa_pool(args.textvqa, exclusions["textvqa"]), total, "ocr", set()),
    }
    # Every category is deterministically divided before development outcomes exist.
    development = [row for category in pools.values() for row in category[:args.development_per_category]]
    confirmation = [row for category in pools.values() for row in category[args.development_per_category:]]
    if len({row["question_id"] for row in development + confirmation}) != 5 * total:
        raise ValueError("Duplicate question IDs")
    if {row["image"] for row in development} & {row["image"] for row in confirmation}:
        raise ValueError("Development/confirmation image leakage")
    args.output.mkdir(parents=True)
    sources = {"gqa": digest(args.gqa), "tallyqa": digest(args.tallyqa), "textvqa": digest(args.textvqa)}
    exclusions_hash = digest(args.exclude_image_requests)
    failed_audit_hashes = {str(path): digest(path) for path in args.exclude_audit}
    freeze_split(args.output, "development", development, sources, exclusions_hash, failed_audit_hashes)
    freeze_split(args.output, "confirmation", confirmation, sources, exclusions_hash, failed_audit_hashes)
    print(json.dumps({"development": len(development), "confirmation": len(confirmation),
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
