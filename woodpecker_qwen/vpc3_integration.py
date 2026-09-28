"""Single-request VPC3 adapter around the frozen native-answer gate.

Expert observations are produced without reading evaluation labels. A Qwen3-VL
service can provide one candidate; callers may also attach observations from
the other expert services. This module does not retrain or retune the gate.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from scripts.run_native_qwen3_expert import PROMPTS
from scripts.train_native_sklearn_gate import ALL_FEATURE_NAMES, feature_vector
from .backends import OpenAICompatibleBackend
from .native_experts import collect_proposals, normalize_answer
from .native_gate import compact_evidence, public_observations
from .native_router_v2 import route_question


DEFAULT_QWEN3_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
IMAGE_KEYED_EXPERTS = frozenset({
    "presence", "color", "siglip2_color", "count_anything", "spatial", "ocr",
})
QUESTION_KEYED_EXPERTS = frozenset({"countgd", "qwen3_vl", "qwen3_vl_32b", "blip_vqa"})
INPUT_EXPERTS = {
    "presence": {"sam3_presence"}, "color": {"sam3_mask_color"},
    "siglip2_color": {"siglip2_sam3_crop_color"},
    "count_anything": {"count_anything"}, "countgd": {"countgd"},
    "spatial": {"grounded_2d_geometry"},
    "ocr": {"ppocrv6_single_line", "ppocrv6_qwen3_verified"},
    "qwen3_vl": {"qwen3_vl_8b", "ppocrv6_qwen3_verified"},
    "qwen3_vl_32b": {"qwen3_vl_32b_fp8", "ppocrv6_qwen3_32b_verified"},
    "blip_vqa": {"blip_vqa_base"},
}
FORBIDDEN_INPUT_FIELDS = frozenset({"label", "references", "ground_truth", "gold_answer"})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _local_image(image: Any, temporary_dir: Path) -> Path:
    if isinstance(image, str):
        source, value = "path", image
    elif isinstance(image, dict):
        source, value = image.get("source"), image.get("value")
    else:
        raise ValueError("image must be a local path or {source, value} object")
    if source in {"path", "file"}:
        path = Path(str(value)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    if source == "base64":
        if not isinstance(value, str):
            raise ValueError("base64 image value must be a string")
        encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
        try:
            content = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Invalid base64 image") from exc
        if not content:
            raise ValueError("Empty base64 image")
        path = temporary_dir / "request_image.png"
        path.write_bytes(content)
        return path
    raise ValueError("image.source must be path or base64; remote URLs are not fetched")


class VPC3Calibrator:
    """Apply the unchanged frozen gate to fresh image-question proposals."""

    def __init__(self, gate_dir: str | Path, *, qwen3_base_url: str | None = None,
                 qwen3_model_name: str = "Qwen3-VL-8B-Instruct",
                 qwen3_model_revision: str = DEFAULT_QWEN3_REVISION):
        gate_dir = Path(gate_dir)
        spec_path, model_path = gate_dir / "gate_model.json", gate_dir / "gate_model.joblib"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        runtime_names = list(ALL_FEATURE_NAMES)
        trained_names = spec.get("feature_names")
        if not isinstance(trained_names, list) or len(set(trained_names)) != len(trained_names):
            raise ValueError("Invalid frozen gate feature schema")
        if set(trained_names) - set(runtime_names):
            raise ValueError("Frozen gate requires unavailable features")
        if spec.get("model_file_sha256") != _sha256(model_path):
            raise ValueError("Frozen gate model SHA256 mismatch")
        if spec.get("confidence_sha256"):
            raise ValueError("This adapter requires the no-confidence VPC3 gate")
        manifest_path = gate_dir / "artifact_manifest.json"
        manifest = None
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for filename, path in (("gate_model.json", spec_path), ("gate_model.joblib", model_path)):
                if manifest["sha256"][filename] != _sha256(path):
                    raise ValueError(f"Packaged gate artifact changed: {filename}")
        self.model = joblib.load(model_path)
        self.feature_indices = [runtime_names.index(name) for name in trained_names]
        self.trained_experts = {name.removeprefix("expert_") for name in trained_names
                                if name.startswith("expert_") and name != "expert_score"}
        if manifest is not None and "proposal_support" in manifest:
            self.trained_experts &= {name for name, count in manifest["proposal_support"].items()
                                     if int(count) > 0}
        self.threshold = float(spec["threshold"])
        self.qwen3_model_name = qwen3_model_name
        self.qwen3_model_revision = qwen3_model_revision
        self.qwen3_backend = (OpenAICompatibleBackend(qwen3_base_url, "EMPTY", qwen3_model_name)
                              if qwen3_base_url else None)

    def _qwen3_evidence(self, image_path: Path, question: str, answer_type: str) -> dict[str, Any]:
        if self.qwen3_backend is None:
            raise ValueError("Qwen3-VL service is not configured")
        response = self.qwen3_backend.generate(
            str(image_path), PROMPTS[answer_type].format(question=question), max_new_tokens=48)
        first = response.strip().splitlines()[0].strip().strip(" .") if response.strip() else ""
        answer = normalize_answer(first, answer_type)
        return {"question": question, "answer": answer,
                "status": "complete" if answer is not None else "invalid_output",
                "model_name": self.qwen3_model_name,
                "model_revision": self.qwen3_model_revision}

    def calibrate(self, request: dict[str, Any]) -> dict[str, Any]:
        """Return the five-field public answer. Expert data is optional input."""
        if not isinstance(request, dict) or FORBIDDEN_INPUT_FIELDS & request.keys():
            raise ValueError("Request must be an object without evaluation labels")
        request_id = str(request["request_id"])
        question = str(request["question"]).strip()
        original = str(request["original_answer"]).strip()
        if not request_id or not question or not original:
            raise ValueError("request_id, question and original_answer must be nonempty")
        supplied = request.get("expert_evidence", {})
        if not isinstance(supplied, dict):
            raise ValueError("expert_evidence must be an object")
        unknown = set(supplied) - IMAGE_KEYED_EXPERTS - QUESTION_KEYED_EXPERTS
        if unknown:
            raise ValueError(f"Unknown expert evidence source: {sorted(unknown)}")
        unsupported = {name for name in supplied
                       if not (INPUT_EXPERTS[name] & self.trained_experts)}
        if unsupported:
            raise ValueError(f"Expert evidence was not represented in gate training: {sorted(unsupported)}")
        if self.qwen3_backend is None and not supplied:
            raise ValueError("Configure a Qwen3-VL service or provide expert_evidence")

        with tempfile.TemporaryDirectory(prefix="vpc3_request_") as temporary:
            image_path = _local_image(request["image"], Path(temporary))
            image_key = str(image_path)
            route = route_question(question)
            answer_type = route["answer_type"]
            evidence: dict[str, dict[str, Any]] = {}
            for name, value in supplied.items():
                key = image_key if name in IMAGE_KEYED_EXPERTS else request_id
                evidence[name] = {key: value}
            if self.qwen3_backend is not None and "qwen3_vl" not in evidence:
                evidence["qwen3_vl"] = {
                    request_id: self._qwen3_evidence(image_path, question, answer_type)}
            question_row = {"question_id": request_id, "image": image_key, "question": question}
            baseline = {"raw_answer": original}
            proposals = [candidate for candidate in
                         collect_proposals(question_row, baseline, route, evidence)
                         if candidate["expert"] in self.trained_experts]
            row = {**question_row, "raw_answer": original, "answer_type": answer_type,
                   "perceptual_task": route["perceptual_task"],
                   "parse_status": route["parse_status"], "proposals": proposals}

            candidates = [candidate for candidate in proposals
                          if str(candidate["answer"]).casefold() != original.casefold()]
            scored = []
            if candidates:
                vectors = np.stack([feature_vector(row, candidate)[self.feature_indices]
                                    for candidate in candidates])
                scored = [(candidate, float(score)) for candidate, score in
                          zip(candidates, self.model.predict(vectors))]
            chosen = max(scored, key=lambda item: item[1]) if scored else None
            accepted = chosen is not None and chosen[1] > self.threshold
            selected = chosen[0] if chosen is not None else None
            final_answer = str(selected["answer"]) if accepted else original
            internal_evidence = (
                {"source": selected["expert"], "candidate_answer": selected["answer"],
                 "observations": public_observations(selected.get("details")),
                 "reason": "highest_scoring_candidate_above_threshold" if accepted
                           else "below_acceptance_threshold"}
                if selected is not None else
                {"source": None, "candidate_answer": None, "observations": {},
                 "reason": "no_expert_disagreement"})
            return {"request_id": request_id, "original_answer": original,
                    "calibrated_answer": final_answer,
                    "changed": final_answer.casefold() != original.casefold(),
                    "evidence": compact_evidence(internal_evidence)}
