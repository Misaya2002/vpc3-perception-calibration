"""Apply packaged VPC3 to one JSON request or a JSON array of requests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from woodpecker_qwen.vpc3_integration import DEFAULT_QWEN3_REVISION, VPC3Calibrator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Single request or JSON array")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gate", type=Path, default=ROOT / "artifacts/vpc3_gate")
    parser.add_argument("--qwen3-base-url", help="OpenAI-compatible /v1 endpoint")
    parser.add_argument("--qwen3-model-name", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--qwen3-model-revision", default=DEFAULT_QWEN3_REVISION)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    calibrator = VPC3Calibrator(
        args.gate, qwen3_base_url=args.qwen3_base_url,
        qwen3_model_name=args.qwen3_model_name,
        qwen3_model_revision=args.qwen3_model_revision)
    result = ([calibrator.calibrate(item) for item in payload] if isinstance(payload, list)
              else calibrator.calibrate(payload))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"requests": len(result) if isinstance(result, list) else 1,
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
