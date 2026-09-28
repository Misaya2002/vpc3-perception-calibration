"""Exercise the packaged gate and single-request contract without a GPU service."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from http.server import HTTPServer
from pathlib import Path
from threading import Thread
from urllib.request import Request, urlopen

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from woodpecker_qwen.vpc3_integration import VPC3Calibrator
from scripts.serve_vpc3 import Handler


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="vpc3_smoke_") as directory:
        image_path = Path(directory) / "red_car.png"
        Image.new("RGB", (16, 16), "red").save(image_path)
        request = {
            "request_id": "smoke-color-001",
            "image": {"source": "path", "value": str(image_path)},
            "question": "What color is the car?",
            "original_answer": "blue",
            "expert_evidence": {"qwen3_vl": {
                "question": "What color is the car?", "answer": "red",
                "status": "complete", "model_name": "Qwen3-VL-8B-Instruct",
                "model_revision": "smoke-fixture-not-a-model-inference"}},
        }
        result = VPC3Calibrator(ROOT / "artifacts/vpc3_gate").calibrate(request)
        request_path, response_path = Path(directory) / "request.json", Path(directory) / "response.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        subprocess.run([sys.executable, str(ROOT / "scripts/run_vpc3.py"),
                        "--input", str(request_path), "--output", str(response_path)],
                       check=True, capture_output=True, text=True)
        assert json.loads(response_path.read_text(encoding="utf-8")) == result
        server = HTTPServer(("127.0.0.1", 0), Handler)
        server.calibrator = VPC3Calibrator(ROOT / "artifacts/vpc3_gate")
        worker = Thread(target=server.handle_request, daemon=True)
        worker.start()
        try:
            post = Request(f"http://127.0.0.1:{server.server_port}/calibrate",
                           data=json.dumps(request).encode("utf-8"),
                           headers={"Content-Type": "application/json"})
            with urlopen(post, timeout=10) as response:
                assert response.status == 200
                assert json.loads(response.read()) == result
        finally:
            worker.join(timeout=10)
            server.server_close()
    assert set(result) == {"request_id", "original_answer", "calibrated_answer",
                           "changed", "evidence"}
    assert set(result["evidence"]) == {"source", "observation", "candidate_answer", "reason"}
    assert result["request_id"] == "smoke-color-001"
    assert result["calibrated_answer"] in {"blue", "red"}
    print(json.dumps({"status": "PASS", "result": result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
