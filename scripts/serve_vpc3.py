"""Serve VPC3's single-request JSON contract over a small local HTTP API."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from woodpecker_qwen.vpc3_integration import DEFAULT_QWEN3_REVISION, VPC3Calibrator


class Handler(BaseHTTPRequestHandler):
    def _json(self, status: int, value: dict) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"status": "ok", "method_version": "VPC3-native-v1"})
        else:
            self._json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/calibrate":
            self._json(404, {"error": "not_found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size < 1 or size > 32 * 1024 * 1024:
                raise ValueError("JSON request must be between 1 byte and 32 MiB")
            request = json.loads(self.rfile.read(size))
            result = self.server.calibrator.calibrate(request)
        except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
            return
        except Exception:
            self._json(500, {"error": "calibration_failed"})
            return
        self._json(200, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--gate", type=Path, default=ROOT / "artifacts/vpc3_gate")
    parser.add_argument("--qwen3-base-url")
    parser.add_argument("--qwen3-model-name", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--qwen3-model-revision", default=DEFAULT_QWEN3_REVISION)
    args = parser.parse_args()
    calibrator = VPC3Calibrator(
        args.gate, qwen3_base_url=args.qwen3_base_url,
        qwen3_model_name=args.qwen3_model_name,
        qwen3_model_revision=args.qwen3_model_revision)
    server = HTTPServer((args.host, args.port), Handler)
    server.calibrator = calibrator
    print(json.dumps({"listening": f"http://{args.host}:{args.port}",
                      "endpoint": "/calibrate"}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
