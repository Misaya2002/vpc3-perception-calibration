"""Download a pinned VPC3 gate release from a Hugging Face model repository."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FILES = ("gate_model.json", "gate_model.joblib", "artifact_manifest.json")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="HF user/model-repo")
    parser.add_argument("--revision", required=True, help="Full commit hash")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Refusing to replace an existing gate directory")
    from huggingface_hub import snapshot_download

    downloaded = Path(snapshot_download(
        repo_id=args.repo_id, revision=args.revision,
        allow_patterns=list(FILES), local_dir=args.output))
    manifest = json.loads((downloaded / "artifact_manifest.json").read_text(encoding="utf-8"))
    for filename in FILES[:2]:
        if digest(downloaded / filename) != manifest["sha256"][filename]:
            raise ValueError(f"Downloaded gate hash mismatch: {filename}")
    print(json.dumps({"gate_dir": str(downloaded), "repo_id": args.repo_id,
                      "revision": args.revision, "hash_verified": True}))


if __name__ == "__main__":
    main()
