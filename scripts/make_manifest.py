"""Refresh file hashes after an intentional update to the example release."""
from pathlib import Path
import hashlib
import json


def main():
    root = Path(__file__).resolve().parents[1]
    excluded = {".git", "__pycache__", ".pytest_cache", ".venv", "results", "build", "dist"}
    records = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or path.name == "MANIFEST.json":
            continue
        if any(part in excluded or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".class"}:
            continue
        content = path.read_bytes()
        records.append(dict(path=relative.as_posix(), bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
    (root / "MANIFEST.json").write_text(json.dumps(dict(version="1.2.1", files=records), indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Manifest updated: {len(records)} files")


if __name__ == "__main__":
    main()
