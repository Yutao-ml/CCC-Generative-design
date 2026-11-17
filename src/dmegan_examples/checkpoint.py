"""Verify and reassemble a byte-split checkpoint without changing its contents."""
from pathlib import Path
import hashlib
import json
import re


def read_checkpoint_parts(directory, expected_sha256=None):
    directory = Path(directory)
    index_path = directory / "selected_generator.pt.parts.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("format") != "byte-split-v1" or not index.get("parts"):
        raise ValueError("Unsupported or empty checkpoint-parts index")
    names = set()
    chunks = []
    for part in index["parts"]:
        name = part["file"]
        if not isinstance(name, str) or re.fullmatch(r"selected_generator\.pt\.part[0-9]{2}", name) is None or name in names:
            raise ValueError("Invalid or repeated checkpoint-part filename")
        names.add(name)
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint part: {name}. Keep every part and its index together.")
        content = path.read_bytes()
        if len(content) != part["bytes"] or hashlib.sha256(content).hexdigest() != part["sha256"]:
            raise ValueError(f"Checkpoint part failed size or SHA-256 verification: {name}")
        chunks.append(content)
    payload = b"".join(chunks)
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != index["original_bytes"] or digest != index["original_sha256"]:
        raise ValueError("Reassembled checkpoint failed size or SHA-256 verification")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Reassembled checkpoint does not match the dataset metadata")
    return payload
