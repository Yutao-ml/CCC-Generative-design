from pathlib import Path
import hashlib
import json
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dmegan_examples.checkpoint import read_checkpoint_parts


def example_parts(directory):
    parts = [b"first byte sequence", b"second byte sequence"]
    records = []
    for number, content in enumerate(parts, 1):
        name = f"selected_generator.pt.part{number:02d}"
        (directory / name).write_bytes(content)
        records.append(dict(file=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
    payload = b"".join(parts)
    index = dict(format="byte-split-v1", original_bytes=len(payload),
                 original_sha256=hashlib.sha256(payload).hexdigest(), parts=records)
    (directory / "selected_generator.pt.parts.json").write_text(json.dumps(index), encoding="utf-8")
    return payload


def test_release_parts_reconstruct_the_original_checkpoint():
    metadata = json.loads((ROOT / "data/metadata.json").read_text("utf-8"))
    payload = read_checkpoint_parts(ROOT / "data/checkpoint", metadata["checkpoint"]["sha256"])
    assert len(payload) == metadata["checkpoint"]["storage"]["original_bytes"]
    index = json.loads((ROOT / "data/checkpoint/selected_generator.pt.parts.json").read_text("utf-8"))
    assert len(index["parts"]) == 3
    assert all(part["bytes"] < 25_000_000 for part in index["parts"])


def test_corrupted_part_is_rejected(tmp_path):
    example_parts(tmp_path)
    (tmp_path / "selected_generator.pt.part01").write_bytes(b"First byte sequence")
    with pytest.raises(ValueError, match="SHA-256"):
        read_checkpoint_parts(tmp_path)


def test_missing_part_is_reported(tmp_path):
    example_parts(tmp_path)
    (tmp_path / "selected_generator.pt.part02").unlink()
    with pytest.raises(FileNotFoundError, match="Missing checkpoint part"):
        read_checkpoint_parts(tmp_path)


def test_unexpected_complete_digest_is_rejected(tmp_path):
    example_parts(tmp_path)
    with pytest.raises(ValueError, match="dataset metadata"):
        read_checkpoint_parts(tmp_path, expected_sha256="0" * 64)
