"""Verify the example archive using only the Python standard library."""
from pathlib import Path
import argparse
import hashlib
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    root = parser.parse_args().root.resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text("utf-8"))
    for record in manifest["files"]:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Invalid or missing manifest file: {record['path']}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            prefix = stream.read(100)
            if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"LFS pointer in place of payload: {record['path']}")
            stream.seek(0)
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if path.stat().st_size != record["bytes"] or digest.hexdigest() != record["sha256"]:
            raise ValueError(f"Checksum mismatch: {record['path']}")
    print(f"PASS: {len(manifest['files'])} files verified")


if __name__ == "__main__":
    main()
