"""Optionally restore the original checkpoint outside the public data directory."""
from pathlib import Path
import argparse
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dmegan_examples.checkpoint import read_checkpoint_parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    target = args.out.resolve()
    if target.is_relative_to((ROOT / "data").resolve()):
        parser.error("Restore to results/ or another output directory, outside data/")
    if target.exists():
        raise FileExistsError("Refusing to overwrite an existing output file")
    content = read_checkpoint_parts(ROOT / "data/checkpoint")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(content)
    print(f"Restored {len(content)} bytes; SHA-256: {hashlib.sha256(content).hexdigest()}")


if __name__ == "__main__":
    main()
