#!/usr/bin/env python3
"""Download and verify a released DHiST checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path


CHECKPOINTS = {
    "babel": {
        "url": (
            "https://github.com/shisyy/DHiST/releases/download/babel-v1.0/"
            "dhist_babel_35_3_epoch94.ckpt.pkl"
        ),
        "sha256": "e014dcc8ec165bb9b0b2aa15d834e2fceddbf37c1fcd185c5de404e1bb233f83",
        "artifact": "babel_35_3",
    },
    "humanml3d": {
        "url": (
            "https://github.com/shisyy/DHiST/releases/download/humanml3d-v1.0/"
            "dhist_humanml3d_49_0_seed123_e15.ckpt.pkl"
        ),
        "sha256": "ba4fd38e3498547fdf13063539a6be25a060ddd8583e7a41c5aa872b034991c7",
        "artifact": "humanml3d_49_0",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=tuple(CHECKPOINTS), default="babel")
    args = parser.parse_args()
    record = CHECKPOINTS[args.dataset]
    root = Path(__file__).resolve().parents[1]
    destination = root / "artifacts" / record["artifact"] / "ckpt.pkl"
    partial = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {record['url']}")
    with urllib.request.urlopen(record["url"]) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    actual = sha256(partial)
    if actual != record["sha256"]:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch: expected {record['sha256']}, got {actual}")
    partial.replace(destination)
    print(f"Saved verified checkpoint to {destination}")


if __name__ == "__main__":
    main()
