#!/usr/bin/env python3
"""Download and verify the released BABEL checkpoint."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path


URL = (
    "https://github.com/shisyy/DHiST/releases/download/babel-v1.0/"
    "dhist_babel_35_3_epoch94.ckpt.pkl"
)
EXPECTED_SHA256 = "e014dcc8ec165bb9b0b2aa15d834e2fceddbf37c1fcd185c5de404e1bb233f83"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    destination = root / "artifacts" / "babel_35_3" / "ckpt.pkl"
    partial = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    actual = sha256(partial)
    if actual != EXPECTED_SHA256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch: expected {EXPECTED_SHA256}, got {actual}")
    partial.replace(destination)
    print(f"Saved verified checkpoint to {destination}")


if __name__ == "__main__":
    main()
