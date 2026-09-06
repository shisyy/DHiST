#!/usr/bin/env python3
"""Remove the frozen RoBERTa backbone from a DHiST checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def keep_text_tensor(name: str) -> bool:
    return (
        name.startswith("linear.")
        or ".lora_A." in name
        or ".lora_B." in name
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    if args.destination.exists():
        raise FileExistsError(args.destination)
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    text_state = checkpoint.get("text_embedding")
    if not isinstance(text_state, dict):
        raise KeyError("checkpoint has no text_embedding state dictionary")

    compact_text_state = {
        name: value for name, value in text_state.items() if keep_text_tensor(name)
    }
    required_lora = [name for name in compact_text_state if ".lora_" in name]
    required_linear = [name for name in compact_text_state if name.startswith("linear.")]
    if len(required_lora) != 48 or len(required_linear) != 2:
        raise RuntimeError(
            f"expected 48 LoRA and 2 projection tensors, got "
            f"{len(required_lora)} and {len(required_linear)}"
        )

    compact = dict(checkpoint)
    compact["text_embedding"] = compact_text_state
    compact["checkpoint_format"] = "dhist-lora-compact-v1"
    compact["frozen_text_backbone"] = "roberta-base"
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(compact, args.destination)
    print(f"source_bytes={args.source.stat().st_size}")
    print(f"destination_bytes={args.destination.stat().st_size}")
    print(f"sha256={sha256(args.destination)}")


if __name__ == "__main__":
    main()
