#!/usr/bin/env python3
"""Verify that a compact checkpoint reconstructs the full model exactly."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


def network_digest(net) -> dict:
    digest = hashlib.sha256()
    tensor_count = 0
    tensor_bytes = 0
    for module_name in net.model_list:
        for name, value in sorted(net[module_name].state_dict().items()):
            if not torch.is_tensor(value):
                continue
            tensor = value.detach().cpu().contiguous()
            digest.update(module_name.encode())
            digest.update(b"\0")
            digest.update(name.encode())
            digest.update(b"\0")
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
            tensor_count += 1
            tensor_bytes += tensor.numel() * tensor.element_size()
    return {
        "sha256": digest.hexdigest(),
        "tensor_count": tensor_count,
        "tensor_bytes": tensor_bytes,
    }


def load_digest(project: Path, config_path: Path, checkpoint_path: Path) -> dict:
    from mlp.base import common_functions as cmf

    config = OmegaConf.load(config_path)
    config.path.datasets = str(project / "datasets")
    config.path.deps = str(project / "deps")
    config.path.code_dir = str(project)
    config.path.working_dir = str(project / "state_compare")
    config.machine.use_gpu = False
    config.resume = ""
    cmf.seed_everything(int(config.seed))
    net, _ = cmf.factory_model(config, dset=None, logger=None)
    epoch, iteration = net.load_checkpoint(str(checkpoint_path), True)
    result = network_digest(net)
    result.update({"epoch": int(epoch), "iteration": int(iteration)})
    del net
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    project = args.project.resolve()
    os.chdir(project)
    sys.path.insert(0, str(project))
    from mlp.launch import prepare  # noqa: F401 - registers resolvers

    report = {
        "full": load_digest(project, args.config.resolve(), args.full.resolve()),
        "compact": load_digest(
            project, args.config.resolve(), args.compact.resolve()),
    }
    report["exact_state_match"] = report["full"] == report["compact"]
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    if not report["exact_state_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
