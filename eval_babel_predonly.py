#!/usr/bin/env python3
"""Re-evaluate the verified BABEL checkpoint with prediction-only inference.

All label-like tensors are forced to zero before the accepted full evaluation.
Ground-truth timestamps are retained only by the metric accumulator.  A separate
first-batch audit compares actual, zero and randomized label tensors to establish
whether the historical label arguments can affect any prediction output.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from omegaconf import OmegaConf
from tqdm import tqdm


METRIC_KEYS = [
    "R1-0.1", "R1-0.3", "R1-0.5", "R1-0.7", "R1-0.9",
    "R5-0.1", "R5-0.3", "R5-0.5", "R5-0.7", "R5-0.9",
    "mIoU", "R5-mIoU",
]
LABEL_INPUT_KEYS = [
    "grounding_h_labels", "grounding_s_labels", "grounding_e_labels",
    "grounding_h_labels_all", "grounding_s_labels_all",
    "grounding_e_labels_all",
]
AUDIT_OUTPUT_KEYS = [
    "grounding_start_loc", "grounding_end_loc", "h_score",
    "uni_h_score", "uni_s_score", "uni_e_score",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clone_inputs(inputs: dict) -> dict:
    return {
        key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in inputs.items()
    }


def zero_labels(inputs: dict) -> None:
    for key in LABEL_INPUT_KEYS:
        if key in inputs:
            inputs[key] = torch.zeros_like(inputs[key])


def randomize_labels(inputs: dict) -> None:
    for key in LABEL_INPUT_KEYS:
        if key in inputs:
            inputs[key] = torch.randint_like(inputs[key], low=0, high=2)


def output_maxdiff(left: dict, right: dict) -> dict:
    result = {}
    for key in AUDIT_OUTPUT_KEYS:
        if key in left and key in right:
            result[key] = float((left[key] - right[key]).abs().max().item())
    return result


def force_recursive_eval(module: torch.nn.Module) -> None:
    """Set evaluation flags without relying on overridden ``train`` methods."""
    module.training = False
    for child in module.children():
        force_recursive_eval(child)


def rng_state() -> tuple[torch.Tensor, list[torch.Tensor]]:
    return torch.get_rng_state(), torch.cuda.get_rng_state_all()


def set_rng_state(state: tuple[torch.Tensor, list[torch.Tensor]]) -> None:
    cpu_state, cuda_states = state
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state_all(cuda_states)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main() -> None:
    repository = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Evaluate the released DHiST BABEL checkpoint without GT inputs."
    )
    parser.add_argument("--project", type=Path, default=repository)
    parser.add_argument(
        "--config",
        type=Path,
        default=repository / "artifacts" / "babel_35_3" / "config.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repository / "artifacts" / "babel_35_3" / "ckpt.pkl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "outputs" / "babel_35_3_eval",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--segment-iou-weight", type=float, default=40.0)
    parser.add_argument("--segment-masklog-weight", type=float, default=12.0)
    args = parser.parse_args()

    project = args.project.resolve()
    config_path = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    write_json(status_path, {
        "status": "running",
        "started_unix": time.time(),
        "inference_label_strategy": "zeros",
        "decoder_inputs": "predicted LP logits and predicted highlight scores only",
    })

    os.chdir(project)
    sys.path.insert(0, str(project))
    from mlp.base import common_functions as cmf
    from mlp.launch import prepare  # noqa: F401 - registers resolvers

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Run scripts/download_checkpoint.py first."
        )

    config = OmegaConf.load(config_path)
    config.path.datasets = str(project / "datasets")
    config.path.deps = str(project / "deps")
    config.path.code_dir = str(project)
    config.path.working_dir = str(output)
    config.data.datapath = str(project / "datasets" / "motions" / "guoh3dfeats")
    config.data.labelpath = str(project / "datasets" / "babel")
    config.data.prediction_only_labels = True
    config.resume = ""
    config.model.text.ckpt_dir = str(checkpoint.parent)
    config.model.eval_checked = True
    config.model.save_pred = True
    config.model.segment_iou_weight = float(args.segment_iou_weight)
    config.model.segment_iou_source = "highlight"
    config.model.segment_score_mode = "soft_iou"
    config.model.segment_masklog_weight = float(args.segment_masklog_weight)
    OmegaConf.save(config, output / "effective_eval_config.yaml")
    cmf.seed_everything(int(config.seed))

    dataset_module = cmf.get_dataset(config.data.dataname)
    datasets, loaders = cmf.get_loader(
        dataset_module, split=[args.split], data_config=config.data)
    logger = cmf.create_logger(config, "PREDONLY", "predonly_eval.log")
    net, _ = cmf.factory_model(config, datasets[args.split], logger)
    epoch_field, iteration_field = net.load_checkpoint(str(checkpoint), True)
    net.eval_mode()

    # RobertaEncoder in this historical snapshot overrides train() and returns
    # early when LoRA finetuning is enabled. Therefore ordinary .eval() leaves
    # its dropout layers active. Force the flags recursively for real inference.
    for model_name in net.model_list:
        force_recursive_eval(net[model_name])

    # Disable stochastic label corruption at inference. This does not alter
    # trained weights and removes the last non-prediction input variation.
    if hasattr(net.predictor, "label_noise_prob"):
        net.predictor.label_noise_prob = 0.0

    first_batch = next(iter(loaders[args.split]))
    actual_inputs, _ = net.prepare_batch(first_batch)
    zero_inputs = clone_inputs(actual_inputs)
    random_inputs = clone_inputs(actual_inputs)
    zero_labels(zero_inputs)
    randomize_labels(random_inputs)
    audit_rng = rng_state()
    with torch.no_grad():
        set_rng_state(audit_rng)
        actual_outputs = net.forward_only(actual_inputs, "PredOnlyAudit")["net_output"]
        set_rng_state(audit_rng)
        zero_outputs = net.forward_only(zero_inputs, "PredOnlyAudit")["net_output"]
        set_rng_state(audit_rng)
        random_outputs = net.forward_only(random_inputs, "PredOnlyAudit")["net_output"]
    invariance = {
        "actual_vs_zero": output_maxdiff(actual_outputs, zero_outputs),
        "zero_vs_random": output_maxdiff(zero_outputs, random_outputs),
        "actual_vs_random": output_maxdiff(actual_outputs, random_outputs),
    }
    write_json(output / "label_invariance_audit.json", invariance)

    net.reset_status()
    net.reset_counters()
    with torch.no_grad():
        for batch in tqdm(loaders[args.split], desc="BABEL prediction-only"):
            inputs, ground_truth = net.prepare_batch(batch)
            zero_labels(inputs)
            outputs = net.forward_only(inputs, "Test")
            net.compute_status(outputs["net_output"], ground_truth)
    net.save_results("latest", mode="Test")
    metrics = {key: float(net.counters[key].get_average()) for key in METRIC_KEYS}
    net.print_counters_info(logger, -1, mode="Test")

    summary = {
        "status": "completed",
        "protocol": {
            "split": f"{args.split} (BABEL validation protocol)",
            "inference_label_strategy": "zeros",
            "label_noise_at_inference": False,
            "dropout_at_inference": False,
            "model_decoder_gt_boundary_inputs": False,
            "ground_truth_usage": "metric computation only",
            "decoder_inputs": "predicted LP logits and predicted highlight scores only",
            "segment_iou_source": "highlight",
            "segment_score_mode": "soft_iou",
            "segment_iou_weight": float(args.segment_iou_weight),
            "segment_masklog_weight": float(args.segment_masklog_weight),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "epoch_field": int(epoch_field),
            "iteration_field": int(iteration_field),
        },
        "metrics": metrics,
        "label_invariance_maxdiff": invariance,
        "completed_unix": time.time(),
    }
    write_json(output / "summary.json", summary)
    write_json(status_path, {
        "status": "completed",
        "completed_unix": summary["completed_unix"],
        "mIoU": metrics["mIoU"],
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
