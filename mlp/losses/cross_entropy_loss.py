"""Boundary-aware cross entropy used by the HumanML3D release recipe.

The historical HumanML3D checkpoint used Gaussian-smoothed boundary targets.
This module keeps that training behavior while retaining ordinary cross
entropy when ``boundary_smoothing_sigma`` is zero.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        name,
        pred_lambda: float,
        smooth: bool = False,
        extend: float = 0.1,
        alpha: float = 0.4,
        boundary_smoothing_sigma: float = 0.0,
        boundary_smoothing_mix: float = 1.0,
        expected_iou_lambda: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.smooth = smooth
        self.extend = extend
        self.alpha = alpha
        self.pred_lambda = pred_lambda
        self.boundary_smoothing_sigma = float(boundary_smoothing_sigma)
        self.boundary_smoothing_mix = float(boundary_smoothing_mix)
        self.expected_iou_lambda = float(expected_iou_lambda)
        if not 0.0 <= self.boundary_smoothing_mix <= 1.0:
            raise ValueError("boundary_smoothing_mix must be in [0, 1]")
        if self.expected_iou_lambda != 0.0:
            raise ValueError(
                "This clean release supports expected_iou_lambda=0 only; "
                "the published checkpoints and recipes both use zero."
            )

    def _boundary_loss(self, logits, labels, mask):
        if self.boundary_smoothing_sigma <= 0.0:
            return F.cross_entropy(logits, labels, reduction="mean")

        hard_targets = labels.float()
        centers = hard_targets.argmax(dim=-1)
        positions = torch.arange(logits.shape[-1], device=logits.device)
        distances = positions.unsqueeze(0) - centers.unsqueeze(1)
        soft_targets = torch.exp(
            -0.5 * (distances.float() / self.boundary_smoothing_sigma).square()
        )
        soft_targets = soft_targets * mask.float()
        soft_targets = soft_targets / soft_targets.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        targets = (
            self.boundary_smoothing_mix * soft_targets
            + (1.0 - self.boundary_smoothing_mix) * hard_targets
        )
        return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

    def forward(self, net_outs, gts):
        start_logits = net_outs["grounding_start_loc"]
        end_logits = net_outs["grounding_end_loc"]
        start_labels = gts["grounding_s_labels"]
        end_labels = gts["grounding_e_labels"]
        mask = gts["motion_masks"]
        start_loss = self._boundary_loss(start_logits, start_labels, mask)
        end_loss = self._boundary_loss(end_logits, end_labels, mask)
        return self.pred_lambda * (start_loss + end_loss)
