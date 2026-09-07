import re

import torch
import torch.nn as nn
import torch.nn.functional as F


class KLLoss(nn.Module):
    def __init__(
        self,
        name,
        kl_lambda: float,
        t: float,
        detach_recovery_teacher: bool = False,
        mask_strategy: str = "original",
        epsilon=1e-8,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.epsilon = epsilon
        self.t = t
        self.kl_lambda = kl_lambda
        self.detach_recovery_teacher = detach_recovery_teacher
        self.mask_strategy = mask_strategy
        self.hybrid_valid_weight = self._parse_hybrid_weight(mask_strategy)
        print(
            "KL Loss - ", name, "mask_strategy=", mask_strategy,
            "hybrid_valid_weight=", self.hybrid_valid_weight)

    @staticmethod
    def _parse_hybrid_weight(mask_strategy):
        if mask_strategy in ("original", "valid"):
            return None
        match = re.fullmatch(r"hybrid(\d{1,3})", str(mask_strategy))
        if match is None:
            raise ValueError(f"unknown KL mask strategy: {mask_strategy}")
        percentage = int(match.group(1))
        if not 0 <= percentage <= 100:
            raise ValueError("hybrid KL percentage must be in [0, 100]")
        return percentage / 100.0

    def _masked_kl(
        self, start_logits, end_logits, dn_s_logits, dn_e_logits, mask,
        strategy,
    ):
        if strategy == "original":
            masked_start_logits = start_logits * mask
            masked_end_logits = end_logits * mask
            masked_dn_s_logits = dn_s_logits * mask
            masked_dn_e_logits = dn_e_logits * mask
        elif strategy == "valid":
            valid = mask.to(torch.bool)
            mask_floor = -30.0
            masked_start_logits = start_logits.masked_fill(~valid, mask_floor)
            masked_end_logits = end_logits.masked_fill(~valid, mask_floor)
            masked_dn_s_logits = dn_s_logits.masked_fill(~valid, mask_floor)
            masked_dn_e_logits = dn_e_logits.masked_fill(~valid, mask_floor)
        else:
            raise ValueError(f"unknown KL mask strategy: {strategy}")

        start_log_probability = F.log_softmax(
            masked_start_logits / self.t, dim=-1)
        recovered_start_probability = F.softmax(
            masked_dn_s_logits / self.t, dim=-1)
        end_log_probability = F.log_softmax(
            masked_end_logits / self.t, dim=-1)
        recovered_end_probability = F.softmax(
            masked_dn_e_logits / self.t, dim=-1)
        if self.detach_recovery_teacher:
            recovered_start_probability = recovered_start_probability.detach()
            recovered_end_probability = recovered_end_probability.detach()
        return (
            F.kl_div(
                start_log_probability, recovered_start_probability,
                reduction="batchmean")
            + F.kl_div(
                end_log_probability, recovered_end_probability,
                reduction="batchmean")
        )

    def forward(self, net_outs, gts):
        mask = gts["motion_masks"]
        values = (
            net_outs["grounding_start_loc"],
            net_outs["grounding_end_loc"],
            net_outs["dn_start_kl_loc"],
            net_outs["dn_end_kl_loc"],
            mask,
        )
        if self.hybrid_valid_weight is None:
            loss = self._masked_kl(*values, self.mask_strategy)
        else:
            original_loss = self._masked_kl(*values, "original")
            valid_loss = self._masked_kl(*values, "valid")
            weight = self.hybrid_valid_weight
            loss = (1.0 - weight) * original_loss + weight * valid_loss
        return self.kl_lambda * loss
