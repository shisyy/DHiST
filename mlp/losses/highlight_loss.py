import torch
import torch.nn as nn
import torch.nn.functional as F

class HighLightLoss(nn.Module):
    def __init__(self,
                 name,
                 epsilon=1e-12,
                 highlight_lambda: float = 5.0,
                 **kwargs):
        super(HighLightLoss, self).__init__()
        self.name = name
        self.epsilon = epsilon
        self.w = highlight_lambda
        print("High Light Loss - ", name)

    def forward(self, net_outs, gts):
        scores = net_outs["h_score"]
        labels = gts["grounding_h_labels"]
        mask = gts["motion_masks"]
        weights = torch.where(labels == 0.0, labels + 1.0, 2.0 * labels)
        loss_per_location = nn.BCELoss(reduction='none')(scores, labels.float())
        loss_per_location = loss_per_location * weights
        mask = mask.type(torch.float32)
        loss = torch.sum(loss_per_location * mask) / (torch.sum(mask) + self.epsilon)

        
        uni_mask = gts["motion_masks"]
        uni_h_scores = net_outs["uni_h_score"] * uni_mask
        uni_h_labels = gts["grounding_h_labels_all"] * uni_mask

        uni_s_scores = net_outs["uni_s_score"] * uni_mask
        uni_s_labels = gts["grounding_s_labels_all"] * uni_mask

        uni_e_scores = net_outs["uni_e_score"] * uni_mask
        uni_e_labels = gts["grounding_e_labels_all"] * uni_mask

        criterion1 = nn.BCELoss(reduction="mean")
        criterion2 = nn.MSELoss(reduction="mean")

        uni_h_loss = criterion1(uni_h_scores, uni_h_labels) + 10 * criterion2(uni_h_scores, uni_h_labels)
        uni_s_loss = criterion1(uni_s_scores, uni_s_labels) + 10 * criterion2(uni_s_scores, uni_s_labels)
        uni_e_loss = criterion1(uni_e_scores, uni_e_labels) + 10 * criterion2(uni_e_scores, uni_e_labels)

        # target = F.log_softmax(uni_labels / self.t, dim=-1)
        # prediction = F.softmax(uni_scores / self.t, dim=-1)
        # uni_loss = F.kl_div(target, prediction, reduction='batchmean')
        print("uni_h_loss: ", uni_h_loss, " uni_s_loss: ", uni_s_loss, " uni_e_loss: ", uni_e_loss)
        return self.w * (loss + uni_h_loss + uni_s_loss + uni_e_loss)
