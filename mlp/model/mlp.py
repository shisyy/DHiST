import os
import torch
import torch.nn as nn
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig

from mlp.base import building_blocks as bb
from mlp.model.base import AbstractNetwork
from mlp.model.text import roberta
from mlp.model.encoder.s_enc_t_enc import FactorisedSTEncoder
from mlp.utils import io_utils, net_utils, vis_utils


class MLP(AbstractNetwork):
    def __init__(self,
                 optim: DictConfig,
                 use_gpu: bool,
                 save_pred: bool,
                 eval_checked: bool,
                 selection_metric: str = "weighted_recall",
                 qahs_boundary_logit_weight: float = 0.0,
                 segment_iou_weight: float = 0.0,
                 segment_iou_source: str = "qahs",
                 segment_score_mode: str = "soft_iou",
                 segment_masklog_weight: float = 0.0,
                 working_dir: str = "",
                 logger=None,
                 **kwargs):
        """Initialize network for Temporal Language Grounding in Human Motions"""
        super(MLP, self).__init__(optim_config=optim, 
                                    working_dir=working_dir, 
                                    dataset=optim["dataname"], 
                                    use_gpu=use_gpu, 
                                    logger=logger)

        self._build_network(kwargs)
        self._build_evaluator()
        self.save_pred = save_pred
        self.eval_checked = eval_checked
        self.selection_metric = selection_metric
        self.qahs_boundary_logit_weight = qahs_boundary_logit_weight
        self.segment_iou_weight = segment_iou_weight
        self.segment_iou_source = segment_iou_source
        self.segment_score_mode = segment_score_mode
        self.segment_masklog_weight = segment_masklog_weight

        # create counters and initialize status
        self._create_counters()
        self.reset_status(init_reset=True)


    def _build_network(self, configs):
       
        use_spatial_encoder = get_class(configs["encoder"]._target_) == FactorisedSTEncoder
        self.text_embedding = instantiate(configs["text"])
        self.motion_embedding = instantiate(configs["motion"], use_spatial_encoder=use_spatial_encoder)
        self.feature_encoder = instantiate(configs["encoder"], in_dim=self.motion_embedding.embed_dim)
        self.cq_attention = instantiate(configs["cq_attention"])
        self.cq_concat = instantiate(configs["cq_concatenate"])
        self.highlight_layer = instantiate(configs["highlight_layer"])
        self.predictor = instantiate(configs["predictor"])

        # build criterion
        self.use_highlight_loss = configs["losses"]["use_highlight_loss"]
        self.use_dn_loss = configs["losses"]["use_dn_loss"]
        self.criterion = bb.MultipleCriterions(
            ["grounding"], [instantiate(
                configs["losses"]["grounding_loss_func"])]
        )
        if self.use_highlight_loss:
            self.criterion.add("highlight", instantiate(
                configs["losses"]["highlight_loss_func"]))
        if self.use_dn_loss:
            self.criterion.add(f"dn_ce", instantiate(
                configs["losses"]["dn_loss_func"]))
            self.criterion.add(f"dn_kl", instantiate(
                    configs["losses"]["dn_kl_loss_func"]))

        # set model list
        self.model_list = ["text_embedding", "motion_embedding", "feature_encoder",
                           "cq_attention", "cq_concat", "highlight_layer",
                           "predictor", "criterion"]
        self.models_to_update = [
            "text_embedding",
            "motion_embedding",
            "feature_encoder",
            "cq_attention",
            "cq_concat",
            "highlight_layer",
            "predictor",
            "criterion"
        ]

        self.log("===> We train [{}]".format("|".join(self.models_to_update)))
        # init parameters
        self.init_parameters()

    def init_parameters(self):
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                m.reset_parameters()

        self.apply(init_weights)

    def forward(self, net_inps, mode="Train"):
        return self._infer(net_inps, mode)

    def visualize(self, vis_inps, vis_gt, prefix):
        self.eval_mode()
        vis_data = self._infer(vis_inps, "visualize", vis_gt)
        # import pandas as pd
        # for i in range(len(vis_gt['qids'])):
        #     res = {}
        #     res['Ps'] = vis_data['grounding_start_prob'][i]
        #     res['Pe'] = vis_data['grounding_end_prob'][i]
        #     res['S'] = vis_data['highlight_score'][i]
        #     res['GT_s'] = vis_gt['grounding_s_labels'][i].cpu().numpy()
        #     res['GT_e'] = vis_gt['grounding_e_labels'][i].cpu().numpy()
        #     res['GT_h'] = vis_gt['grounding_h_labels'][i].cpu().numpy()
        #     df = pd.DataFrame(res)
        #     df.to_csv(f"qid_{vis_gt['qids'][i]}.csv")
        vis_utils.visualize_MLP(self.working_dir, vis_data, prefix)

    def extract_output(self, vis_inps, vis_gt, save_dir):
        vis_data = self._infer(vis_inps, "save_output", vis_gt)

        qids = vis_data["qids"]
        preds = net_utils.loc2mask(loc, seg_masks)
        for i, qid in enumerate(qids):
            out = dict()
            for k in vis_data.keys():
                out[k] = vis_data[k][i]
            # save output
            save_path = os.path.join(save_dir, "{}.pkl".format(qid))
            io_utils.check_and_create_dir(save_dir)
            io_utils.write_pkl(save_path, out)

    def _infer(self, net_inps, mode="Train", gts=None):
        # fetch inputs
        motion_features = net_inps["motion_feats"]  # [B,T,d_v]
        m_mask = net_inps["motion_masks"]  # [B,T]
        query = net_inps["query"]  # [B,T]
        if mode == "Train":
            h_labels = net_inps["grounding_h_labels"]  # [B,T]
            uni_h_labels = net_inps["grounding_h_labels_all"]  # [B,T]
            uni_s_labels = net_inps["grounding_s_labels_all"]  # [B,T]
            uni_e_labels = net_inps["grounding_e_labels_all"]  # [B,T]
            predictor_s_labels = (
                net_inps["grounding_s_labels"] > 0
            ).to(torch.int64)
            predictor_e_labels = (
                net_inps["grounding_e_labels"] > 0
            ).to(torch.int64)
        else:
            # Prediction-only inference: no GT boundary/highlight tensor is
            # allowed to enter the model. GT timestamps remain available only
            # to compute metrics after the predictions have been produced.
            h_labels = torch.zeros_like(net_inps["grounding_h_labels"])
            uni_h_labels = torch.zeros_like(
                net_inps["grounding_h_labels_all"]
            )
            uni_s_labels = torch.zeros_like(
                net_inps["grounding_s_labels_all"]
            )
            uni_e_labels = torch.zeros_like(
                net_inps["grounding_e_labels_all"]
            )
            predictor_s_labels = torch.zeros_like(
                net_inps["grounding_s_labels"], dtype=torch.int64
            )
            predictor_e_labels = torch.zeros_like(
                net_inps["grounding_e_labels"], dtype=torch.int64
            )

        query_features, q_mask = self.text_embedding(query)
        motion_features = self.motion_embedding(motion_features, m_mask)
        motion_features, query_features, uni_scores = self.feature_encoder(
            motion_features,
            query_features,
            m_mask,
            q_mask,
            cur_step=self.it,
            total_step=self.total_step,
            h_mask=h_labels,
            uni_h_mask=uni_h_labels,
            uni_s_mask=uni_s_labels,
            uni_e_mask=uni_e_labels,
            mode=mode,
        )
        features = self.cq_attention(motion_features, query_features, m_mask, q_mask)
        features = self.cq_concat(features, query_features, q_mask)
        h_scores = self.highlight_layer(features, m_mask, self.it, self.total_step, h_labels, mode)
        features = features * h_scores.unsqueeze(2)
        outs = self.predictor(
            features,
            mask=m_mask,
            s_labels=predictor_s_labels,
            e_labels=predictor_e_labels,
            add_noise=(mode == "Train"),
        )

        # Fuse QAHS's predicted boundary confidence into the LP-Predictor logits.
        # At evaluation uni_scores are model predictions, never ground-truth masks.
        if self.qahs_boundary_logit_weight != 0.0:
            eps = torch.finfo(uni_scores[1].dtype).eps
            start_prior = torch.logit(uni_scores[1].clamp(min=eps, max=1.0 - eps))
            end_prior = torch.logit(uni_scores[2].clamp(min=eps, max=1.0 - eps))
            outs["grounding_start_loc"] = (
                outs["grounding_start_loc"]
                + self.qahs_boundary_logit_weight * start_prior
            )
            outs["grounding_end_loc"] = (
                outs["grounding_end_loc"]
                + self.qahs_boundary_logit_weight * end_prior
            )

        outs["uni_h_score"] = uni_scores[0]
        outs["uni_s_score"] = uni_scores[1]
        outs["uni_e_score"] = uni_scores[2]
        if mode != "visualize":
            if self.use_highlight_loss:
                outs["h_score"] = h_scores
        else:
            start_logits = outs["grounding_start_loc"]
            end_logits = outs["grounding_end_loc"]
            outs["mids"] = gts["mids"]
            outs["qids"] = gts["qids"]
            outs["query"] = net_inps["query"]
            outs["grounding_start_prob"] = net_utils.to_data(nn.Softmax(dim=1)(start_logits))
            outs["grounding_end_prob"] = net_utils.to_data(nn.Softmax(dim=1)(end_logits))
            outs["nfeats"] = gts["nfeats"]

            if self.use_highlight_loss:
                outs["grounding_gt"] = net_utils.to_data(gts["grounding_h_labels"])
                outs["highlight_score"] = net_utils.to_data(h_scores)

            else:
                outs["grounding_gt"] = net_utils.to_data(torch.zeros_like(start_logits))
                outs["highlight_score"] = net_utils.to_data(torch.zeros_like(start_logits))



            if mode == "save_output":
                outs["duration"] = gts["duration"]
                outs["timestamps"] = gts["timestamps"]

        return outs

    def prepare_batch(self, batch):
        self.gt_list = [
            "mids",
            "qids",
            "timestamps",
            "timestamps_ckd",
            "duration",
            "grounding_s_labels",
            "grounding_e_labels",
            "grounding_att_masks",
            "nfeats",
            "motion_feats",
            "motion_masks",
            "grounding_h_labels",
            'grounding_h_labels_all',
            'grounding_s_labels_all',
            'grounding_e_labels_all',
        ]
        self.both_list = ["query", "motion_feats", "motion_masks",
         "grounding_h_labels", "grounding_s_labels", "grounding_e_labels",
          'grounding_h_labels_all', 'grounding_s_labels_all', 'grounding_e_labels_all',
         ]

        net_inps, gts = {}, {}
        for k in batch.keys():
            item = (
                batch[k].to(self.device) if net_utils.istensor(
                    batch[k]) else batch[k]
            )

            if k in self.gt_list:
                gts[k] = item
            else:
                net_inps[k] = item

            if k in self.both_list:
                net_inps[k] = item

        return net_inps, gts


    """ methods for status & counters """
    def reset_status(self, init_reset=False):
        """Reset (initialize) metric scores or losses (status)."""
        super(MLP, self).reset_status(init_reset=init_reset)

        # initialize prediction maintainer for each epoch
        self.results = {
            "predictions": [],
            "gts": [],
            "gts_ckd": [],
            "durations": [],
            "mids": [],
            "qids": [],
        }

    def compute_status(self, net_outs, gts, mode="Train"):
        with_ckd = 'timestamps_ckd' in gts.keys()  
        # fetch data
        start_logits = net_outs["grounding_start_loc"].detach()
        end_logits = net_outs["grounding_end_loc"].detach()
        B = start_logits.size(0)
        gt_ts = gts["timestamps"]
        vid_d = gts["duration"]
        nfeats = gts["nfeats"]
        if with_ckd:
            gt_ts_ckd = gts["timestamps_ckd"]
        if self.segment_iou_source == "highlight":
            segment_scores = net_outs.get("h_score", net_outs["uni_h_score"])
        else:
            segment_scores = net_outs["uni_h_score"]
        start_indices, end_indices = self.extract_index(
            start_logits, end_logits, segment_scores=segment_scores)
        start_indices = start_indices.cpu().numpy()
        end_indices = end_indices.cpu().numpy()
        nfeats = nfeats.cpu().numpy()

        # prepare results for evaluation
        for ii in range(B):
            start_times, end_times = net_utils.index_to_time(
                start_indices[ii], end_indices[ii], nfeats[ii], vid_d[ii])
            pred = [[start_times[j], end_times[j]]
                    for j in range(len(start_times))]
            self.results["predictions"].append(pred)
            self.results["gts"].append([gt_ts[ii]])
            self.results["durations"].append(vid_d[ii])
            self.results["mids"].append(gts["mids"][ii])
            self.results["qids"].append(gts["qids"][ii])
            if with_ckd:
                self.results["gts_ckd"].append(gt_ts_ckd[ii])

    def extract_index(self, start_logits, end_logits, top_k=5,
                      segment_scores=None):
        start_prob = nn.Softmax(dim=1)(start_logits)
        end_prob = nn.Softmax(dim=1)(end_logits)
        outer = torch.matmul(start_prob.unsqueeze(dim=2),
                             end_prob.unsqueeze(dim=1))
        if self.segment_iou_weight != 0.0 and segment_scores is not None:
            # Rerank candidate segments by their soft IoU with a predicted
            # foreground mask. This uses predictions only and is O(T^2).
            batch_size, num_steps = start_prob.shape
            scores = segment_scores.clamp(min=0.0, max=1.0)
            prefix = torch.cat(
                [scores.new_zeros(batch_size, 1), scores.cumsum(dim=1)], dim=1
            )
            intersection = (
                prefix[:, 1:].unsqueeze(1) - prefix[:, :-1].unsqueeze(2)
            )
            indices = torch.arange(num_steps, device=scores.device)
            lengths = (
                indices.view(1, 1, -1) - indices.view(1, -1, 1) + 1
            ).to(scores.dtype)
            union = lengths + prefix[:, -1].view(-1, 1, 1) - intersection
            if self.segment_score_mode == "mask_loglik":
                evidence = torch.logit(scores.clamp(min=1e-6, max=1.0 - 1e-6))
                evidence_prefix = torch.cat(
                    [evidence.new_zeros(batch_size, 1),
                     evidence.cumsum(dim=1)], dim=1
                )
                segment_score = (
                    evidence_prefix[:, 1:].unsqueeze(1)
                    - evidence_prefix[:, :-1].unsqueeze(2)
                ) / float(num_steps)
            elif self.segment_score_mode == "inside_mean":
                segment_score = intersection / lengths.clamp_min(1.0)
            else:
                segment_score = intersection / union.clamp_min(1e-6)
            joint_score = (
                torch.log(start_prob.clamp_min(1e-12)).unsqueeze(2)
                + torch.log(end_prob.clamp_min(1e-12)).unsqueeze(1)
                + self.segment_iou_weight * segment_score
            )
            if self.segment_masklog_weight != 0.0:
                evidence = torch.logit(
                    scores.clamp(min=1e-6, max=1.0 - 1e-6)
                )
                evidence_prefix = torch.cat(
                    [evidence.new_zeros(batch_size, 1),
                     evidence.cumsum(dim=1)], dim=1
                )
                mask_loglik = (
                    evidence_prefix[:, 1:].unsqueeze(1)
                    - evidence_prefix[:, :-1].unsqueeze(2)
                ) / float(num_steps)
                joint_score = (
                    joint_score + self.segment_masklog_weight * mask_loglik
                )
            valid = torch.triu(
                torch.ones(num_steps, num_steps, dtype=torch.bool,
                           device=scores.device)
            )
            joint_score = joint_score.masked_fill(~valid.unsqueeze(0), -float("inf"))
            flat_indices = torch.topk(
                joint_score.flatten(1), k=top_k, dim=1
            ).indices
            return flat_indices // num_steps, flat_indices % num_steps
        # Keep elements on the diagonal and above
        outer = torch.triu(outer, diagonal=0)
        # Get the top k starting/ending indices with the highest probability
        _, start_indices = torch.topk(torch.max(outer, dim=2)[
                                          0], k=top_k, dim=1)  # (batch_size, top_k)
        _, end_indices = torch.topk(torch.max(outer, dim=1)[
                                        0], k=top_k, dim=1)  # (batch_size, top_k)
        return start_indices, end_indices

    def save_results(self, prefix, mode="Train"):
        # save predictions
        if self.save_pred:
            save_dir = os.path.join(self.working_dir, "predictions", mode)
            save_to = os.path.join(save_dir, prefix + ".json")
            io_utils.check_and_create_dir(save_dir)
            io_utils.write_json(save_to, self.results_obj2array())

        # compute performances
        nb = float(len(self.results["gts"]))
        self.evaluator.set_duration(self.results["durations"])
        rank1, rank5, miou, r5miou = self.evaluator.eval(
            self.results["predictions"], self.results["gts"]
        )

        for k, v in rank1.items():
            self.counters[k].add(v / nb * 100, 1)
        for k, v in rank5.items():
            self.counters[k].add(v / nb * 100, 1)
        self.counters["mIoU"].add(miou / nb * 100, 1)
        self.counters["R5-mIoU"].add(r5miou / nb * 100, 1)

        # Evaluate checked protocol
        if self.eval_checked:
            rank1_ckd, rank5_ckd, miou_ckd, r5miou_ckd = self.evaluator.eval(
                self.results["predictions"], self.results["gts_ckd"]
            )
            for k, v in rank1_ckd.items():
                self.counters[f'{k}_ckd'].add(v / nb * 100, 1)
            for k, v in rank5_ckd.items():
                self.counters[f'{k}_ckd'].add(v / nb * 100, 1)
            self.counters["mIoU_ckd"].add(miou_ckd / nb * 100, 1)
            self.counters["R5-mIoU_ckd"].add(r5miou_ckd / nb * 100, 1)

    def results_obj2array(self):
        arr = []
        B = len(self.results["predictions"])
        for ii in range(B):
            obj = dict()
            obj["qid"] = self.results["qids"][ii]
            obj["mid"] = self.results["mids"][ii]
            obj["predictions"] = self.results["predictions"][ii]
            obj["gts"] = self.results["gts"][ii]
            obj["durations"] = self.results["durations"][ii]
            arr.append(obj)
        return arr

    def renew_best_score(self):
        cur_score = self._get_score()
        if (self.best_score is None) or (cur_score > self.best_score):
            self.best_score = cur_score
            self.log(
                "Iteration {}: New best score {:4f}".format(
                    self.it, self.best_score)
            )
            return True
        self.log("Iteration {}: Current score {:4f}".format(self.it, cur_score))
        self.log(
            "Iteration {}: Current best score {:4f}".format(
                self.it, self.best_score)
        )
        return False

    def _get_score(self):
        if self.selection_metric == "mIoU":
            return self.counters["mIoU"].get_average()
        return super()._get_score()

    def bring_dataset_info(self, dset):
        """ methods for updating configuration """
        super(MLP, self).bring_dataset_info(dset)

    def save_checkpoint(self, ckpt_path, epoch, save_crit=False):
        super(MLP, self).save_checkpoint(ckpt_path, epoch, save_crit)

        if isinstance(self.text_embedding, roberta.RobertaEncoder):
            # In order to save storage space, only the weight of LoRA is saved
            ckpt_path = os.path.dirname(ckpt_path)
            self.text_embedding.save_lora(ckpt_path)
    
    
