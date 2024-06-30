import os
import numpy as np
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss, kl_div_loss, entropy_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.utils import cat
from .model_msg_passing import IMPContext
from .model_vtranse import VTransEFeature
from .model_vctree import VCTreeLSTMContext
from .model_motifs import LSTMContext, FrequencyBias
from .model_motifs_with_attribute import AttributeLSTMContext
from .model_transformer import TransformerContext
from .utils_relation import layer_init, get_box_info, get_box_pair_info
from maskrcnn_benchmark.data import get_dataset_statistics
from .utils_vctree import get_overlap_info
from .utils_motifs import obj_edge_vectors, to_onehot, encode_box_info
from math import pi
from .model_runet import RUNetContext, Boxes_Encode


@registry.ROI_RELATION_PREDICTOR.register("FLAN2Predictor")
class FLAN2Predictor(nn.Module):
    def __init__(self, config, in_channels):
        super(FLAN2Predictor, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)

        self.context_type = config.MODEL.ROI_RELATION_HEAD.PART_CONTEXT_TYPE
        if self.context_type == 'Motifs':
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        self.alpha = config.MODEL.ROI_RELATION_HEAD.ALPHA
        dropout = 0.1
        num_head = 4
        self.self_attn_node_lt = nn.MultiheadAttention(self.pooling_dim, num_head, dropout=dropout)
        self.self_attn_node_rt = nn.MultiheadAttention(self.pooling_dim, num_head, dropout=dropout)
        self.dropout_lt_1 = nn.Dropout(dropout)
        self.dropout_rt_1 = nn.Dropout(dropout)
        self.dropout_lt_2 = nn.Dropout(dropout)
        self.dropout_rt_2 = nn.Dropout(dropout)
        self.ffn_lt = FFN(self.pooling_dim, self.hidden_dim)
        self.ffn_rt = FFN(self.pooling_dim, self.hidden_dim)
        self.norm_lt = nn.LayerNorm(self.pooling_dim)
        self.norm_rt = nn.LayerNorm(self.pooling_dim)
        self.norm_lt_1 = nn.LayerNorm(self.pooling_dim)
        self.norm_rt_1 = nn.LayerNorm(self.pooling_dim)
        self.norm_lt_2 = nn.LayerNorm(self.pooling_dim)
        self.norm_rt_2 = nn.LayerNorm(self.pooling_dim)
        self.get_boxes_encode = Boxes_Encode()
        self.fc_dim = self.pooling_dim + 64
        # pure fc
        self.rel_compress = nn.Linear(self.fc_dim, self.num_rel_cls, bias=True)
        self.rel_compress_lt = nn.Linear(self.fc_dim, self.num_rel_cls, bias=True)
        self.rel_compress_rt = nn.Linear(self.fc_dim, self.num_rel_cls, bias=True)
        layer_init(self.rel_compress, xavier=True)
        layer_init(self.rel_compress_lt, xavier=True)
        layer_init(self.rel_compress_rt, xavier=True)

        # enhance feature
        self.post_emb_s = nn.Linear(self.pooling_dim, self.pooling_dim)
        layer_init(self.post_emb_s, xavier=True)
        self.post_emb_o = nn.Linear(self.pooling_dim, self.pooling_dim)
        layer_init(self.post_emb_o, xavier=True)
        self.merge_obj_low = nn.Linear(self.pooling_dim + 5 + 200, self.pooling_dim)
        layer_init(self.merge_obj_low, xavier=True)
        self.merge_obj_high = nn.Linear(self.hidden_dim, self.pooling_dim)
        layer_init(self.merge_obj_high, xavier=True)
        self.ort_embedding = nn.Parameter(self.get_ort_embeds(self.num_obj_cls, 200), requires_grad=False)
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        self.fg_matrix = statistics['fg_matrix']
        if self.fg_matrix is not None:
            self.fg_matrix[:, :, 0] = 0
        self.fg_matrix = self.fg_matrix.cuda()

        if self.use_bias:
            self.freq_bias = FrequencyBias(config, statistics)

    def get_ort_embeds(self, k, dims):
        ind = torch.arange(1, k + 1).float().unsqueeze(1).repeat(1, dims)
        lin_space = torch.linspace(-pi, pi, dims).unsqueeze(0).repeat(k, 1)
        t = ind * lin_space
        return torch.sin(t) + torch.cos(t)

    def get_relation_embedding(self, rel_pair_idxs, obj_preds, roi_features, union_features,
                               obj_preds_embeds, proposals, edge_ctxs, merge_obj_high, part_features=None):

        prod_reps = []
        pair_preds = []
        spt_feats = []

        for rel_pair_idx, obj_pred, roi_feat, union_feat, obj_embed, bboxes, edge_ctx_i, part_feature in zip(
                rel_pair_idxs, obj_preds, roi_features, union_features,
                obj_preds_embeds, proposals, edge_ctxs, part_features):
            if torch.numel(rel_pair_idx) == 0:
                continue
            w, h = bboxes.size
            bboxes_tensor = bboxes.bbox
            transfered_boxes = torch.stack(
                (
                    bboxes_tensor[:, 0] / w,
                    bboxes_tensor[:, 3] / h,
                    bboxes_tensor[:, 2] / w,
                    bboxes_tensor[:, 1] / h,
                    (bboxes_tensor[:, 2] - bboxes_tensor[:, 0]) * \
                    (bboxes_tensor[:, 3] - bboxes_tensor[:, 1]) / w / h,
                ), dim=-1
            )
            obj_features_low = cat(
                (
                    roi_feat, obj_embed, transfered_boxes
                ), dim=-1
            )

            obj_features = self.merge_obj_low(obj_features_low) + merge_obj_high(edge_ctx_i)

            subj_rep, obj_rep = self.post_emb_s(obj_features), self.post_emb_o(obj_features)
            assert torch.numel(rel_pair_idx) > 0

            spt_feats.append(self.get_boxes_encode(bboxes_tensor, rel_pair_idx, w, h, part_boxes=None))

            if part_feature is not None:
                prod_reps.append(subj_rep[rel_pair_idx[:, 0]] * obj_rep[rel_pair_idx[:, 1]] * union_feat * part_feature)
            else:
                prod_reps.append(
                    subj_rep[rel_pair_idx[:, 0]] * obj_rep[rel_pair_idx[:, 1]] * union_feat)
            pair_preds.append(torch.stack((obj_pred[rel_pair_idx[:, 0]], obj_pred[rel_pair_idx[:, 1]]), dim=1))

        prod_reps = cat(prod_reps, dim=0)
        pair_preds = cat(pair_preds, dim=0)
        spt_feats = cat(spt_feats, dim=0)
        prod_reps = cat((prod_reps, spt_feats), dim=-1)
        return prod_reps, pair_preds

    def get_relation_embedding_part(self, rel_pair_idxs, obj_preds, roi_features, union_features,
                                    obj_preds_embeds, proposals, edge_ctxs, merge_obj_high, all_proosals,
                                    part_features):

        prod_reps = []
        pair_preds = []
        spt_feats = []

        for rel_pair_idx, obj_pred, roi_feat, union_feat, obj_embed, bboxes, edge_ctx_i, all_bboxes, part_feature in zip(
                rel_pair_idxs, obj_preds, roi_features, union_features,
                obj_preds_embeds, proposals, edge_ctxs, all_proosals, part_features):
            if torch.numel(rel_pair_idx) == 0:
                continue
            w, h = bboxes.size
            bboxes_tensor = bboxes.bbox
            transfered_boxes = torch.stack(
                (
                    bboxes_tensor[:, 0] / w,
                    bboxes_tensor[:, 3] / h,
                    bboxes_tensor[:, 2] / w,
                    bboxes_tensor[:, 1] / h,
                    (bboxes_tensor[:, 2] - bboxes_tensor[:, 0]) * \
                    (bboxes_tensor[:, 3] - bboxes_tensor[:, 1]) / w / h,
                ), dim=-1
            )
            obj_features_low = cat(
                (
                    roi_feat, obj_embed, transfered_boxes
                ), dim=-1
            )

            bboxes_tensor_all = all_bboxes.bbox
            transfered_boxes_all = torch.stack(
                (
                    bboxes_tensor_all[:, 0] / w,
                    bboxes_tensor_all[:, 3] / h,
                    bboxes_tensor_all[:, 2] / w,
                    bboxes_tensor_all[:, 1] / h,
                    (bboxes_tensor_all[:, 2] - bboxes_tensor_all[:, 0]) * \
                    (bboxes_tensor_all[:, 3] - bboxes_tensor_all[:, 1]) / w / h,
                ), dim=-1
            )
            obj_features_low_all = cat(
                (
                    roi_feat, obj_embed, transfered_boxes_all
                ), dim=-1
            )

            obj_features = self.merge_obj_low(obj_features_low)

            obj_features_all = self.merge_obj_low(obj_features_low_all)

            subj_rep, obj_rep = self.post_emb_s(obj_features_all), self.post_emb_o(obj_features)
            assert torch.numel(rel_pair_idx) > 0

            spt_feats.append(self.get_boxes_encode(bboxes_tensor_all, rel_pair_idx, w, h, part_boxes=bboxes_tensor))

            prod_reps.append(subj_rep[rel_pair_idx[:, 0]] * obj_rep[rel_pair_idx[:, 1]] * union_feat * part_feature)
            pair_preds.append(torch.stack((obj_pred[rel_pair_idx[:, 0]], obj_pred[rel_pair_idx[:, 1]]), dim=1))

        prod_reps = cat(prod_reps, dim=0)
        pair_preds = cat(pair_preds, dim=0)
        spt_feats = cat(spt_feats, dim=0)
        prod_reps = cat((prod_reps, spt_feats), dim=-1)
        return prod_reps, pair_preds

    def part_feature(self, rel_pair_idxs, roi_features, roi_features_part, proposals_part, union_features_part,
                     self_attn_node_part, dropout_part_1, dropout_part_2, norm_part_1, norm_part_2, ffn_part):
        return_feature = []
        for pair_idx, roi_feature, roi_feature_part, proposal_part, union_feature_part in zip(rel_pair_idxs,
                                                                                              roi_features,
                                                                                              roi_features_part,
                                                                                              proposals_part,
                                                                                              union_features_part):
            sub_all_feature = roi_feature[pair_idx[:, 0]]
            obj_part_feature = roi_feature_part[pair_idx[:, 1]]
            src2 = self_attn_node_part(sub_all_feature, obj_part_feature, obj_part_feature, need_weights=False)[0]
            src = union_feature_part
            src = src + dropout_part_1(src2)
            src = norm_part_1(src)
            src2 = ffn_part(src)
            src = src + dropout_part_2(src2)
            src = norm_part_2(src)
            return_feature.append(src)
        return cat(return_feature, dim=0)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features,
                part_features, part_proposals, union_features_part, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        add_losses = {}

        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, logger)

        obj_preds_embeds = self.ort_embedding.index_select(0, obj_preds.long())

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        obj_preds = obj_preds.split(num_objs, dim=0)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]

        union_features_ = union_features.split(num_rels, dim=0)
        roi_features_ = roi_features.split(num_objs, dim=0)
        obj_preds_embeds = obj_preds_embeds.split(num_objs, dim=0)
        edge_ctxs = edge_ctx.split(num_objs, dim=0)

        roi_features_lt, roi_features_rt, _, _ = part_features
        proposals_lt, proposals_rt, _, _ = part_proposals
        union_features_lt, union_features_rt = union_features_part
        union_features_lt_, union_features_rt_ = union_features_lt.split(num_rels, dim=0), union_features_rt.split(
            num_rels, dim=0)
        roi_features_lt, roi_features_rt = roi_features_lt.split(num_objs, dim=0), roi_features_rt.split(num_objs, dim=0)

        prod_rep, pair_pred = self.get_relation_embedding(rel_pair_idxs, obj_preds, roi_features_,
                                                          union_features_,
                                                          obj_preds_embeds, proposals, edge_ctxs,
                                                          self.merge_obj_high, [None for i in proposals])
        part_features_lt = self.part_feature(rel_pair_idxs, roi_features_, roi_features_lt, proposals_lt,
                                             union_features_lt_,
                                             self.self_attn_node_lt, self.dropout_lt_1, self.dropout_lt_2,
                                             self.norm_lt_1, self.norm_lt_2, self.ffn_lt)
        part_features_rt = self.part_feature(rel_pair_idxs, roi_features_, roi_features_rt, proposals_rt,
                                             union_features_rt_,
                                             self.self_attn_node_rt, self.dropout_rt_1, self.dropout_rt_2,
                                             self.norm_rt_1, self.norm_rt_2, self.ffn_rt)

        part_features_lt_ = part_features_lt.split(num_rels, dim=0)
        part_features_rt_ = part_features_rt.split(num_rels, dim=0)

        prod_rep_lt, _ = self.get_relation_embedding_part(rel_pair_idxs, obj_preds, roi_features_lt,
                                                          union_features_lt_,
                                                          obj_preds_embeds, proposals_lt, edge_ctxs,
                                                          self.merge_obj_high, proposals, part_features_lt_)
        prod_rep_rt, _ = self.get_relation_embedding_part(rel_pair_idxs, obj_preds, roi_features_rt,
                                                          union_features_rt_,
                                                          obj_preds_embeds, proposals_rt, edge_ctxs,
                                                          self.merge_obj_high, proposals, part_features_rt_)

        rel_dists = self.rel_compress(prod_rep)
        rel_dists_lt = self.rel_compress_lt(prod_rep_lt)
        rel_dists_rt = self.rel_compress_rt(prod_rep_rt)

        if not self.training:
            rel_dists_lt[:, 1:] = rel_dists_lt[:, 1:] + self.freq_bias.index_with_labels(pair_pred.long())[:, 1:]
            rel_dists_rt[:, 1:] = rel_dists_rt[:, 1:] + self.freq_bias.index_with_labels(pair_pred.long())[:, 1:]
            rel_dists[:, 1:] = (rel_dists_lt[:, 1:] + rel_dists_rt[:, 1:]) * (1 - self.alpha) + rel_dists[:, 1:] * self.alpha

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses, rel_labels


class FFN(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x