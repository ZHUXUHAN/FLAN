# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
import numpy as np
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from ..attribute_head.roi_attribute_feature_extractors import make_roi_attribute_feature_extractor
from ..box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_relation_feature_extractors import make_roi_relation_feature_extractor
from .roi_relation_predictors import make_roi_relation_predictor
from .inference import make_roi_relation_post_processor
from .loss import make_roi_relation_loss_evaluator
from .sampling import make_roi_relation_samp_processor

from maskrcnn_benchmark.structures.boxlist_ops import split_part_boxlist


class ROIRelationHead(torch.nn.Module):
    """
    Generic Relation Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIRelationHead, self).__init__()
        self.cfg = cfg.clone()
        self.union_feature_extractor = make_roi_relation_feature_extractor(cfg, in_channels)
        if cfg.MODEL.ATTRIBUTE_ON:
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, half_out=True)
            self.att_feature_extractor = make_roi_attribute_feature_extractor(cfg, in_channels, half_out=True)
            feat_dim = self.box_feature_extractor.out_channels * 2
        else:
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
            feat_dim = self.box_feature_extractor.out_channels
        self.predictor = make_roi_relation_predictor(cfg, feat_dim)
        self.post_processor = make_roi_relation_post_processor(cfg)
        self.loss_evaluator = make_roi_relation_loss_evaluator(cfg)
        self.samp_processor = make_roi_relation_samp_processor(cfg)

        self.part_type = cfg.MODEL.ROI_RELATION_HEAD.PART_HALF_TYPE
        # parameters
        self.use_union_box = self.cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION

    def forward(self, features, proposals, targets=None, logger=None, indexes=1):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes. Note: it has been post-processed (regression, nms) in sgdet mode
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            # relation subsamples and assign ground truth label during training
            with torch.no_grad():
                if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
                    proposals, rel_labels, rel_pair_idxs, rel_binarys, rel_tokens = self.samp_processor.gtbox_relsample(
                        proposals, targets)
                else:
                    proposals, rel_labels, rel_pair_idxs, rel_binarys, rel_tokens = self.samp_processor.detect_relsample(
                        proposals, targets)
        else:
            rel_labels, rel_binarys, rel_tokens = None, None, None
            rel_pair_idxs = self.samp_processor.prepare_test_pairs(features[0].device, proposals)

        left_top_box_lists = []
        right_top_box_lists = []
        left_bottom_box_lists = []
        right_bottom_box_lists = []
        for proposal in proposals:
            left_top_box_list, right_top_box_list, left_bottom_box_list, right_bottom_box_list = split_part_boxlist(
                proposal) ### LF TD
            left_top_box_lists.append(left_top_box_list)
            right_top_box_lists.append(right_top_box_list)
            left_bottom_box_lists.append(left_bottom_box_list)
            right_bottom_box_lists.append(right_bottom_box_list)
        if self.part_type == 'TD':
            left_top_box_lists = left_bottom_box_lists
            right_top_box_lists = right_bottom_box_lists
        roi_features_left_top = self.box_feature_extractor(features, left_top_box_lists)
        roi_features_right_top = self.box_feature_extractor(features, right_top_box_lists)
        part_features = [roi_features_left_top, roi_features_right_top, None, None]
        part_proposals = [left_top_box_lists, right_top_box_lists, None, None]
        roi_features = self.box_feature_extractor(features, proposals)

        if self.cfg.MODEL.ATTRIBUTE_ON:
            att_features = self.att_feature_extractor(features, proposals)
            roi_features = torch.cat((roi_features, att_features), dim=-1)

        if self.use_union_box:
            union_features = self.union_feature_extractor(features, proposals, rel_pair_idxs)
            union_features_lt_sub = self.union_feature_extractor(features, proposals, rel_pair_idxs, left_top_box_lists, part=True)
            union_features_rt_sub = self.union_feature_extractor(features, proposals, rel_pair_idxs, right_top_box_lists, part=True)
            union_features_part = [union_features_lt_sub, union_features_rt_sub]
        else:
            union_features = None

        refine_logits, relation_logits, add_losses, rel_labels = self.predictor(proposals, rel_pair_idxs, rel_labels, rel_binarys,  roi_features, union_features,
                                                                                part_features, part_proposals, union_features_part, logger)

        # for test
        if not self.training:
            result = self.post_processor((relation_logits, refine_logits), rel_pair_idxs, proposals)
            return roi_features, result, {}

        loss_relation, loss_refine = self.loss_evaluator(proposals, rel_labels, relation_logits, refine_logits)

        if self.cfg.MODEL.ATTRIBUTE_ON and isinstance(loss_refine, (list, tuple)):
            output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine[0], loss_refine_att=loss_refine[1])
        else:
            # output_losses = {}
            output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine)

        output_losses.update(add_losses)

        return roi_features, proposals, output_losses


def build_roi_relation_head(cfg, in_channels):
    """
    Constructs a new relation head.
    By default, uses ROIRelationHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIRelationHead(cfg, in_channels)
