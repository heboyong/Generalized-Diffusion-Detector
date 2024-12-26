# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Tuple
import torch
from torch import Tensor

from mmdet.models.utils import (rename_loss_dict,
                                reweight_loss_dict)
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector
from ..losses import KDLoss
from ..utils import unpack_gt_instances
from mmdet.structures.bbox import bbox2roi


@MODELS.register_module()
class DomainDetector(BaseDetector):
    """Base class for semi-supervised detectors.

    Semi-supervised detectors typically consisting of a teacher model
    updated by exponential moving average and a student model updated
    by gradient descent.

    Args:
        detector (:obj:`ConfigDict` or dict): The detector config.                     
        semi_train_cfg (:obj:`ConfigDict` or dict, optional):
            The semi-supervised training config.
        semi_test_cfg (:obj:`ConfigDict` or dict, optional):
            The semi-supervised testing config.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 detector: ConfigType,
                 train_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        assert train_cfg is not None, "train_cfg is must not None"
        assert train_cfg.detector_cfg.get('type',
                                          None) is not None, "train_cfg.detector_cfg must use type select one detector"
        assert train_cfg.detector_cfg.get('type') in ['SemiBaseDift']
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.model = MODELS.build(detector)
        self.detector_name = detector.get('type')
        self.train_cfg = train_cfg

        # cross model setting
        self.enable_cross_loss = self.train_cfg.cross_loss_cfg.get(
            'enable_cross_loss')
        self.cross_type = self.train_cfg.cross_loss_cfg.get('cross_type')

        # feature loss setting
        self.enable_feature_loss = self.train_cfg.feature_loss_cfg.get('enable_feature_loss')
        if self.enable_feature_loss:
            self.feature_loss_type = self.train_cfg.feature_loss_cfg.get(
                'feature_loss_type')
            self.feature_loss_weight = self.train_cfg.feature_loss_cfg.get(
                'feature_loss_weight')
            self.feature_loss = KDLoss(
                loss_weight=self.feature_loss_weight, loss_type=self.feature_loss_type)

        self.loss_cls_kd = MODELS.build(self.train_cfg.kd_cfg['loss_cls_kd'])
        self.loss_reg_kd = MODELS.build(self.train_cfg.kd_cfg['loss_reg_kd'])

        self.burn_up_iters = self.train_cfg.detector_cfg.get(
            'burn_up_iters', 0)
        self.local_iter = 0

    @property
    def with_rpn(self):
        if self.with_student:
            return hasattr(self.model.student, 'rpn_head')
        else:
            return hasattr(self.student, 'rpn_head')

    @property
    def with_student(self):
        return hasattr(self.model, 'student')

    def loss(self, batch_inputs: Dict[str, Tensor],
             batch_data_samples: Dict[str, SampleList]) -> dict:
        """Calculate losses from multi-branch inputs and data samples.
        Returns:
            dict: A dictionary of loss components
        """
        losses = dict()
        if self.local_iter >= self.burn_up_iters:
            losses.update(**self.model.student.loss(batch_inputs, batch_data_samples))
            losses.update(
                **self.loss_cross(batch_inputs, batch_data_samples))
        else:
            losses.update(**self.model.student.loss(batch_inputs, batch_data_samples))
            
        self.local_iter += 1

        return losses

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        if self.with_student:
            if self.model.semi_test_cfg.get('predict_on', 'teacher') == 'teacher':
                return self.model.teacher(batch_inputs, batch_data_samples, mode='predict')
            elif self.model.semi_test_cfg.get('predict_on', 'teacher') == 'student':
                return self.model.student(batch_inputs, batch_data_samples, mode='predict')
            elif self.model.semi_test_cfg.get('predict_on', 'teacher') == 'dift_detector':
                return self.model.dift_detector(batch_inputs, batch_data_samples, mode='predict')
        else:
            return self.model(batch_inputs, batch_data_samples, mode='predict')

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> SampleList:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        return self.model(
            batch_inputs, batch_data_samples, mode='tensor')

    def extract_feat(self, batch_inputs: Tensor):
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        if not self.with_student:
            x_backbone = self.model.backbone(batch_inputs)
            x_neck = self.model.neck(x_backbone)
        else:
            x_backbone = self.model.student.backbone(batch_inputs)
            x_neck = self.model.student.neck(x_backbone)
        return x_neck

    def extract_feat_from_dift(self, batch_inputs: Tensor):
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """

        x_backbone = self.model.dift_detector.backbone(batch_inputs)
        x_neck = self.model.dift_detector.neck(x_backbone)

        return x_neck

    def cross_loss_dift_to_student(self, batch_data_samples: SampleList, dift_fpn):
        losses = dict()
        if not self.with_rpn:
            detector_loss = self.model.student.bbox_head.loss(
                dift_fpn, batch_data_samples)
            losses.update(rename_loss_dict(
                'dift_to_student_cross_', detector_loss))
        else:
            proposal_cfg = self.model.student.train_cfg.get(
                'rpn_proposal', self.model.student.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = torch.zeros_like(
                    data_sample.gt_instances.labels)
            rpn_losses, rpn_results_list = self.model.student.rpn_head.loss_and_predict(
                dift_fpn, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rename_loss_dict(
                'dift_to_student_cross_', rpn_losses))
            roi_losses = self.model.student.roi_head.loss(
                dift_fpn, rpn_results_list, batch_data_samples)
            losses.update(rename_loss_dict(
                'dift_to_student_cross_', roi_losses))
        return losses

    def cross_loss_student_to_dift(self, batch_data_samples: SampleList, student_fpn):
        losses = dict()
        if not self.with_rpn:
            detector_loss = self.model.dift_detector.bbox_head.loss(
                student_fpn, batch_data_samples)
            losses.update(rename_loss_dict(
                'student_to_dift_cross_', detector_loss))
        else:
            proposal_cfg = self.model.dift_detector.train_cfg.get(
                'rpn_proposal', self.model.dift_detector.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = torch.zeros_like(
                    data_sample.gt_instances.labels)
            rpn_losses, rpn_results_list = self.model.dift_detector.rpn_head.loss_and_predict(
                student_fpn, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rename_loss_dict(
                'student_to_dift_cross_', rpn_losses))
            roi_losses = self.model.dift_detector.roi_head.loss(
                student_fpn, rpn_results_list, batch_data_samples)
            losses.update(rename_loss_dict(
                'student_to_dift_cross_', roi_losses))
        return losses

    ###########################################################################################################################
    @staticmethod
    def align_features(input_feats, refer_feats):
        aligned_feats = []

        for input_feat, refer_feat in zip(input_feats, refer_feats):
            N, C, H, W = input_feat.size()

            # Normalize input feature
            input_feat = input_feat.permute(1, 0, 2, 3).reshape(C, -1)
            input_mean = input_feat.mean(dim=-1, keepdim=True)
            input_std = input_feat.std(dim=-1, keepdim=True)
            input_feat = (input_feat - input_mean) / (input_std + 1e-6)

            # Normalize refer feature
            refer_feat = refer_feat.permute(1, 0, 2, 3).reshape(C, -1)
            refer_mean = refer_feat.mean(dim=-1, keepdim=True)
            refer_std = refer_feat.std(dim=-1, keepdim=True)

            # Align input feature with reference feature
            aligned_feat = input_feat * refer_std + refer_mean
            aligned_feat = aligned_feat.reshape(C, N, H, W).permute(1, 0, 2, 3)
            aligned_feats.append(aligned_feat)

        return tuple(aligned_feats)
    ###########################################################################################################################

    def loss_cross(self, batch_inputs: Tensor,
                   batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        student_x = self.model.student.extract_feat(batch_inputs)
        dift_x = self.model.dift_detector.extract_feat(batch_inputs)
        losses = dict()

        # cross model loss
        ##############################################################################################################
        if self.enable_cross_loss:
            if "student_to_dift" in self.cross_type:
                self.align_features(student_x, dift_x)
                losses.update(
                    **self.cross_loss_student_to_dift(batch_data_samples, student_x))
            if "dift_to_student" in self.cross_type:
                self.align_features(dift_x, student_x)
                losses.update(
                    **self.cross_loss_dift_to_student(batch_data_samples, dift_x))
        ###############################################################################################################

        # feature kd loss
        ##############################################################################################################
        if self.enable_feature_loss:
            feature_loss = dict()
            feature_loss['pkd_feature_loss'] = 0
            for i, (student_feature, dift_feature) in enumerate(zip(student_x, dift_x)):
                layer_loss = self.feature_loss(
                    student_feature, dift_feature)
                feature_loss['pkd_feature_loss'] += layer_loss/len(dift_x)
            losses.update(feature_loss)
        ##############################################################################################################

        # RPN forward
        if self.with_rpn:
            proposal_cfg = self.model.student.train_cfg.get(
                'rpn_proposal', self.model.student.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = torch.zeros_like(
                    data_sample.gt_instances.labels)
            _, rpn_results_list = self.model.student.rpn_head.loss_and_predict(student_x, rpn_data_samples,
                                                                               proposal_cfg=proposal_cfg)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        ##############################################################################################################
        # Apply cross-kd in ROI head
        roi_losses_kd = self.roi_head_loss_with_kd(
            student_x, dift_x, rpn_results_list, batch_data_samples)
        losses.update(roi_losses_kd)
        ##############################################################################################################

        return losses

    def roi_head_loss_with_kd(self,
                              student_x: Tuple[Tensor],
                              dift_x: Tuple[Tensor],
                              rpn_results_list,
                              batch_data_samples):
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs
        roi_head = self.model.student.roi_head

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = roi_head.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = roi_head.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in student_x])
            sampling_results.append(sampling_result)

        losses = dict()
        # bbox head loss
        if roi_head.with_bbox:
            bbox_results = self.bbox_loss_with_kd(
                student_x, dift_x, sampling_results)
            losses.update(bbox_results['loss_bbox_kd'])

        return losses

    def bbox_loss_with_kd(self, student_x, dift_x, sampling_results):
        rois = bbox2roi([res.priors for res in sampling_results])

        student_head, dift_head = self.model.student.roi_head, self.model.dift_detector.roi_head
        stu_bbox_results = student_head._bbox_forward(student_x, rois)
        dift_bbox_results = dift_head._bbox_forward(dift_x, rois)
        reused_bbox_results = dift_head._bbox_forward(student_x, rois)

        losses_kd = dict()
        # classification KD
        reused_cls_scores = reused_bbox_results['cls_score']
        dift_cls_scores = dift_bbox_results['cls_score']
        avg_factor = sum([res.avg_factor for res in sampling_results])
        loss_cls_kd = self.loss_cls_kd(
            reused_cls_scores,
            dift_cls_scores,
            avg_factor=avg_factor)
        losses_kd['loss_cls_kd'] = loss_cls_kd
        # l1 loss
        assert student_head.bbox_head.reg_class_agnostic \
            == dift_head.bbox_head.reg_class_agnostic
        num_classes = student_head.bbox_head.num_classes
        reused_bbox_preds = reused_bbox_results['bbox_pred']
        dift_bbox_preds = dift_bbox_results['bbox_pred']
        dift_cls_scores = dift_cls_scores.softmax(dim=1)[:, :num_classes]
        reg_weights, reg_distill_idx = dift_cls_scores.max(dim=1)
        if not student_head.bbox_head.reg_class_agnostic:
            reg_distill_idx = reg_distill_idx[:, None, None].repeat(1, 1, 4)
            reused_bbox_preds = reused_bbox_preds.reshape(-1, num_classes, 4)
            reused_bbox_preds = reused_bbox_preds.gather(
                dim=1, index=reg_distill_idx)
            reused_bbox_preds = reused_bbox_preds.squeeze(1)
            dift_bbox_preds = dift_bbox_preds.reshape(-1, num_classes, 4)
            dift_bbox_preds = dift_bbox_preds.gather(
                dim=1, index=reg_distill_idx)
            dift_bbox_preds = dift_bbox_preds.squeeze(1)

        loss_reg_kd = self.loss_reg_kd(
            reused_bbox_preds,
            dift_bbox_preds,
            weight=reg_weights[:, None],
            avg_factor=reg_weights.sum() * 4)
        losses_kd['loss_reg_kd'] = loss_reg_kd

        bbox_results = dict()
        for key, value in stu_bbox_results.items():
            bbox_results['stu_' + key] = value
        for key, value in dift_bbox_results.items():
            bbox_results['dift_' + key] = value
        for key, value in reused_bbox_results.items():
            bbox_results['reused_' + key] = value
        bbox_results['loss_bbox_kd'] = losses_kd
        return bbox_results
