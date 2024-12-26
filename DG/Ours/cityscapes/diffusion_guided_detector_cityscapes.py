_base_ = [
    '../../_base_/models/semi_faster_rcnn_r101+dift_fpn.py',
    '../../_base_/dg_setting/dg_gdd_20k.py',
    '../../_base_/datasets/cityscapes/cityscapes.py'
]

detector = _base_.model
detector.data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=64)

detector.detector.roi_head.bbox_head.num_classes = 8
detector.dift_model.config = 'DG/Ours/cityscapes/diffusion_detector_cityscapes_source.py'
detector.dift_model.pretrained_model = 'trained_models/diffusion_detector_cityscapes_source.pth'

model = dict(
    _delete_=True,
    type='DomainDetector',
    detector=detector,
    data_preprocessor=detector.data_preprocessor,
    train_cfg=dict(
        detector_cfg=dict(type='SemiBaseDift',
                          burn_up_iters=_base_.burn_up_iters),
        cross_loss_cfg=dict(
            enable_cross_loss=True,
            cross_type=['dift_to_student'],
            cross_loss_weight=0.5
        ),
        feature_loss_cfg=dict(
            enable_feature_loss=True,
            feature_loss_type='mse',
            feature_loss_weight=0.5
        ),
        kd_cfg=dict(
            loss_cls_kd=dict(type='KnowledgeDistillationKLDivLoss',
                             class_reduction='sum', loss_weight=1.0),
            loss_reg_kd=dict(type='L1Loss', loss_weight=1.0),
        ),
    )
)

optim_wrapper = dict(clip_grad=dict(max_norm=1, norm_type=2))
