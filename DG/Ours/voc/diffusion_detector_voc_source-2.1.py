_base_ = [
    '../../_base_/models/faster-rcnn_dift_fpn-2.1.py',
    '../../_base_/dg_setting/dg_20k.py',
    '../../_base_/datasets/voc/voc.py'
]

detector = _base_.model
detector.roi_head.bbox_head.num_classes = 20
detector.backbone.dift_config.scheduler_timesteps = [100 * 5, 80 * 5, 60 * 5, 40 * 5, 20 * 5]
train_cfg = dict(val_interval=20000)
