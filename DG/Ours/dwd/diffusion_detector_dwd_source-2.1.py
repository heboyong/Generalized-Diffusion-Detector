_base_ = [
    '../../_base_/models/faster-rcnn_dift_fpn-2.1.py',
    '../../_base_/dg_setting/dg_20k.py',
    '../../_base_/datasets/dwd/dwd.py'
]

detector = _base_.model
detector.roi_head.bbox_head.num_classes = 7

train_cfg = dict(val_interval=20000)
