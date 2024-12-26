_base_ = [
    '../../_base_/models/faster-rcnn_dift_fpn.py',
    '../../_base_/dg_setting/dg_20k.py',
    '../../_base_/datasets/cityscapes/cityscapes.py'
]

detector = _base_.model
detector.roi_head.bbox_head.num_classes = 8
