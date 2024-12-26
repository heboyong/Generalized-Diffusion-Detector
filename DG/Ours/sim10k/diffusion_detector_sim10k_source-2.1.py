_base_ = [
    '../../_base_/models/faster-rcnn_dift_fpn-2.1.py',
    '../../_base_/dg_setting/dg_20k.py',
    '../../_base_/datasets/sim10k/sim10k.py'
]

detector = _base_.model
detector.roi_head.bbox_head.num_classes = 1
