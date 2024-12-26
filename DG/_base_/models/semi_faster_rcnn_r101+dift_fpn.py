_base_ = [
    './faster-rcnn_r101_fpn.py',

]

detector = _base_.model
detector.data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=64)

model = dict(
    _delete_=True,
    type='SemiBaseDiftDetector',
    detector=detector,
    dift_model=dict(config='', pretrained=''),
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        student_pretrained=None,
        freeze_teacher=True,),
    semi_test_cfg=dict(predict_on='teacher'),

)
