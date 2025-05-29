_base_ = 'mmdetection\configs\sparse_rcnn\sparse-rcnn_r50_fpn_1x_coco.py'

default_hooks = dict(
    checkpoint=dict(interval=3, type='CheckpointHook'),
    early_stopping=dict(
        type='EarlyStoppingHook',
        monitor='val/bbox_mAP',
        rule='greater',
        patience=5,
        min_delta=0.001,
        priority=75
    )
)

# VOC 20类
classes = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
)

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='DIIHead',  # ←这才是官方 Sparse R-CNN 的 head
                num_classes=20
            )
        ] * 6
    )
)

dataset_type = 'CocoDataset'
data_root = 'split_data/'

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        ann_file='annotations/train.json',
        data_prefix=dict(img='train/'),
        data_root=data_root,
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=True, min_size=32)
    )
)
val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        ann_file='annotations/val_cleaned.json',
        data_prefix=dict(img='val/'),
        data_root=data_root,
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=True, min_size=32)
    )
)
test_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        ann_file='annotations/val_cleaned.json',
        data_prefix=dict(img='val/'),
        data_root=data_root,
        metainfo=dict(classes=classes),
        test_mode=True
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val_cleaned.json',
    metric=['bbox'],
    format_only=False,
    classwise=True
)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val_cleaned.json',
    outfile_prefix='./work_dirs/config1/test',
    metric=['bbox'],
    classwise=True
)

train_cfg = dict(max_epochs=50, type='EpochBasedTrainLoop', val_interval=1)

param_scheduler = [
    dict(begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=30,
        gamma=0.1,
        milestones=[8, 11],
        type='MultiStepLR'),
]

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ]
)

fp16 = dict(loss_scale='dynamic')
