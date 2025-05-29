auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
classes = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
)
data_root = 'split_data/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=3, type='CheckpointHook'),
    early_stopping=dict(
        min_delta=0.001,
        monitor='val/bbox_mAP',
        patience=5,
        priority=75,
        rule='greater',
        type='EarlyStoppingHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
fp16 = dict(loss_scale='dynamic')
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        add_extra_convs='on_input',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=4,
        out_channels=256,
        start_level=0,
        type='FPN'),
    roi_head=dict(
        bbox_head=[
            dict(num_classes=20, type='DIIHead'),
            dict(num_classes=20, type='DIIHead'),
            dict(num_classes=20, type='DIIHead'),
            dict(num_classes=20, type='DIIHead'),
            dict(num_classes=20, type='DIIHead'),
            dict(num_classes=20, type='DIIHead'),
        ],
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=2, type='RoIAlign'),
            type='SingleRoIExtractor'),
        num_stages=6,
        proposal_feature_channel=256,
        stage_loss_weights=[
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        type='SparseRoIHead'),
    rpn_head=dict(
        num_proposals=100,
        proposal_feature_channel=256,
        type='EmbeddingRPNHead'),
    test_cfg=dict(rcnn=dict(max_per_img=100), rpn=None),
    train_cfg=dict(
        rcnn=[
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(box_format='xyxy', type='BBoxL1Cost', weight=5.0),
                        dict(iou_mode='giou', type='IoUCost', weight=2.0),
                    ],
                    type='HungarianAssigner'),
                pos_weight=1,
                sampler=dict(type='PseudoSampler')),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(box_format='xyxy', type='BBoxL1Cost', weight=5.0),
                        dict(iou_mode='giou', type='IoUCost', weight=2.0),
                    ],
                    type='HungarianAssigner'),
                pos_weight=1,
                sampler=dict(type='PseudoSampler')),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(box_format='xyxy', type='BBoxL1Cost', weight=5.0),
                        dict(iou_mode='giou', type='IoUCost', weight=2.0),
                    ],
                    type='HungarianAssigner'),
                pos_weight=1,
                sampler=dict(type='PseudoSampler')),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(box_format='xyxy', type='BBoxL1Cost', weight=5.0),
                        dict(iou_mode='giou', type='IoUCost', weight=2.0),
                    ],
                    type='HungarianAssigner'),
                pos_weight=1,
                sampler=dict(type='PseudoSampler')),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(box_format='xyxy', type='BBoxL1Cost', weight=5.0),
                        dict(iou_mode='giou', type='IoUCost', weight=2.0),
                    ],
                    type='HungarianAssigner'),
                pos_weight=1,
                sampler=dict(type='PseudoSampler')),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(box_format='xyxy', type='BBoxL1Cost', weight=5.0),
                        dict(iou_mode='giou', type='IoUCost', weight=2.0),
                    ],
                    type='HungarianAssigner'),
                pos_weight=1,
                sampler=dict(type='PseudoSampler')),
        ],
        rpn=None),
    type='SparseRCNN')
num_proposals = 100
num_stages = 6
optim_wrapper = dict(
    clip_grad=dict(max_norm=1, norm_type=2),
    optimizer=dict(lr=2.5e-05, type='AdamW', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=30,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='annotations/val_cleaned.json',
        backend_args=None,
        data_prefix=dict(img='val/'),
        data_root='split_data/',
        metainfo=dict(
            classes=(
                'aeroplane',
                'bicycle',
                'bird',
                'boat',
                'bottle',
                'bus',
                'car',
                'cat',
                'chair',
                'cow',
                'diningtable',
                'dog',
                'horse',
                'motorbike',
                'person',
                'pottedplant',
                'sheep',
                'sofa',
                'train',
                'tvmonitor',
            )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='split_data/annotations/val_cleaned.json',
    backend_args=None,
    classwise=True,
    format_only=False,
    metric=[
        'bbox',
    ],
    outfile_prefix='./work_dirs/config1/test',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=30, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ann_file='annotations/train.json',
        backend_args=None,
        data_prefix=dict(img='train/'),
        data_root='split_data/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(
            classes=(
                'aeroplane',
                'bicycle',
                'bird',
                'boat',
                'bottle',
                'bus',
                'car',
                'cat',
                'chair',
                'cow',
                'diningtable',
                'dog',
                'horse',
                'motorbike',
                'person',
                'pottedplant',
                'sheep',
                'sofa',
                'train',
                'tvmonitor',
            )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='annotations/val_cleaned.json',
        backend_args=None,
        data_prefix=dict(img='val/'),
        data_root='split_data/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(
            classes=(
                'aeroplane',
                'bicycle',
                'bird',
                'boat',
                'bottle',
                'bus',
                'car',
                'cat',
                'chair',
                'cow',
                'diningtable',
                'dog',
                'horse',
                'motorbike',
                'person',
                'pottedplant',
                'sheep',
                'sofa',
                'train',
                'tvmonitor',
            )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='split_data/annotations/val_cleaned.json',
    backend_args=None,
    classwise=True,
    format_only=False,
    metric=[
        'bbox',
    ],
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = './work_dirs\\config2'
