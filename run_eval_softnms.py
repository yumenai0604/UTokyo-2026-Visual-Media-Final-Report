custom_imports = dict(imports=['msfa'], allow_failed_imports=False)

dataset_type = 'SAR_Det_Finegrained_Dataset'
data_root = '/root/autodl-tmp/sardet100k/SARDet_100K/'
backend_args = None

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(800, 800), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

test_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type, data_root=data_root,
        ann_file='Annotations/test.json',
        data_prefix=dict(img='JPEGImages/test/'),
        test_mode=True, pipeline=test_pipeline, backend_args=backend_args,
    ),
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'Annotations/test.json',
    metric='bbox', classwise=True,
    format_only=False, backend_args=backend_args,
)

model = dict(
    type='FasterRCNN',
    train_cfg=None,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True, pad_size_divisor=32,
    ),
    backbone=dict(
        type='MSFA',
        use_sar=True,
        use_hog=False,
        use_canny=False,
        use_wavelet=True,      # ←与checkpoint匹配
        use_haar=False,
        use_grad_edge=False,
        input_size=(800, 800),
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=None,     # ← 不用ImageNet权重（通道数82，不兼容）
        ),
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256, num_outs=5,
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256, feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8], ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256, featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256, fc_out_channels=1024, roi_feat_size=7,
            num_classes=6,# ← 确认正确
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            reg_class_agnostic=False,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        ),
    ),
    test_cfg=dict(
        rpn=dict(nms_pre=1000, max_per_img=1000,
                 nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
        rcnn=dict(score_thr=0.001,
                  nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.001), max_per_img=300),
    ),
)

default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
test_cfg = dict(type='TestLoop')
