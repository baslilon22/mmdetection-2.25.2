classes_num = 16
pred_segm = False

model = dict(
    type='MaskRCNN',
    backbone=dict(type='nextvit_small',
                  frozen_stages=-1,
                  norm_eval=True,
                  with_extra_norm=False
                  ),
    neck=dict(
        type='FPN',
        in_channels=[96, 256, 512, 1024],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=classes_num,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]) if pred_segm else None,
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=classes_num,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)) if pred_segm else None,
        ),
    train_cfg = dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg = dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05, #0.05
            #nms=dict(type='nms', iou_threshold=0.5),
            nms=dict(type='soft_nms', iou_threshold=0.5),
            max_per_img=500,  #100
            mask_thr_binary=0.5 if pred_segm else None,
            ))
)

classes = ('mnc_dj_gq', 'mnc_dj_hmz', 'mnc_dj_htj', 'mnc_dj_lsz', 'mnc_dj_mpt', 'mnc_dj_rxb', 'mnc_dj_tj', 'mnc_dj_tsz', 'mnc_dj_wt', 'mnc_dj_yzta', 'mnc_dj_yztg', 'mnc_region_front', 'mnc_region_mid', 'mnc_region_other', 'mnc_sp_sp', 'mnc_sp_xl')

train_root = '/data4/wh/Monaco/'
test_root = train_root
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # 新增光照变换：
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[{
            'type':
                'Resize',
            'img_scale': [(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                          (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                          (736, 1333), (768, 1333), (800, 1333)],
            'multiscale_mode':
                'value',
            'keep_ratio':
                True
        }],
            [{
                'type': 'Resize',
                'img_scale': [(400, 1333), (500, 1333), (600, 1333)],
                'multiscale_mode': 'value',
                'keep_ratio': True
            }, {
                'type': 'RandomCrop',
                'crop_type': 'absolute_range',
                'crop_size': (384, 600),
                'allow_negative_crop': True
            }, {
                'type':
                    'Resize',
                'img_scale': [(480, 1333), (512, 1333), (544, 1333),
                              (576, 1333), (608, 1333), (640, 1333),
                              (672, 1333), (704, 1333), (736, 1333),
                              (768, 1333), (800, 1333)],
                'multiscale_mode':
                    'value',
                'override':
                    True,
                'keep_ratio':
                    True
            }]]),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        classes=classes,
        ann_file=train_root + 'annotation/train.json',
        img_prefix=train_root + 'train_img_zb/',
        pipeline=train_pipeline),
    val=dict(
        type='CocoDataset',
        classes=classes,
        ann_file=test_root + 'annotation/test.json',
        img_prefix=test_root + 'test_img_zb/',
        pipeline=test_pipeline),
    test=dict(
        type='CocoDataset',
        classes=classes,
        ann_file=test_root + 'annotation/test.json',
        img_prefix=test_root + 'test_img_zb/',
        pipeline=test_pipeline))

checkpoint_config = dict(max_keep_ckpts=1) #只保留最新的几个模型
evaluation = dict(interval=1, metric=['bbox'], save_best='bbox_mAP_50')
load_from = 'checkpoints/mask_rcnn_3x_nextvit_small.pth'
runner = dict(type='EpochBasedRunner', max_epochs=50) #36

log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1)]
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
# Default setting for scaling LR automatically
auto_scale_lr = dict(enable=True, base_batch_size=16)

optimizer = dict(
    type='AdamW',
    lr=1e-4,  # 1e-4
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(norm=dict(decay_mult=0.0))))

optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001, # 1e-6
    step=[27, 33])