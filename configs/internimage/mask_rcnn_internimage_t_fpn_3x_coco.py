# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

classes_num = 3
pred_segm = False

model = dict(
    backbone=dict(
        _delete_=True,
        type='InternImage',
        core_op='DCNv3',
        channels=64,
        depths=[4, 4, 18, 4],
        groups=[4, 8, 16, 32],
        mlp_ratio=4.,
        drop_path_rate=0.2,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=1.0,
        post_norm=False,
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=5),
    roi_head=dict(
        bbox_head=dict(num_classes=classes_num,
                       #loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1.0, 1.0, 2.0, 2.0]), #交叉熵，根据类别增加权重
                       loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0), #Focal loss
                       ),
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
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.05,
            max_per_img=500,
            mask_thr_binary=0.5 if pred_segm else None,
        ))
)


dataset_type = 'CocoDataset'
#train_root = '/data4/wh/DrinkData/'
test_root = '/data4/wh/bottle_data/'
train_root = test_root
classes = ('daiding_101', 'daiding_102', 'daiding_103')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # 新增光照变换：
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                           (736, 1333), (768, 1333), (800, 1333)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
                  [
                      dict(
                          type='Resize',
                          img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                          multiscale_mode='value',
                          keep_ratio=True),
                      dict(
                          type='RandomCrop',
                          crop_type='absolute_range',
                          crop_size=(384, 600),
                          allow_negative_crop=True),
                      dict(
                          type='Resize',
                          img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                     (576, 1333), (608, 1333), (640, 1333),
                                     (672, 1333), (704, 1333), (736, 1333),
                                     (768, 1333), (800, 1333)],
                          multiscale_mode='value',
                          override=True,
                          keep_ratio=True)
                  ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=train_root + 'annotation/train_xt_lb.json',
        img_prefix=train_root + 'train_xt_lb/',
        pipeline=train_pipeline
        ),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=test_root + 'annotation/val.json',
        img_prefix=test_root + 'val/'),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=test_root + 'annotation/val.json',
        img_prefix=test_root + 'val/'
        ))

optimizer = dict(
    _delete_=True, type='AdamW',
    lr=1e-5,  # 1e-4
    weight_decay=0.05,
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=30, layer_decay_rate=1.0,
                       depths=[4, 4, 18, 4]))
optimizer_config = dict(grad_clip=None)
lr_config = dict(step=[27, 33])

# Default setting for scaling LR automatically
auto_scale_lr = dict(enable=False, base_batch_size=16)

# 修改评价指标相关配置
runner = dict(type='EpochBasedRunner', max_epochs=40) #36
checkpoint_config = dict(max_keep_ckpts=1) #只保留最新的几个模型
evaluation = dict(interval=1, metric=['bbox'], save_best='bbox_mAP_50')
log_config = dict(interval=100)

# 使用预训练的模型权重来做初始化，可以提高模型性能
#load_from = 'checkpoints/mask_rcnn_internimage_t_fpn_3x_coco.pth'
load_from = 'work_dirs/mask_rcnn_internimage_t_fpn_3x_coco/epoch_40_bottleAndWine.pth'