_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        ),
    neck=dict(in_channels=[96, 192, 384, 768]),
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[1.0, 1.0, 2.0, 2.0]), #交叉熵，根据类别增加权重
            #loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0), # Focal loss
        ),
        mask_roi_extractor=None,
        mask_head=None,
    ),
    test_cfg=dict(
        rcnn=dict(
            mask_thr_binary=None, #预测时，可取消mask分支
            score_thr=0.05,   # src 0.05
            max_per_img=500)  # src:100
        )
    )

# 修改数据集相关配置
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
        ann_file=test_root + 'annotation/test.json',
        img_prefix=test_root + 'test/'
        ))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    #lr=0.0001,
    lr=1e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
lr_config = dict(warmup_iters=1000, step=[27, 33])

# 修改评价指标相关配置
runner = dict(type='EpochBasedRunner', max_epochs=40) #36
#checkpoint_config = dict(interval=-1)
checkpoint_config = dict(max_keep_ckpts=1) #只保留最新的几个模型
evaluation = dict(interval=1, metric=['bbox'], save_best='bbox_mAP_50')
log_config = dict(interval=100)

# 使用预训练的模型权重来做初始化，可以提高模型性能
#load_from = 'checkpoints/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_20210906_131725-bacf6f7b.pth'
load_from = 'work_dirs/Drink_swint_mask_rcnn/epoch_50_bottleAndWine.pth'