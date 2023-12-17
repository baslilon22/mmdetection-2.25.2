_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]


classes_num = 46
model = dict(
    type='RetinaNet',
    backbone=dict(
        depth=101,
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=None,
    ),
    bbox_head=dict(num_classes=classes_num),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=500)
)

classes = ('fake', 'yl_jml_bhc_500P', 'yl_jml_bhc_600P', 'yl_jml_btxl_500P', 'yl_jml_cc_bdnmbhc_500P', 'yl_jml_cc_bzqtlc_500P', 'yl_jml_cc_kxjymlh_500P', 'yl_jml_dpbhc_750P', 'yl_jml_dpbtxl_750P', 'yl_jml_dpfmyz_750P', 'yl_jml_dplc_750P', 'yl_jml_dplzs_750P', 'yl_jml_dpmlmc_750P', 'yl_jml_dpqmlc_750P', 'yl_jml_fmyz_500P', 'yl_jml_jjnm_500P', 'yl_jml_jk_kqs_570P', 'yl_jml_kqs_550P', 'yl_jml_lbk_550P', 'yl_jml_lc_500P', 'yl_jml_lc_600P', 'yl_jml_lpbhc_1000P', 'yl_jml_lpbtxl_1000P', 'yl_jml_lpfmyz_1000P', 'yl_jml_lplc_1000P', 'yl_jml_lpmlmc_1000P', 'yl_jml_lpqmlc_1000P', 'yl_jml_mdxz_hls_500P', 'yl_jml_mdxz_mts500P', 'yl_jml_mdxz_mts_500P', 'yl_jml_mdxz_nms500P', 'yl_jml_mdxz_nms_500P', 'yl_jml_mdxz_qms_500P', 'yl_jml_mdxz_qpgs500P', 'yl_jml_mdxz_qpgs_500P', 'yl_jml_mdxz_xgs_500P', 'yl_jml_mlmc_500P', 'yl_jml_mlmc_600P', 'yl_jml_others', 'yl_jml_qmlc_500P', 'yl_jml_rsjs_450P', 'yl_jml_sdsbtw_450P', 'yl_jml_sdsnmw_450P', 'yl_jml_sdsyw_450P', 'yl_jml_smt_500P', 'yl_jml_wsss_450P')
train_root = '/data4/wh/JML_Shrank/'
test_root = '/data4/wh/JML_Shrank/'
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
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
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
        ann_file=train_root + 'annotation/train_All.json',
        img_prefix=train_root + 'train_All/',
        pipeline=train_pipeline),
    val=dict(
        type='CocoDataset',
        classes=classes,
        ann_file=test_root + 'annotation/test.json',
        img_prefix=test_root + 'test/',
        pipeline=test_pipeline),
    test=dict(
        type='CocoDataset',
        classes=classes,
        ann_file=test_root + 'annotation/test.json',
        img_prefix=test_root + 'test/',
        pipeline=test_pipeline))

checkpoint_config = dict(max_keep_ckpts=1) #只保留最新的几个模型
evaluation = dict(interval=1, metric=['bbox'], save_best='bbox_mAP_50')
load_from = '/data4/wh/CalibratedTeacher/checkpoints/retinanet_r101_fpn_mstrain_3x_coco_20210720_214650-7ee888e0.pth'
runner = dict(type='EpochBasedRunner', max_epochs=40) #36

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
    _delete_=True,
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
