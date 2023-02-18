_base_ = [
    '../configs/_base_/models/vit-base-p16.py',
    '../configs/_base_/datasets/imagenet_bs64_pil_resize_autoaug.py',
    '../configs/_base_/schedules/imagenet_bs4096_AdamW.py',
    '../configs/_base_/default_runtime.py'
]

model = dict(backbone=dict(img_size=384),
             head=dict(
                 num_classes=30,
                 in_channels=768,#指定和transformer一样默认的768
             )
             )

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=384, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(384, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=384),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
load_from = '../checkpoints/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth'
global_data_prefix = "../mmcls/data/fruit30_split/"
data = dict(
    # VIT模型比较占内存，调小了batch
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type='CustomDataset', data_prefix= global_data_prefix + 'train', ann_file= None, pipeline=train_pipeline),
    val=dict(
        type='CustomDataset', data_prefix= global_data_prefix + 'val', ann_file= None, pipeline=test_pipeline),
    test=dict(
        type='CustomDataset', data_prefix= global_data_prefix + 'val', ann_file= None, pipeline=test_pipeline),
)
# 1卡训练，学习率调低10倍
optimizer = dict(lr=0.01)
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=30)
work_dir = 'work_dir/vit-base-p16_ft-64xb64_in1k-224_fruit30'