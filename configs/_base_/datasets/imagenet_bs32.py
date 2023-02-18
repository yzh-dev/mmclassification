# dataset settings
dataset_type = 'ImageNet'  # 数据集名称，
img_norm_cfg = dict(  # 图像归一化配置，用来归一化输入的图像。
    mean=[123.675, 116.28, 103.53],  # 预训练里用于预训练主干网络模型的平均值。
    std=[58.395, 57.12, 57.375],  # 预训练里用于预训练主干网络模型的标准差。
    to_rgb=True)  # 是否反转通道，使用 cv2, mmcv 读取图片默认为 BGR 通道顺序，这里 Normalize 均值方差数组的数值是以 RGB 通道顺序， 因此需要反转通道顺序。
# 训练数据流水线
train_pipeline = [
    dict(type='LoadImageFromFile'),  # 读取图片
    dict(type='RandomResizedCrop', size=224),  # 随机缩放抠图
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),  # 以概率为0.5随机水平翻转图片
    dict(type='Normalize', **img_norm_cfg),  # 归一化
    dict(type='ImageToTensor', keys=['img']),  # image 转为 torch.Tensor
    dict(type='ToTensor', keys=['gt_label']),  # gt_label 转为 torch.Tensor
    dict(type='Collect', keys=['img', 'gt_label'])  # 决定数据中哪些键应该传递给检测器的流程
]
# 测试数据流水线
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])  # test 时不传递 gt_label
]
data = dict(
    samples_per_gpu=32,  # 单个 GPU 的 Batch size
    workers_per_gpu=2,  # 单个 GPU 的 线程数
    train=dict(  # 训练数据信息
        type=dataset_type,  # 数据集名称
        data_prefix='data/imagenet/train',  # 数据集目录，当不存在 ann_file 时，类别信息从文件夹自动获取
        pipeline=train_pipeline),  # 数据集需要经过的 数据流水线
    val=dict(  # 验证数据集信息
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',  # 标注文件路径，存在 ann_file 时，不通过文件夹自动获取类别信息
        pipeline=test_pipeline),
    test=dict(  # 测试数据集信息
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline))
evaluation = dict(  # evaluation hook 的配置
    interval=1,  # 验证期间的间隔，单位为 epoch 或者 iter， 取决于 runner 类型。
    metric='accuracy')  # 验证期间使用的指标。
