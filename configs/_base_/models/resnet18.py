# model settings
model = dict(
    type='ImageClassifier',     # 分类器类型
    backbone=dict(
        type='ResNet',          # 主干网络类型
        depth=50,               # 主干网网络深度， ResNet 一般有18, 34, 50, 101, 152 可以选择
        num_stages=4,           # 主干网络状态(stages)的数目，这些状态产生的特征图作为后续的 head 的输入。
        out_indices=(3, ),      # 输出的特征图输出索引。越远离输入图像，索引越大
        frozen_stages=-1,       # 网络微调时，冻结网络的stage（训练时不执行反相传播算法），若num_stages=4，backbone包含stem 与 4 个 stages。frozen_stages为-1时，不冻结网络； 为0时，冻结 stem； 为1时，冻结 stem 和 stage1； 为4时，冻结整个backbone
        style='pytorch'),       # 主干网络的风格，'pytorch' 意思是步长为2的层为 3x3 卷积， 'caffe' 意思是步长为2的层为 1x1 卷积。
    neck=dict(type='GlobalAveragePooling'),    # 颈网络类型
    head=dict(
        type='LinearClsHead',     # 线性分类头，全连接
        num_classes=1000,         # 输出类别数，这与数据集的类别数一致
        in_channels=2048,         # 输入通道数，这与 neck 的输出通道一致
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0), # 损失函数配置信息
        topk=(1, 5),              # 评估指标，Top-k 准确率， 这里为 top1 与 top5 准确率
    ))
