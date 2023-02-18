# 用于构建优化器的配置文件。支持 PyTorch 中的所有优化器，同时它们的参数与 PyTorch 里的优化器参数一致。
optimizer = dict(type='SGD',  # 优化器类型
                 lr=0.1,  # 优化器的学习率，参数的使用细节请参照对应的 PyTorch 文档。
                 momentum=0.9,  # 动量(Momentum)
                 weight_decay=0.0001)  # 权重衰减系数(weight decay)。
# optimizer hook 的配置文件
optimizer_config = dict(grad_clip=None)  # 大多数方法不使用梯度限制(grad_clip)。
# 学习率调整配置，用于注册 LrUpdater hook。
lr_config = dict(policy='step',  # 调度流程(scheduler)的策略，也支持 CosineAnnealing, Cyclic, 等。
                 step=[30, 60, 90])  # 在 epoch 为 30, 60, 90 时， lr 进行衰减
runner = dict(type='EpochBasedRunner',  # 将使用的 runner 的类别，如 IterBasedRunner 或 EpochBasedRunner。
              max_epochs=100)  # runner 总回合数， 对于 IterBasedRunner 使用 `max_iters`
