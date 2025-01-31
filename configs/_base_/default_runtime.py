# Checkpoint hook 的配置文件。
checkpoint_config = dict(interval=1)  # 保存的间隔是 1，单位会根据 runner 不同变动，可以为 epoch 或者 iter。
# 日志配置信息。
log_config = dict(
    interval=100,  # 打印日志的间隔， 单位 iters
    hooks=[
        dict(type='TextLoggerHook'),  # 用于记录训练过程的文本记录器(logger)。
        # dict(type='TensorboardLoggerHook')  # 同样支持 Tensorboard 日志
    ])

dist_params = dict(backend='nccl')  # 用于设置分布式训练的参数，端口也同样可被设置。
log_level = 'INFO'  # 日志的输出级别。
resume_from = None  # 从给定路径里恢复检查点(checkpoints)，训练模式将从检查点保存的轮次开始恢复训练。
workflow = [('train', 1)]  # runner 的工作流程，[('train', 1)] 表示只有一个工作流且工作流仅执行一次。
work_dir = 'work_dir'  # 用于保存当前实验的模型检查点和日志的目录文件地址。
