# Batch Attribute Alignment

该方法只改变进入每个 batch 的样本组合，不增加 loss、不修改图片或模型。
输入训练集属性矩阵、二分类标签和原本 sampler 生成的 batch 索引计划。
在相邻 batch 间交换同一类别的样本，让 batch 内 Real/Fake 属性直方图更相似。

每个类别的直方图分别归一化；距离是两个直方图相减、平方求和再开方。
多属性默认等权，先除以训练集参考 batch 的平均距离统一尺度。
完整保留每轮样本出现次数以及每个 batch 原有的类别槽位。

安装：`pip install -e .`。运行：`python examples/synthetic_demo.py`。
测试：`pip install -e '.[dev]'` 后运行 `pytest`。
默认只依赖 NumPy，接入 PyTorch 可使用 `PlanBatchSampler`。

六类代表属性可使用人脸面积、垂直位置、绝对 yaw、图像像素面积的 log2、
锐度、黑色边框比例。属性需要自行提取，单位和预处理须保持一致。
训练代码保持原有设置，两组实验共享原始 epoch 计划和逐图增强随机性。
验证集和测试集不能用于选择属性分箱、权重或 batch。

距离下降不保证检测性能提高，也不能单独证明模型学到了 shortcut。
仓库中的 Xception 结果为单 seed 历史记录，包含负向结果。
详见 [完整方法](method.md) 和 [主说明](../README.md)。
