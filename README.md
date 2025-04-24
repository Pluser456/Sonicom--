# 代码功能

使用 Sonicom 人耳数据集作为神经网络输入，预测个性化HRTF。存储库中的代码仅包含神经网络部分，不包含数据文件。

# 日程更新

## 2025/4/24

### 目前结果（单位 dB）

学姐复现代码：5.7

Resnet（Hutubs）：4.9

Resnet（Sonicom）：5.2

Resnet（Hutubs）Single Frequency：4.2

Average：5.5

Vit（Sonicom）：5.2

### 现在的任务：奚顺加，欧阳洋

1. 修改模型为单个方向看看效果如何，来确定是否是方位角无法识别到导致的欠拟合

2. 修改模型为单个频率点看看效果如何，看看能不能识别出来方位角

### 后续的任务：王润邦

1. 把pos_embedding放到transfromer较前面的层中，或者是参考多模态transformer（文本和图像）的输入来修改模型

2. 复现vae的论文代码

### 其他的数据集：

1. 数据集可以考虑widespread，1000个数据集，单纯耳朵的模型，没有人头模型

2. 体素化数据张量，比如说3维位置取128x128点，需要关注模型的尺度是否不变

### 更远的任务：

1. 基于transformer的neural processes
