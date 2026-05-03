# 基于医学 Prompt 的 SVD 文本子空间构建：处理流程与技术理解报告

## 1. 项目背景

为了将模型进一步拓展到：

```text
持续学习（Continual Learning）
MoE（Mixture of Experts）
Prompt 路由
任务识别
跨病灶专家选择
```

需要一种轻量、可解释、可扩展的方法，对不同任务的文本语义进行建模。

本项目选择使用 **SVD（Singular Value Decomposition，奇异值分解）** 对医学 Prompt 的文本 embedding 进行子空间建模，用于描述不同病灶任务的语义结构，并作为后续专家路由依据。

------

# 2. 核心思想概述

对于每一类病灶任务，收集若干文本 Prompt：

```text
A breast ultrasound image showing a malignant lesion...
A colonoscopy image showing a yellow round polyp...
A brain MRI image showing a pituitary tumor...
```

通过冻结文本编码器（CLIP / MedCLIP / UniMedCLIP）将 Prompt 编码为向量：

[
x_i \in \mathbb{R}^{d}
]

其中：

```text
d = 文本 embedding 维度（实际实验中为 512）
```

然后将同类 Prompt 堆叠成矩阵：

[
X \in \mathbb{R}^{M \times d}
]

其中：

```text
M = Prompt 数量
d = embedding 维度
```

再通过 SVD 提取该任务的主要语义方向：

[
X = U\Sigma V^T
]

保留前 k 个主方向，形成该任务的文本语义子空间。

------

# 3. Prompt 数据预处理流程

## 3.1 原始数据来源

多个医学数据集，每个数据集包含对应 Prompt 表格，例如：

```text
BKAI
Kvasir
BUSI
BUID
BTMRI
ISIC
Covid19
...
```

每个数据集下包含：

```text
Train_text.xlsx
Test_text_original.xlsx
...
```

最终仅使用：

```text
Train_text.xlsx
```

构建 SVD 子空间。

------

## 3.2 Prompt 清洗策略

对训练 Prompt 进行统一清洗，生成：

```text
all_train_text_svd_prompt_100_clean.xlsx
```

清洗内容包括：

### （1）去重

删除重复 Prompt，减少模板冗余。

### （2）修复异常文本

例如：

```text
of breast of the breast
center region of breast
showing the breast scan a
```

### （3）限制数量

每个数据集最多保留：

```text
100 条 Prompt
```

用于平衡不同数据集规模。

------

# 4. 按病灶类型分组构建子空间

为了提升语义稳定性，不按单数据集建模，而按**同类病灶任务**合并。

------

## 4.1 分组策略

### Colonoscopy Polyp

```text
BKAI
Kvasir
ClinicDB
ColonDB
CVC300
```

### Breast Ultrasound Lesion

```text
BUSI
BUSUC
BUSBRA
BUID
UDIAT
```

### Brain MRI Tumor

```text
BTMRI
BRISC
```

### Skin Lesion

```text
ISIC
UWaterlooSkinCancer
```

### Chest X-ray Infection

```text
Covid19
```

------

## 4.2 分组原因

相比单数据集构建：

```text
Prompt 数量更多
语义更稳定
子空间更鲁棒
适合专家级路由
```

------

# 5. 文本编码器处理

使用冻结文本编码器：

```text
UniMedCLIP / CLIP / MedCLIP
```

对每条 Prompt 编码：

[
x_i = Encoder(prompt_i)
]

实际运行结果：

```text
embedding_dim = 512
```

说明文本编码器正常工作。

同时进行：

```text
L2 Normalize
```

即：

[
x_i = \frac{x_i}{|x_i|}
]

提升几何稳定性。

------

# 6. SVD 子空间构建流程

对于某一任务组 Prompt embedding：

[
X \in \mathbb{R}^{M \times 512}
]

先求均值：

[
\mu = \frac{1}{M}\sum x_i
]

中心化：

[
X_c = X - \mu
]

再做 SVD：

[
X_c = U\Sigma V^T
]

其中：

```text
V 的列向量表示语义主方向
```

------

## 6.1 子空间基

保留前 k 个主方向：

[
B = V[:,1:k]
]

得到：

[
B \in \mathbb{R}^{512 \times k}
]

这是真正的语义子空间。

------

## 6.2 投影矩阵

进一步构造：

[
P = BB^T
]

其中：

[
P \in \mathbb{R}^{512 \times 512}
]

作用是将任意 Prompt 投影到该任务子空间。

------

# 7. 对子空间维度 k 的理解

------

## 7.1 初始理解误区

最初认为：

```text
k 越大越好
energy 越高越好
```

例如统一设置：

```text
k = 20
```

所有组指标都变高。

------

## 7.2 后续修正理解（正确）

实际上：

```text
energy 是累计指标
k 越大天然越高
```

这不代表子空间更优。

过大的 k 会导致：

```text
引入噪声维度
降低组间区分度
路由变模糊
MoE expert 混淆
```

------

## 7.3 正确认知

对子空间任务而言：

```text
不是能量最高最好
而是区分能力最好最好
```

即：

```text
Route Accuracy 最重要
```

------

# 8. 当前实验结果理解

| Group                    | k    | Energy |
| ------------------------ | ---- | ------ |
| colonoscopy_polyp        | 20   | 94%    |
| breast_ultrasound_lesion | 16   | 94%    |
| brain_mri_tumor          | 16   | 95%    |
| dermoscopic_skin_lesion  | 8    | 79%    |
| chest_xray_infection     | 4    | 86%    |

------

## 结果分析

### Brain MRI

主方向集中，模板稳定，子空间质量高。

### Colonoscopy

多样性适中，表现优秀。

### Skin Lesion

变化复杂，当前维度偏低，可提升到 12~16。

### Chest X-ray

样本少，4维足够。

------

# 9. route_prompt_by_subspace 的理解

该函数作用：

输入新 Prompt：

```text
A breast ultrasound image showing...
```

编码为向量 z。

分别投影到所有任务子空间：

```text
breast
brain
polyp
skin
chest
```

比较：

```text
投影相似度（越高越好）
残差距离（越低越好）
```

最终选择最匹配子空间：

```text
route 到对应 expert
```

------

# 10. 与持续学习 / MoE 的关系

## 持续学习

新增任务时：

```text
新增 Prompt → 新建 SVD 子空间
```

无需重训旧任务。

------

## MoE

Prompt 输入后：

```text
先 SVD 路由
再选择对应 expert
```

实现专家稀疏激活。

------

# 11. 当前技术理解总结

已经形成以下正确认知：

------

## （1）SVD 的本质

不是降维工具而已，而是：

```text
提取任务语义主方向的工具
```

------

## （2）子空间基才是核心

```text
basis [512,k]
```

是真正的任务知识表示。

------

## （3）投影矩阵只是使用工具

```text
projector [512,512]
```

用于快速计算相似度。

------

## （4）k 不是越大越好

应兼顾：

```text
表达能力
泛化能力
区分能力
```

------

## （5）最终评价指标不是 energy

而是：

```text
route accuracy
expert selection accuracy
downstream segmentation gain
```

------

# 12. 后续建议

## 短期

```text
做不同 k 的路由准确率实验
```

## 中期

```text
接入 LoRA expert / Adapter expert
```

## 长期

```text
构建 Continual Medical MoE Framework
```

------

# 13. 总结一句话

本项目已成功将多医学数据集 Prompt 通过文本编码 + SVD 建模为多个病灶语义子空间，并完成从“仅理解 SVD 数学形式”到“理解其在持续学习与 MoE 路由中的工程价值”的完整转变。