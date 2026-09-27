# JEPA 表征学习与视觉任务应用技术学习文档

## 1. 学习主题概述

今天主要学习了 **JEPA（Joint-Embedding Predictive Architecture，联合嵌入预测架构）** 的基本思想，并进一步梳理了它与普通 Joint-Embedding Architecture、I-JEPA、World Model 以及 CLIP 视觉主干之间的关系。

JEPA 的核心思想可以概括为：

> **不直接预测原始像素，而是在特征空间中，根据已知信息预测未知区域或未来状态的表征。**

这一思想一方面可以用于自监督视觉表征学习，另一方面也为后续的 latent-space World Model 提供了重要基础。

------

# 2. Joint-Embedding Architecture

普通的 Joint-Embedding Architecture 会分别对两个相关输入 \(x\) 和 \(y\) 进行编码：

\[ s_x=f_\theta(x) \]\[ s_y=f_{\bar{\theta}}(y) \]

然后直接约束二者在 embedding 空间中的关系：

\[ L=D(s_x,s_y) \]

其中：

- \(x\) 和 \(y\) 可以是同一张图片的不同增强；
- \(s_x\) 和 \(s_y\) 是对应的特征表示；
- \(D(\cdot)\) 表示两个 embedding 之间的距离或相似度函数。

其核心思想是：

> 对于语义相同或者相关的两个输入，它们经过编码后的表征也应该保持一致。

例如一张狗的彩色图片和对应的灰度图片：

```
彩色图片 x → Encoder → s_x
                        ↓
                     D(s_x,s_y)
                        ↑
灰度图片 y → Encoder → s_y
```

这种方法主要学习的是**表征不变性**。

------

# 3. JEPA 与普通 Joint Embedding 的区别

JEPA 在 Joint-Embedding Architecture 的基础上增加了一个关键模块：

**Predictor。**

普通 Joint Embedding 是：

\[ s_x \leftrightarrow s_y \]

直接比较两个 encoder 得到的表示。

而 JEPA 是：

\[ s_x \xrightarrow{g_\phi} \hat{s}_y \]

再比较：

\[ \hat{s}_y \]

和真实目标表示：

\[ s_y \]

因此 JEPA 的目标可以写成：

\[ L=D(\hat{s}_y,s_y) \]

其中：

\[ \hat{s}_y=g_\phi(s_x) \]

整个流程为：

```
x
↓
x-Encoder
↓
s_x
↓
Predictor
↓
ŝ_y ───────┐
            │
            ↓
         Distance
            ↑
            │
s_y ────────┘
↑
y-Encoder
↑
y
```

因此，两者最核心的区别可以概括为：

**Joint Embedding：**

> 将两个输入分别编码，然后直接进行表征对齐。

**JEPA：**

> 根据一个输入的表征，主动预测另一个输入应该具有怎样的表征。

即：

\[ \boxed{\text{Joint Embedding：Alignment}} \]

而：

\[ \boxed{\text{JEPA：Prediction}} \]

------

# 4. I-JEPA 的核心思想

I-JEPA 是 JEPA 思想在图像领域中的典型实现。

其基本任务是：

> 根据图像中已经看到的区域，预测被遮挡目标区域对应的 latent representation。

假设一张图像经过 ViT 被划分成多个 patch：

```
□ □ □ □ □ □
□ □ ■ ■ □ □
□ □ ■ ■ □ □
□ □ □ □ □ □
□ ■ ■ □ □ □
□ ■ ■ □ □ □
```

其中：

- 白色区域为 context；
- 黑色区域为 target。

Context Encoder 只处理可见区域：

\[ s_x=f_\theta(x) \]

Target Encoder 则根据完整图像产生目标区域的真实表示：

\[ s_y=f_{\bar{\theta}}(y) \]

Predictor 根据 context representation 和 target 位置信息进行预测：

\[ \hat{s}_y=g_\phi(s_x,p_y) \]

其中：

\[ p_y \]

表示目标 patch 的空间位置。

最后计算：

\[ L_{\text{JEPA}} = \left\| \hat{s}_y-s_y \right\|_2 \]

因此 I-JEPA 并不是恢复：

\[ RGB \]

而是在恢复：

\[ \text{Latent Representation} \]

------

# 5. JEPA 与 MAE 的区别

MAE 和 JEPA 都会遮挡图像的一部分，但是二者预测目标完全不同。

MAE 的思想为：

\[ \text{context} \rightarrow \text{pixel reconstruction} \]

例如模型需要恢复：

- RGB 值；
- 纹理；
- 光照；
- 颜色；
- 像素细节。

而 JEPA 为：

\[ \text{context} \rightarrow \text{latent representation} \]

它不要求模型精确恢复：

> 被遮挡区域具体长什么样。

而要求模型理解：

> 这块区域在语义空间中应该是什么。

因此：

\[ \boxed{ \text{MAE：预测 pixels} } \]\[ \boxed{ \text{JEPA：预测 representations} } \]

JEPA 更倾向于忽略那些不可预测、但又不重要的像素级细节，例如：

- 随机纹理；
- 光照；
- 阴影；
- 背景噪声。

而重点学习：

- 物体结构；
- 区域关系；
- 高层语义；
- 场景上下文。

------

# 6. Predictor 的作用与结构

JEPA 中最核心的新增模块就是 Predictor：

\[ g_\phi \]

Predictor 的作用不是负责提取原始图像特征，而是：

> 根据已经得到的 context representation，推理目标区域应该具有怎样的 latent representation。

在 I-JEPA 中，Predictor 并不是简单的一层 MLP，而通常是一个**较轻量的 Transformer / ViT 网络**。

其输入可以表示为：

\[ [s_x,p_y] \]

其中：

- \(s_x\) 为 context patch features；
- \(p_y\) 为目标区域的位置 token。

经过 Predictor：

\[ \hat{s}_y=g_\phi(s_x,p_y) \]

例如：

```
Context tokens

z1  z2  z3
z4      z6
z7  z8  z9
```

如果需要预测中间区域：

```
z5 = ?
```

Predictor 就要综合：

\[ z_1,z_2,\cdots,z_9 \]

之间的关系，并结合：

\[ p_5 \]

所表示的位置，预测：

\[ \hat{z}_5 \]

Transformer 中的 Attention 非常适合完成这种“利用多个 context token 推断目标 token”的任务。

因此可以将 Predictor 理解为：

> **JEPA 中真正执行 latent-space prediction 的模块。**

------

# 7. 多个 Target 与 Predictor 的关系

I-JEPA 中，一张图片通常会采样多个 target block。

例如：

```
Target 1：蓝色区域
Target 2：红色区域
Target 3：黄色区域
```

这些 target block 之间允许存在一定程度的空间重叠。

需要特别注意：

图中即使画出了多个 Predictor，也不代表使用多个参数不同的网络。

实际上通常使用的是：

\[ \boxed{\text{同一个 Predictor }g_\phi} \]

只是针对不同 target 进行多次调用：

\[ \hat{s}_{t_1} = g_\phi(s_x,p_{t_1}) \]\[ \hat{s}_{t_2} = g_\phi(s_x,p_{t_2}) \]\[ \hat{s}_{t_3} = g_\phi(s_x,p_{t_3}) \]

最后分别与 Target Encoder 得到的真实表示比较：

\[ L= \sum_i D(\hat{s}_{t_i},s_{t_i}) \]

------

# 8. Target Encoder 为什么存在

I-JEPA 中通常存在两个 encoder：

```
Context Encoder
Target Encoder
```

Context Encoder：

\[ f_\theta \]

通过梯度正常训练。

Target Encoder：

\[ f_{\bar{\theta}} \]

通常不直接接受梯度更新，而是使用 Context Encoder 参数的指数移动平均进行更新：

\[ \bar{\theta} \leftarrow \tau\bar{\theta} + (1-\tau)\theta \]

Target Encoder 的作用可以理解为：

> 为 Predictor 提供一个相对稳定的“正确答案”。

即：

\[ s_y=f_{\bar{\theta}}(y) \]

Predictor 需要让：

\[ \hat{s}_y \]

逐渐逼近：

\[ s_y \]

从而训练 Context Encoder 学习更好的视觉表示。

------

# 9. JEPA 本质上是表征学习方法

需要明确一个概念：

**JEPA 本身不是一种 Backbone。**

例如：

- ViT；
- ResNet；
- ConvNeXt；

这些属于网络 Backbone。

而：

- MAE；
- DINO；
- CLIP；
- JEPA；

更接近于**表征学习框架或者训练方法**。

因此：

> JEPA 解决的是“如何训练 Encoder 学到更好的 representation”，而不是规定 Encoder 一定是什么网络。

I-JEPA 使用 ViT 作为 backbone，只是 JEPA 的一种具体实现。

------

# 10. JEPA 与下游任务的关系

I-JEPA 本身主要研究的是：

\[ \boxed{\text{Self-Supervised Representation Learning}} \]

其预训练阶段并不直接要求完成：

- 分类；
- 检测；
- 分割；
- 跟踪。

JEPA 训练过程中：

```
Image
↓
Encoder
↓
Predictor
↓
JEPA Loss
```

真正希望训练好的核心模块是：

\[ Encoder \]

Predictor 更多是为了训练 Encoder 而存在的。

完成预训练之后，可以将 Predictor 去掉：

```
JEPA Predictor
     ×
     ↓

Image
 ↓
Encoder
 ↓
Downstream Head
```

然后 Encoder 可以用于：

```
Encoder
 ├── Classification Head
 ├── Detection Head
 └── Segmentation Head
```

所以 JEPA 和 MAE 类似：

> 预训练中的预测模块并不一定用于真正的下游推理。

------

# 11. 为什么 JEPA 与 World Model 密切相关

World Model 的目标是：

> 建立世界的内部状态表示，并预测这个状态如何变化。

一个典型 World Model 可以写成：

\[ z_t,a_t \rightarrow \hat{z}_{t+1} \]

其中：

- \(z_t\)：当前世界状态；
- \(a_t\)：当前执行的动作；
- \(\hat{z}_{t+1}\)：预测的下一时刻状态。

传统生成式 World Model 可能进行：

\[ I_t,a_t \rightarrow \hat{I}_{t+1} \]

即预测下一帧图像。

这意味着模型需要预测：

- 每一个像素；
- 光照变化；
- 阴影；
- 背景；
- 纹理；
- 不重要的随机细节。

而 JEPA 提出：

> 没有必要预测世界所有像素，只需要预测世界的重要抽象状态。

于是：

\[ I_t \rightarrow z_t \]

再进行：

\[ z_t,a_t \rightarrow \hat{z}_{t+1} \]

例如机器人推动杯子时，真正需要预测的是：

```
杯子在哪里
↓
执行向右推动动作
↓
杯子之后在哪里
```

并不需要准确预测：

```
杯子反光如何变化
桌面纹理如何变化
阴影具体移动几个像素
```

所以 JEPA 的核心思想：

\[ \boxed{ \text{Predict the world in representation space} } \]

与 World Model 非常契合。

------

# 12. I-JEPA 为什么还不能直接称为 World Model

虽然 JEPA 与 World Model 的思想关系很密切，但是 **I-JEPA 本身还不是完整的 World Model**。

因为 I-JEPA 处理的是：

\[ \text{Image Context} \rightarrow \text{Image Target} \]

本质还是静态图像内部的预测。

它没有显式包含：

- 时间；
- 状态转移；
- action；
- environment dynamics。

所以 I-JEPA 更准确地属于：

\[ \boxed{\text{Representation Learning}} \]

进一步发展到视频以后：

\[ z_t \rightarrow \hat{z}_{t+1} \]

就开始具有时间动态。

如果再加入动作：

\[ z_t,a_t \rightarrow \hat{z}_{t+1} \]

就逐渐成为真正意义上的 World Model。

因此可以把发展关系理解为：

```
I-JEPA
  ↓
学习空间上下文关系

V-JEPA
  ↓
加入时间变化

Action-conditioned prediction
  ↓
加入动作

World Model
  ↓
预测未来状态并进行规划/控制
```

因此更准确的说法是：

> **JEPA 为 latent-space World Model 提供了重要的表征学习与预测思想基础。**

------

# 13. JEPA 与 CLIP 的关系

CLIP 主要学习的是：

\[ \text{Image} \leftrightarrow \text{Text} \]

即：

\[ z_I \approx z_T \]

它重点解决：

> 图像和语言之间的语义对齐。

而 I-JEPA 解决的是：

\[ \text{Visual Context} \rightarrow \text{Visual Target} \]

重点学习：

> 图像内部不同区域之间的视觉结构与语义关系。

因此两者解决的是不同问题。

可以概括为：

\[ \boxed{ CLIP： Vision-Language Alignment } \]\[ \boxed{ JEPA： Visual Predictive Representation Learning } \]

所以理论上完全可以将 JEPA 思想加入 CLIP Visual Encoder。

------

# 14. JEPA 加入 CLIP 的一种形式

假设原始分类模型为：

```
Image
 ↓
CLIP Visual Encoder
 ↓
Global Feature
 ↓
Classifier
 ↓
Prediction
```

可以增加一个 JEPA 自监督分支：

```
                          → Classifier
                         /       ↓
Image → CLIP Visual Tower      L_cls
                         \
                          → Predictor
                               ↓
                             L_JEPA
```

联合训练目标可以写成：

\[ L = L_{\text{cls}} + \lambda L_{\text{JEPA}} \]

但是对于存在严重标签噪声的数据，更适合考虑两阶段方案。

------

# 15. 当前分类数据中的标签噪声问题

当前分类任务存在一个比较明显的数据质量问题：

> 一个类别文件夹中存在大量实际上不属于该类别的图像。

例如：

```
Cat 文件夹

cat_001.jpg      正确
cat_002.jpg      正确
cat_003.jpg      正确
dog_021.jpg      错误
car_007.jpg      错误
```

训练分类模型时，文件夹名会被直接作为 label。

因此：

\[ x=\text{dog} \]

但是：

\[ y=\text{cat} \]

普通交叉熵：

\[ L_{\text{CE}} = -\log P(y|x) \]

会强制模型学习：

\[ \text{dog feature} \rightarrow \text{cat} \]

这会产生错误梯度，并进一步污染 CLIP Visual Encoder 原本较好的 feature space。

------

# 16. 为什么 JEPA 特别适合这种噪声数据

JEPA 最有价值的一点是：

\[ \boxed{\text{它不需要类别标签}} \]

对于一张狗的图片，即使它被错误放在 Cat 文件夹：

```
dog.jpg

Folder Label = Cat
```

JEPA 在训练时根本不会使用：

\[ y=\text{Cat} \]

它只会执行：

```
狗的可见区域
       ↓
CLIP Encoder
       ↓
Context Feature
       ↓
Predictor
       ↓
预测狗的其他区域 Feature
```

所以无论：

```
dog.jpg
```

放在：

```
Cat/
Dog/
Car/
```

哪个文件夹里，其 JEPA 学习过程基本都不受影响。

因此：

> JEPA 可以让 CLIP 在接触错误标签之前，先根据图片自身学习当前数据集中的视觉结构与语义规律。

------

# 17. 当前任务更加推荐的 JEPA 使用方式

相比直接使用：

\[ L=L_{\text{CE}}+\lambda L_{\text{JEPA}} \]

当前任务更值得尝试的是：

## Stage 1：JEPA 自监督领域适应

首先忽略全部标签：

```
全部训练图像
      ↓
CLIP Visual Encoder
      ↓
JEPA Self-Supervised Learning
      ↓
Domain-adapted CLIP
```

这一阶段只优化：

\[ L_{\text{JEPA}} \]

完全不使用文件夹类别。

因此即使存在大量错误标签，也不会产生错误类别梯度。

其目的为：

> 让 CLIP Visual Encoder 首先适应当前数据集真实的视觉分布。

------

## Stage 2：监督分类训练

完成 JEPA 适应之后：

```
Image
 ↓
JEPA-adapted CLIP
 ↓
Classification Head
 ↓
Classification Loss
```

即：

\[ \boxed{ CLIP \xrightarrow{\text{JEPA}} \text{Domain-adapted CLIP} \xrightarrow{\text{Classification}} \text{Classifier} } \]

相比从一开始就使用大量脏标签训练 CLIP，这种方式能够减少错误标签对 backbone 表征学习阶段的干扰。

------

# 18. JEPA 对标签噪声的能力边界

需要注意：

> **JEPA 并不会自动识别或者修改错误标签。**

JEPA 能解决的是：

\[ \boxed{ \text{减少错误标签对 representation learning 的污染} } \]

而不是：

\[ \boxed{ \text{自动完成 noisy-label correction} } \]

也就是说，JEPA 可以让 backbone 更鲁棒，但后续分类训练仍然可能受到：

\[ \text{dog}\rightarrow\text{cat} \]

这种错误监督影响。

因此完整方案还可以进一步结合：

- noisy sample detection；
- prototype similarity；
- small-loss selection；
- consistency learning；
- pseudo-label correction。

不过这些属于下一阶段的问题。

------

# 19. 今天形成的整体认识

今天对 JEPA 的理解可以最终整理成以下逻辑链：

```
Joint Embedding
       ↓
两个输入分别编码
       ↓
直接进行 Feature Alignment
       ↓
JEPA
       ↓
加入 Predictor
       ↓
根据已知 Feature 预测未知 Feature
       ↓
I-JEPA
       ↓
预测图像被遮挡区域的 Representation
       ↓
V-JEPA
       ↓
预测视频中的时空 Representation
       ↓
Action-conditioned Prediction
       ↓
根据 State + Action 预测 Future State
       ↓
World Model
```

对应数学关系可以概括为：

普通 Joint Embedding：

\[ s_x \leftrightarrow s_y \]

I-JEPA：

\[ s_x \rightarrow \hat{s}_y \]

视频预测：

\[ z_t \rightarrow \hat{z}_{t+1} \]

World Model：

\[ z_t,a_t \rightarrow \hat{z}_{t+1} \]

整个思想的发展本质上是从：

> **“让两个表示相似”**

逐渐发展为：

> **“根据当前已知信息，在 latent space 中预测未知世界。”**

------

# 20. 总结

JEPA 并不是一种新的视觉 backbone，而是一种**基于 latent representation prediction 的自监督表征学习框架**。与普通 Joint Embedding 直接进行 feature alignment 不同，JEPA 引入 Predictor，根据 context representation 主动预测 target representation，从而迫使 Encoder 学习更加高层、稳定且具有上下文关系的视觉特征。

I-JEPA 首先将这一思想应用于静态图像，通过预测被遮挡区域的 embedding 学习视觉结构；进一步加入时间，可以发展为视频状态预测；再加入 action，则逐渐形成 \(z_t,a_t\rightarrow z_{t+1}\) 的 World Model。

对于当前使用 CLIP Visual Encoder 的分类任务，数据中存在较严重的文件夹错标问题，因此 JEPA 具有比较明确的应用价值：**它能够完全绕开类别标签，首先利用图像自身完成 CLIP 的自监督领域适应，从而减少错误标签对 backbone 特征空间的直接污染。** 后续再在增强后的视觉表征基础上进行分类训练，会是一条值得进一步实验验证的路线。