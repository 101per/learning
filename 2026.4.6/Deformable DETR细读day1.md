# Deformable DETR



> DETR has its own issues: (1) It requires much longer training epochs to converge than the existing object detectors. For example, on the COCO (Lin et al., 2014) benchmark, DETR needs 500 epochs to converge, which is around 10 to 20 times slower than Faster R-CNN (Ren et al., 2015). (2) DETR delivers relatively low performance at detecting small objects.

DETR**训练效率低~1~**并且**处理不了高分辨率图片~2~**，还有**小目标检测效果不好~3~**。DeformableDETR解决了前两个问题，改善了第三个问题。接下来具体看看是如何解决的。



> we propose Deformable DETR, which mitigates the slow convergence and high complexity issues of DETR. It combines the best of the sparse spatial sampling of deformable convolution, and the relation modeling capability of Transformers. We propose the deformable attention module, which attends to a small set of sampling locations as a pre-filter for prominent key elements out of all the feature map pixels.
>
> ![image-20260406215435148](./Deformable DETR细读.assets/image-20260406215435148.png)

DETR中backbone一般用的是**CNN来提取特征**的，还没有用到ViT。同样，在Deformable DETR中，并没有改动backbone，而是改了transfomer Encoder和Decoder。



## 补充DETR内容：

DETR 的主干流程：

**图像 → CNN backbone → feature map → Transformer Encoder → Transformer Decoder → 分类头和框回归头**

更具体一点：

- 输入图像 (x)
- 用 CNN 提取特征，得到二维特征图
- 把二维特征图展平成一串 token （要加入位置编码）
- 送入 Transformer Encoder 做全局建模
- 再送入 Transformer Decoder，与一组可学习的 **object queries** 交互
- Decoder 的每个 query 输出一个候选目标
- 最后每个 query 各自预测：
  - 一个类别
  - 一个边界框



Encoder 输入是一串图像 token：
$$
X = [x_1, x_2, \dots, x_{HW}] \in \mathbb{R}^{HW \times d}
$$
每个X~i~ 代表一个pixel

举个直觉例子。

假设图中有一只猫，猫头在一个位置，猫身在另一个位置，沙发在远处。

对于“猫头”位置，Encoder 的 self-attention 可能会关注：

- 猫耳朵位置
- 猫身体位置
- 猫尾巴位置
- 背景位置较少

于是编码后，这个位置就不只是“局部像猫耳朵”，而是更像：

> “这个位置很可能属于一整只猫的一部分”

这对检测非常重要，因为检测需要的是 **对象级表示**，不是单点局部纹理。


$$
E = [e_1, e_2, \dots, e_{HW}] \in \mathbb{R}^{HW \times d}
$$
它和输入长度一样，但每个位置的特征已经变成了：

> **融合了全局上下文的图像记忆（memory）**

这个 memory 会传给 Decoder。

所以你可以把 Encoder 输出理解成：

> “整张图经过全局关系建模后的高层表示库”

------

# 四、Transformer Decoder 详细讲解

现在到 DETR 最有代表性的部分了。

------

## 1. Decoder 的核心任务

Decoder 不是逐像素预测，也不是滑窗预测。
它做的是：

> **用一组固定数量的 query，去整张图的 memory 中检索目标。**

这和传统检测器非常不一样。

传统检测器通常是：

- 在很多 anchor 上预测
- 或对每个位置都分类回归

而 DETR 的 Decoder 是：

- 先放 (N) 个 query
- 每个 query 尝试去“找一个目标”
- 最后一个 query 对应一个输出框

------

## 2. Decoder 的输入是什么

Decoder 有两个主要输入：

### (1) Encoder 输出的 memory

[
E \in \mathbb{R}^{HW \times d}
]

### (2) 一组 object queries

[
Q^{obj} = [q_1^{obj}, q_2^{obj}, \dots, q_N^{obj}] \in \mathbb{R}^{N \times d}
]

这里 (N) 是固定的，比如 100。

这些 object queries 是 **可学习参数**，不是从图像里算出来的。

------

## 3. object query 到底是什么

这是很多人第一次看 DETR 时最迷惑的点。

你可以把 object query 理解成：

> 一组“检测槽位”或者“一组询问向量”。

每个 query 就像在问：

- 图里有没有一个目标可以由我负责？
- 我该去关注哪里？
- 我最后应该输出什么类别和框？

它不是某个具体物体类别的模板，也不是某个空间位置的 anchor。
它更像一个抽象的“候选目标指针”。

------

## 4. Decoder 每层做什么

每层 Decoder 有三步：

### (1) query 之间先做 self-attention

### (2) query 再和 Encoder memory 做 cross-attention

### (3) 再经过 FFN

------

# 五、Decoder 第一步：Self-Attention

设当前 Decoder 输入是：

[
Y = [y_1, y_2, \dots, y_N]
]

开始时可以看作 object queries 本身。

先做 query 之间的 self-attention：

[
\tilde{Y} = \mathrm{LN}(Y + \mathrm{MSA}(Y))
]

------

## 这一步的意义是什么

这一步不是看图，而是 **让不同 query 彼此沟通**。

为什么要沟通？

因为如果 100 个 query 完全独立，它们可能都去抢同一个目标。
self-attention 可以让它们互相“协调”：

- 某个 query 已经强烈关注左边那只狗
- 另一个 query 就少去重复关注那只狗
- 有些 query 会转而去关注别的区域

所以 Decoder 的 self-attention 有点像：

> **多个候选检测槽位之间的竞争与协商机制**

------

# 六、Decoder 第二步：Cross-Attention

这是 Decoder 真正“读图”的步骤。

把 Decoder 当前状态 (\tilde{Y}) 作为 query，
把 Encoder 输出 (E) 作为 key 和 value：

[
Q = \tilde{Y}W_Q,\quad K = EW_K,\quad V = EW_V
]

然后计算：

[
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
]

注意这里矩阵维度变了：

- query 数量是 (N)
- memory 长度是 (HW)

所以注意力矩阵大小大致是：

[
N \times HW
]

即：

> 每个 object query 会对整张图所有位置分配注意力。

------

## 这一步的直觉是什么

对于第 (i) 个 object query，它会计算：

[
\alpha_{ij} = \mathrm{softmax}_j \left( \frac{q_i k_j^\top}{\sqrt{d_k}} \right)
]

这里 (j) 遍历整张图的所有空间位置。

然后得到：

[
o_i = \sum_{j=1}^{HW} \alpha_{ij} v_j
]

也就是说：

> 第 (i) 个 query 从整张图中挑选它认为和自己最相关的区域，并聚合成一个对象级表示。

你可以把它想成：

- Encoder 输出的是“全图特征地图册”
- Decoder 的某个 query 像一个搜索器
- 它去全图里找到与某个潜在目标相关的位置
- 把这些位置的信息整合起来
- 最后形成“这个 query 对应的目标表示”

------

## 举例理解

假设图中有一辆车和一个人。

- query 1 经过 cross-attention 后，主要关注车的各个部分
- query 2 主要关注人的头、身体、腿
- query 3 可能关注背景，最后预测 no object
- query 4 也许一开始和 query 1 有冲突，但经过多层 Decoder 后逐渐分工

于是，每个 query 最终都变成一个候选目标向量。

------

# 七、Decoder 第三步：FFN

cross-attention 后再经过 FFN：

[
Y' = \mathrm{LN}(O + \mathrm{FFN}(O))
]

它的作用是进一步非线性变换，增强表示能力。

经过多层 Decoder 反复迭代之后，query 表示会越来越明确：

- 我负责什么目标
- 该关注哪里
- 这个目标是什么类别
- 边界框大概在哪

------

# 八、Decoder 的输出是什么

最终 Decoder 输出：

[
Y^{(L)} = [y_1^{(L)}, y_2^{(L)}, \dots, y_N^{(L)}]
]

每个 (y_i^{(L)} \in \mathbb{R}^d) 都对应一个 query 的最终表示。

然后对每个 query，接两个预测头：

### 1. 分类头

输出类别概率：

[
\hat{c}_i = \mathrm{Linear}(y_i^{(L)})
]

这里类别里还包括一个特殊类：

[
\varnothing
]

表示 **no object**。

### 2. 边框回归头

输出归一化框参数：

[
\hat{b}_i = \mathrm{MLP}(y_i^{(L)})
]

一般表示为：

[
\hat{b}_i = (c_x, c_y, w, h)
]

取值在 ([0,1]) 范围内，相对于图像大小归一化。

------

# 九、为什么 Decoder 需要固定数量的 queries

因为 DETR 把检测任务改写成了：

> **集合预测问题**

它不是说“每个位置都预测一个框”，而是：

- 我预设 (N) 个输出槽位
- 每个槽位输出一个候选目标
- 最后通过 bipartite matching，让预测集合和真实目标集合一一匹配

如果图中目标少于 (N)，多出来的 query 就预测 no object。

所以：

- (N) 要大于图中可能的最大目标数
- 并不是每个 query 都一定对应真实物体

------

# 十、Encoder 和 Decoder 的本质分工

这个你一定要彻底分清。

## Encoder 做什么

**对图像所有位置进行全局上下文建模**

它回答的是：

- 图中各区域之间有什么关系？
- 某个位置和全图其他位置如何交互？
- 经过上下文整合后，每个位置应该如何表达？

所以 Encoder 输出的是 **上下文化的空间特征**。

------

## Decoder 做什么

**从这些空间特征里，提取对象级表示**

它回答的是：

- 图里有哪些目标？
- 第 (i) 个 query 应该负责哪个目标？
- 这个目标类别是什么？框在哪里？

所以 Decoder 输出的是 **对象级 prediction slots**。

------

# 十一、为什么说 Decoder 比 Encoder 更“检测专用”

Encoder 是比较通用的。
很多视觉任务都能用 Encoder 做全局特征建模。

但 Decoder 的设计更有 DETR 风格，因为它引入了：

- object queries
- query 与图像特征的 cross-attention
- 固定数量输出槽位
- 一对一匹配

这套东西就是 DETR 从“密集预测”转向“集合预测”的关键。

------

# 十二、和传统检测器相比，DETR 的 Decoder 特别在哪里

传统检测器一般是：

### 两阶段

- 先产生 proposals
- 再分类和回归

### 一阶段

- 在密集位置/anchor 上直接预测

而 DETR：

- 没有 anchor
- 没有 NMS
- 没有 proposal pipeline
- 直接让 query 去全图检索目标

所以你可以说：

> **DETR 的 Decoder 相当于一组可学习的目标提议器 + 目标识别器，但它们是端到端联合学习的。**

------

# 十三、DETR 中 Encoder/Decoder 的一个核心问题

你后面读 Deformable DETR 时会特别明显感受到这一点。

原始 DETR 的问题之一是：

## 1. Encoder 的 self-attention 计算量大

因为要在所有 (HW) 个位置之间两两计算关系。

复杂度近似是：

[
O((HW)^2)
]

图像分辨率高时很贵。

------

## 2. Decoder 的 cross-attention 也要看全图

每个 query 都要对所有空间位置做注意力。

这意味着：

- 学习难
- 收敛慢
- 小目标不容易学好

这也是为什么 Deformable DETR 要把“全局密集 attention”改成“稀疏采样 attention”。

所以你现在理解原始 DETR 的 Encoder/Decoder，后面看 Deformable DETR 才能明白：

> 它改的不是大框架，而是注意力的实现方式。

# 

### Encoder

> 把整张图每个位置都变成“看过全局”的特征

### Decoder

> 用一组查询向量去全图特征里找目标，并让每个查询最终负责一个目标

