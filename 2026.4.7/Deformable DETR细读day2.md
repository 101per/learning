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

于是编码后，这个位置很可能属于一整只猫的一部分

这对检测非常重要，因为检测需要的是 **对象级表示**，不是单点局部纹理。

输出：

$$
E = [e_1, e_2, \dots, e_{HW}] \in \mathbb{R}^{HW \times d}
$$
它和输入长度一样，但每个位置的特征已经变成了**融合了全局上下文的图像记忆（memory）**这个 memory 会传给 Decoder。



Decoder 有两个主要输入：

$$
E \in \mathbb{R}^{HW \times d}
$$

$$
Q^{obj} = [q_1^{obj}, q_2^{obj}, \dots, q_N^{obj}] \in \mathbb{R}^{N \times d}
$$

这里 (N) 是固定的，比如 100。

这些 object queries 是 **可学习参数**，不是从图像里算出来的。

object queries 不是某个具体物体类别的模板，也不是某个空间位置的 anchor。它更像一个抽象的“候选目标指针”。

每层 Decoder 有三步：

(1) query 之间先做 self-attention

(2) query 再和 Encoder memory 做 cross-attention

(3) 再经过 FFN（两层MLP）

最终 Decoder 输出：
$$
Y^{(L)} = [y_1^{(L)}, y_2^{(L)}, \dots, y_N^{(L)}]
$$
每个 $(y_i^{(L)} \in \mathbb{R}^d) $都对应一个 query 的最终表示。

然后对每个 query，接两个预测头：

1. 分类头: $\hat{c}_i = \mathrm{Linear}(y_i^{(L)})$  这里类别里还包括一个特殊类：$\varnothing$表示 **no object**。

2. 边框回归头: $\hat{b}_i = \mathrm{MLP}(y_i^{(L)})$一般表示为：$\hat{b}_i = (c_x, c_y, w, h)$
   

