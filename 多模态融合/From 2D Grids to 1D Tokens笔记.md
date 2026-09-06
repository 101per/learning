下面这版可以直接作为你的论文阅读笔记。

# From 2D Grids to 1D Tokens：核心笔记

## 1. 论文研究目标

论文虽然面向的是**多模态图像融合（Multimodal Image Fusion）**，例如：

- 红外-可见光融合；
- 医学图像融合；

但它真正关注的重点并不是单纯设计一个新的 Fusion Module，而是重新思考：

> **图像中的不同类型信息，是否应该使用同一种二维特征表示？**

论文认为，传统方法把所有信息都放进：

F∈RH×W×CF\in\mathbb R^{H\times W\times C}

这样的二维 Feature Map 中并不合理。

------

## 2. 核心观点

图像中的信息可以大致分成两类：

### 全局信息 Base

包括：

- 整体亮度；
- 对比度；
- 色调；
- 全局视觉外观。

这类信息具有：

Global + Low-frequency + Weak spatial dependency\text{Global + Low-frequency + Weak spatial dependency}

即它描述的是整幅图像，而不是某个具体空间位置。

因此论文认为：

Base / Global Information→1D Tokens\boxed{ \text{Base / Global Information} \rightarrow \text{1D Tokens} }

------

### 局部细节 Detail

包括：

- 边缘；
- 纹理；
- 局部结构；
- 目标轮廓。

这类信息与空间位置高度相关，因此适合使用：

Detail / Local Information→2D Feature Map\boxed{ \text{Detail / Local Information} \rightarrow \text{2D Feature Map} }

------

## 3. 论文最核心的表示思想

传统图像融合：

Image→2D Feature Map→FusionImage \rightarrow 2D\ Feature\ Map \rightarrow Fusion

全局信息和局部信息都被混合在二维网格中。

论文提出：

1D Global+2D Local\boxed{ 1D\ Global + 2D\ Local }

即：

```text
                    Image
                      │
             ┌────────┴────────┐
             │                 │
         1D Tokens         2D Feature
             │                 │
          Global             Detail
             │                 │
      亮度 / 对比度       边缘 / 纹理
      全局视觉外观         局部空间结构
```

所以论文的核心思想不是：

> 把二维特征简单拉平成一维。

而是：

> **让不同性质的信息使用不同的表示空间。**

------

## 4. 1D Token 是什么？

论文使用预训练图像 Tokenizer **TiTok** 将图像压缩成少量 Token：

Z=τ(I)Z=\tau(I)

其中：

Z∈RN×dZ\in\mathbb R^{N\times d}

例如论文主要配置中：

N=32N=32

即整张图像只使用几十个 Token 表示。

它不是：

H×W×C→HW×CH\times W\times C \rightarrow HW\times C

这种简单 Flatten。

而是真正将整幅图像压缩成少量具有全局表达能力的 Token。

因此这些 Token 不再严格对应：

> “图像中的某一个 patch 在哪里”。

而更适合作为：

> **图像整体属性的紧凑表示。**

------

## 5. 为什么不能全部使用 1D Token？

因为图像融合不仅需要全局视觉信息，还必须保存：

- 细小纹理；
- 边缘；
- 物体结构；
- 空间位置。

如果只使用高度压缩的 1D Token：

Image→TokensImage\rightarrow Tokens

容易损失局部细节。

因此论文并没有完全抛弃二维表示，而是保留：

2D Spatial Features2D\ Spatial\ Features

负责 Detail。

所以论文并不是：

2D→1D2D\rightarrow1D

而是：

2D Only→1D Global+2D Local\boxed{ 2D\ Only \rightarrow 1D\ Global + 2D\ Local }

------

## 6. Token-to-Map

由于 1D Token 最终还要参与图像生成，因此论文设计了：

Token−to−Map\boxed{ Token-to-Map }

将：

Z∈RN×dZ\in\mathbb R^{N\times d}

重新映射为：

F∈RH×W×CF\in\mathbb R^{H\times W\times C}

大致流程：

```text
1D Tokens
    ↓
Linear Mapping
    ↓
低分辨率 2D Feature
    ↓
逐级 Upsampling
    ↓
加入原图多尺度细节
    ↓
完整 2D Feature Map
```

需要注意：

> Token-to-Map 并不意味着把思想重新退回传统二维表示。

1D Token 仍然承担**全局信息的压缩和调控**，而二维 Feature Map 主要用于恢复空间结构并完成最终生成。

------

## 7. Base / Detail 分解

获得特征之后，网络进一步将信息显式分成：

B(m)B^{(m)}

和：

D(m)D^{(m)}

其中：

B(m)=Base FeatureB^{(m)} = \text{Base Feature}

主要负责：

- 全局亮度；
- 对比度；
- 整体外观。

而：

D(m)=Detail FeatureD^{(m)} = \text{Detail Feature}

主要负责：

- 边缘；
- 纹理；
- 局部结构。

之后分别进行融合：

Bf=FB(B1,B2)B^f = \mathcal F_B (B^1,B^2)Df=FD(D1,D2)D^f = \mathcal F_D (D^1,D^2)

最后：

[Bf,Df]→Decoder→If[B^f,D^f] \rightarrow Decoder \rightarrow I^f

------

## 8. Selective Token Editing

论文进一步发现：

> 不是所有 1D Token 对图像全局外观的贡献都一样。

有些 Token 对：

- 亮度；
- 对比度；
- 清晰度；
- 全局视觉质量；

影响更明显。

因此提出：

Selective Token Editing (STE)\boxed{ Selective\ Token\ Editing\ (STE) }

核心思想是：

> 不修改全部 Token，只编辑少数关键 Token 和 Channel。

形式上可以表示为：

Z~=Z+M⊙Δ\tilde Z = Z+M\odot\Delta

其中：

- ZZ：原始 Token；
- MM：选择 Mask；
- Δ\Delta：可学习修改量。

如果：

Mij=0M_{ij}=0

则该 Token-channel 不修改。

如果：

Mij=1M_{ij}=1

则对其进行学习和调节。

因此 STE 本质上是在做：

Sparse Global Representation Editing\boxed{ \text{Sparse Global Representation Editing} }

------

## 9. 论文真正的创新点

### 创新 1：改变共享表示形式

传统：

Global + Local→2D Grid\text{Global + Local} \rightarrow 2D\ Grid

论文：

Global→1D,Local→2D\boxed{ Global\rightarrow1D,\qquad Local\rightarrow2D }

这是最核心的创新。

------

### 创新 2：使用紧凑 1D Token 表示全局信息

不再让全局亮度、对比度等信息重复存储在二维网格的每个位置中，而是集中编码到少量 Token 中。

这样能够减少：

Global information\text{Global information}

与：

Local detail\text{Local detail}

之间的表示纠缠。

------

### 创新 3：Selective Token Editing

通过寻找关键 Token，只对少量位置进行调整，就能控制融合图像整体外观。

即：

Dense Editing→Sparse Editing\text{Dense Editing} \rightarrow \text{Sparse Editing}

------

### 创新 4：1D 与 2D 混合表示

论文并不是要证明：

> 1D 一定比 2D 好。

而是认为：

不同信息应该使用适合自己的表示方式\boxed{ \text{不同信息应该使用适合自己的表示方式} }

即：

1D Token1D\ Token

擅长全局信息，

2D Grid2D\ Grid

擅长局部空间信息。

------

# 10. 最容易理解错的地方

### 错误理解 1

> 论文把所有 2D Feature 都改成了 1D Token。

错误。

正确的是：

1D Global+2D Local\boxed{ 1D\ Global + 2D\ Local }

二维 Feature Map 仍然非常重要。

------

### 错误理解 2

> 论文只是把 Feature Map Flatten 成 Transformer Token。

错误。

论文使用的是图像 Tokenizer，将整张图高度压缩为少量：

1D Tokens1D\ Tokens

不是普通 Patch Flatten。

------

### 错误理解 3

> 论文重点是提出一种新的多模态融合算法。

不完全准确。

更准确地说：

> **多模态融合是它的应用场景，而“共享表示应该怎样设计”才是论文的核心研究问题。**

------

# 11. 一句话总结

这篇论文最核心的思想可以记成：

全局信息用 1D Token 表示，局部细节用 2D Grid 表示，再进行融合。\boxed{ \text{全局信息用 1D Token 表示，局部细节用 2D Grid 表示，再进行融合。} }

或者更简洁一点：

> **这篇论文的重点不是单纯研究“怎么融合”，而是先解决“用什么表示再融合”。**

------

# 12. 论文逻辑主线

最终可以记住这一条：

```text
传统方法
Global + Detail
      ↓
统一塞入 2D Feature Map
      ↓
两类信息相互纠缠


本文
                Image
                  ↓
         ┌────────┴────────┐
         ↓                 ↓
     1D Tokens          2D Grid
         ↓                 ↓
      Global             Detail
         ↓                 ↓
   Token Editing      Detail Preserve
         └────────┬────────┘
                  ↓
               Fusion
                  ↓
             Fused Image
```

**最核心关键词：**

Representation Reform\boxed{ Representation\ Reform }1D Global+2D Local\boxed{ 1D\ Global + 2D\ Local }Selective Token Editing\boxed{ Selective\ Token\ Editing }Base/Detail Decoupling\boxed{ Base / Detail\ Decoupling }