# 世界模型（World Model）主流技术路线分类

## 1. World Model 是什么

World Model，即世界模型，可以理解为模型在内部学习一个关于外部环境的预测模型，使智能体能够根据当前状态以及可能执行的动作，预测环境未来会如何变化。

一个最典型的 World Model 可以写为：

$$
s_{t+1}=f(s_t,a_t)
$$

其中：

- $$s_t$$：当前世界状态；
- $$a_t$$：智能体执行的动作；
- $$s_{t+1}$$：执行动作后未来世界的状态；
- $$f$$：模型学习到的环境动力学规律。

因此，World Model 最核心的能力不是简单地识别“现在有什么”，而是：

> **理解当前世界，并预测世界接下来会怎样变化。**

在强化学习、机器人、自动驾驶和具身智能中，World Model 进一步可以在模型内部进行多步预测：

$$
s_t\rightarrow s_{t+1}\rightarrow s_{t+2}\rightarrow \cdots
$$

从而让智能体在真正执行动作之前，先在内部模拟多个未来，再进行规划。

截至 2026 年，World Model 已经从传统 Model-Based Reinforcement Learning 中的动力学模型扩展到生成式视频模型、机器人模型、JEPA、自回归模型等多个方向。近期综述也指出，目前所谓的 World Model 可以指 recurrent latent dynamics、diffusion video generator、JEPA-style predictor 等不同形式。

# 2. World Model 的四条主流技术路线

如果按照“**模型如何表示和预测未来世界**”进行分类，目前可以重点理解为以下四条路线：

1. **Latent Dynamics World Model**
2. **Autoregressive World Model**
3. **Diffusion / Flow World Model**
4. **JEPA / Joint-Embedding Predictive World Model**

它们之间最大的区别可以概括为：

| 路线             | 主要预测对象        | 核心思想                             |
| ---------------- | ------------------- | ------------------------------------ |
| Latent Dynamics  | 潜在状态            | 在低维 latent space 中模拟世界动力学 |
| Autoregressive   | 离散 Token          | 像语言模型一样逐 Token 预测未来      |
| Diffusion / Flow | 图像、视频或 latent | 通过生成模型生成未来可能状态         |
| JEPA             | 高层语义 Embedding  | 不重建像素，只预测未来抽象表征       |

需要注意，这四种方式**不是严格互斥的网络架构**。例如一个系统可能同时使用 latent representation 和 autoregressive Transformer，也可能在 latent space 内进行 diffusion。更准确地说，它们代表目前 World Model 中几种主要的**预测范式**。2026 年机器人 World Model 综述同样指出，recurrent latent dynamics、autoregressive token modeling、diffusion/flow generation 和 masked latent prediction 都属于当前主要实现机制。

# 3. Latent Dynamics World Model

## 3.1 基本思想

Latent Dynamics 是 World Model 中最经典、最成熟的一条路线。

由于真实图像维度非常高，直接预测：

$$
I_t\rightarrow I_{t+1}
$$

计算成本非常高。

因此这类方法首先利用 Encoder 将真实观测：

$$
o_t
$$

编码成一个低维潜在状态：

$$
z_t=E(o_t)
$$

然后学习：

$$
z_{t+1}=f(z_t,a_t)
$$

即：

> 不直接模拟真实世界的每一个像素，而是在一个压缩后的潜在空间中模拟世界。

基本结构为：

```
真实环境
   ↓
Observation o_t
   ↓
Encoder
   ↓
Latent State z_t
   ↓
 + Action a_t
   ↓
Dynamics Model
   ↓
Latent State z_{t+1}
```

部分模型还会加入 Decoder：

```
z_{t+1}
   ↓
Decoder
   ↓
预测 Observation
```

## 3.2 代表模型

这一路线的重要代表包括：

- World Models
- PlaNet
- RSSM
- Dreamer
- DreamerV2
- DreamerV3

其中 Dreamer 系列是 Model-Based Reinforcement Learning 中非常具有代表性的 World Model。

Dreamer 的核心思想就是：

> 在 latent space 中学习环境动力学，然后直接在模型内部进行 imagined rollout。

例如：

$$
z_t
\overset{a_t}{\rightarrow}
z_{t+1}
\overset{a_{t+1}}{\rightarrow}
z_{t+2}
$$

智能体甚至不需要真的执行这些动作，就可以先在世界模型内部“想象”这些动作的结果。

## 3.3 优势

Latent Dynamics 的主要优势是：

- 相比直接预测像素，计算成本低；
- 非常适合强化学习和控制；
- 可以进行多步 rollout；
- 可以直接用于 planning；
- 已经形成比较成熟的理论体系。

## 3.4 局限

它的一个重要问题是：

> **latent representation 到底应该保留哪些信息？**

如果模型同时要求：

$$
z_t\rightarrow \hat o_t
$$

即重建原始 observation，那么 latent space 往往必须保留大量：

- 纹理；
- 背景；
- 光照；
- 颜色；
- 细微像素信息。

而这些信息对于机器人真正理解：

> “杯子在哪里？”

可能并不重要。

这也是 JEPA 思想出现的重要背景之一。

# 4. Autoregressive World Model

## 4.1 基本思想

Autoregressive World Model 的核心思想来自语言模型。

LLM 做的是：

$$
x_1,x_2,\cdots,x_t
\rightarrow
x_{t+1}
$$

也就是：

> 根据之前的 Token 预测下一个 Token。

于是研究者自然想到：

> 能不能把整个世界也 Token 化？

例如：

```
Image / Video
      ↓
Tokenizer
      ↓
z1 z2 z3 z4 ... zn
```

然后使用 Transformer：

$$
p(z_{t+1}|z_1,\cdots,z_t)
$$

预测未来 Token。

如果再加入 action：

$$
p(z_{t+1}|z_{\leq t},a_{\leq t})
$$

模型就可以预测：

> 执行某个动作以后，世界接下来会出现什么 Token。

## 4.2 一个直观例子

例如机器人当前看到：

```
桌子
+
杯子在左边
```

执行：

```
向右推
```

可以将：

```
视觉 Token
+
Action Token
```

一起输入 Transformer：

```
[视觉Token]
[动作Token]
[视觉Token]
[动作Token]
...
```

然后预测下一时刻的视觉 Token。

## 4.3 代表方向

这一方向常见于：

- Genie 系列；
- 游戏 World Model；
- Transformer-based World Model；
- Video Token World Model；
- Vision-Language-Action Model 中的部分世界预测模块。

2024—2026 年的 World Model 工作中，autoregressive prediction 已经成为非常重要的一种技术范式，尤其适合把视频、动作、语言统一表示成序列。

## 4.4 优势

最大的优势是：

> 可以直接复用 LLM 已经非常成熟的 Transformer + Token Prediction 范式。

因此：

```
Language Token
+
Visual Token
+
Action Token
```

理论上都可以统一起来。

最终可以形成：

$$
P(\text{future}|\text{history},\text{action})
$$

这种统一预测框架。

## 4.5 局限

主要问题包括：

- 视频 Token 数量非常巨大；
- 长时间 rollout 成本高；
- autoregressive error 会不断累积；
- 离散 Token 可能损失连续物理信息。

# 5. Diffusion / Flow World Model

## 5.1 基本思想

Diffusion World Model 的想法非常直接：

> 既然 Diffusion Model 能够生成图片和视频，那么就让它直接生成世界未来的状态。

例如：

$$
I_t+a_t
\rightarrow
I_{t+1}
$$

也就是输入：

- 当前画面；
- 动作；

然后生成：

- 下一时刻画面。

进一步还可以：

$$
I_t+a_{t+k}
\rightarrow
I_{t+1+k}
$$

直接生成未来一段视频。

## 5.2 基本结构

```
当前 Observation
       +
     Action
       ↓
Diffusion / Flow Model
       ↓
Future Observation
       ↓
Future Video
```

例如：

> 一个机器人看到桌上的杯子，然后执行“推杯子”。

Diffusion World Model 可以直接生成：

```
Frame t
杯子在左边

↓

Frame t+1
杯子开始移动

↓

Frame t+2
杯子到达右边
```

因此这种 World Model 非常直观：

> **World Model 本身就像一个可交互的视频生成器。**

## 5.3 当前发展

随着视频生成模型快速发展，Diffusion 和 Flow Matching 已经成为 World Model 中非常活跃的一条路线。

特别是在：

- 自动驾驶；
- 机器人；
- 游戏环境；
- Physical AI；
- Video World Model

领域越来越常见。

例如 Cosmos 等工作就代表了大型生成式 Physical AI / World Model 的发展趋势。

当前综述也把 diffusion 和 flow-matching generation 作为机器人 World Model 的主要实现机制之一。

## 5.4 优势

优势是：

- 可以生成高质量未来场景；
- 能表示多模态未来；
- 适合视频世界建模；
- 可以直接作为环境模拟器。

例如：

一个动作可能存在多个未来：

```
         → 杯子倒下
推杯子
         → 杯子滑动
```

Diffusion 天然适合描述：

$$
p(s_{t+1}|s_t,a_t)
$$

这种概率分布，而不是只预测唯一答案。

## 5.5 局限

最大问题之一是：

> **生成所有视觉细节非常昂贵。**

模型可能花费大量算力预测：

- 阴影；
- 光照；
- 背景；
- 纹理；
- 水面波纹；
- 无关物体细节。

但这些内容可能对机器人规划没有任何帮助。

这也是 JEPA 路线和生成式 World Model 一个非常关键的分歧。

# 6. JEPA / Joint-Embedding Predictive World Model

## 6.1 核心思想

JEPA 的全称是：

**Joint-Embedding Predictive Architecture**

它提出一个非常重要的问题：

> World Model 真的有必要预测未来所有像素吗？

JEPA 给出的答案是：

> **没有必要。**

它认为 World Model 应该预测：

$$
\text{未来世界的抽象语义表示}
$$

而不是：

$$
\text{未来世界的所有像素}
$$

因此 JEPA 的基本形式是：

$$
x\rightarrow E(x)=z
$$

然后：

$$
z_{\text{context}}
\rightarrow
\hat z_{\text{target}}
$$

目标是：

$$
\hat z_{\text{target}}
\approx
z_{\text{target}}
$$

# 7. I-JEPA：JEPA 最基础的形式

I-JEPA 最初并不是一个完整的机器人 World Model，而主要是一种自监督表征学习方法。

例如一张图片：

```
狗头 | 狗身体 | 狗腿
```

遮住：

```
狗腿
```

Context Encoder 输入：

```
狗头 + 狗身体
```

得到：

$$
z_c
$$

Predictor：

$$
P(z_c)
$$

预测：

$$
\hat z_t
$$

Target Encoder 则计算真实狗腿区域：

$$
z_t
$$

最后优化：

$$
\hat z_t\approx z_t
$$

重点在于：

**模型不需要把狗腿的像素生成出来。**

只需要预测：

> “这里应该存在一个具有‘狗腿’语义的信息。”

# 8. V-JEPA：从图像扩展到视频

当 JEPA 从 Image 扩展到 Video 后，就变得越来越像 World Model。

模型可以根据视频前面的内容：

$$
z_t
$$

预测未来：

$$
z_{t+1}
$$

于是开始具备：

- motion understanding；
- temporal prediction；
- physical understanding；

等能力。

# 9. V-JEPA 2：进一步向真正 World Model 演化

V-JEPA 2 进一步将 JEPA 用于物理世界预测。

Meta 在 2025 年发布的 V-JEPA 2 首先使用超过一百万小时的视频进行 action-free 自监督预训练，然后进一步通过机器人轨迹进行 action-conditioned training。

因此模型从：

$$
z_t
\rightarrow
z_{t+1}
$$

进一步发展到：

$$
z_t+a_t
\rightarrow
z_{t+1}
$$

即：

> 当前世界处于状态 $$z_t$$，如果机器人采取动作 $$a_t$$，那么世界接下来应该变成什么状态？

这已经是标准的 World Model 问题。

V-JEPA 2-AC 使用不到 62 小时的机器人视频进行 action-conditioned post-training，并用于真实机器人 zero-shot planning。

# 10. JEPA 和传统 Latent Dynamics 的关系

这是最容易混淆的地方。

因为两者最终都可能写成：

$$
z_t+a_t\rightarrow z_{t+1}
$$

所以不能简单理解为：

```
Latent Dynamics
vs
JEPA
```

它们实际上存在大量重叠。

更准确的理解是：

```
                 Latent-space World Model

                        │
             ┌──────────┴──────────┐
             │                     │
      传统 Latent Dynamics       JEPA-style
             │                     │
        Dreamer / RSSM          V-JEPA
             │                     │
    学习状态动力学             学习预测性表征
             │                     │
 经常结合 reconstruction       不要求重建像素
```

传统 Latent Dynamics 更关注：

> **如何在潜空间模拟环境动力学。**

JEPA 更关注：

> **World Model 应该预测什么样的抽象 representation。**

因此二者解决的问题侧重点不同。

# 11. 四种 World Model 的核心差异

假设机器人看到：

> 桌子上有一个杯子。

机器人执行：

> 向右推。

四种 World Model 会采取完全不同的方式。

## Latent Dynamics

首先：

$$
I_t\rightarrow z_t
$$

然后：

$$
z_t+a_t\rightarrow z_{t+1}
$$

得到一个未来 latent state：

> 杯子已经移动到右边。

## Autoregressive World Model

首先将世界变成：

```
Visual Tokens
```

然后预测：

$$
Token_1\rightarrow Token_2\rightarrow Token_3...
$$

最终得到未来世界的 Token Sequence。

## Diffusion World Model

直接生成：

```
当前画面
+
向右推动作
       ↓
未来视频
```

得到：

> 杯子逐渐向右移动的视频。

## JEPA

模型：

$$
z_t+a_t
\rightarrow
\hat z_{t+1}
$$

只预测：

> “杯子已经移动到右侧”对应的抽象 representation。

至于：

- 杯子上的反光；
- 桌子的木纹；
- 阴影变化；

不一定需要精确预测。

# 12. 四条路线的整体比较

| 特性             | Latent Dynamics | Autoregressive | Diffusion / Flow | JEPA       |
| ---------------- | --------------- | -------------- | ---------------- | ---------- |
| 预测空间         | Latent State    | Token          | Pixel / Latent   | Embedding  |
| 代表模型         | Dreamer         | Genie 类       | Cosmos 等        | V-JEPA     |
| 是否生成画面     | 可选            | 可以           | 通常可以         | 通常不需要 |
| 是否适合 Rollout | 强              | 强             | 可以             | 可以       |
| 是否适合 RL      | 非常适合        | 可以           | 可以             | 正在发展   |
| 是否适合机器人   | 强              | 强             | 强               | 强         |
| 训练计算量       | 相对低          | 较高           | 很高             | 相对较低   |
| 是否关注像素细节 | 较少/中等       | 中等           | 很高             | 很低       |
| 语义抽象能力     | 强              | 中等～强       | 中等             | 强         |
| 当前成熟度       | 很成熟          | 很高           | 很高             | 快速发展   |

# 13. 一个更准确的 World Model 技术谱系

因此最好不要简单理解成：

```
World Model
├── Dreamer
├── Diffusion
├── JEPA
└── Transformer
```

因为这里混合了：

- representation；
- architecture；
- objective；
- training method。

更加合理的理解应该分成几个维度。

## 第一维：预测什么？

### Observation Space

$$
I_t\rightarrow I_{t+1}
$$

预测：

- Pixel；
- Video。

典型：

- Video World Model；
- Diffusion World Model。

### Token Space

$$
z_1,z_2,\cdots,z_t
\rightarrow
z_{t+1}
$$

典型：

- Autoregressive World Model。

### Latent / Embedding Space

$$
z_t\rightarrow z_{t+1}
$$

典型：

- Dreamer；
- JEPA。

# 14. 第二维：怎么预测？

可以使用：

```
RNN / RSSM
Transformer
Autoregressive
Diffusion
Flow Matching
JEPA Predictor
```

所以：

> **Latent 并不等于某一种网络。**

完全可以：

```
Latent + Transformer
```

也可以：

```
Latent + Diffusion
```

甚至：

```
Latent + Autoregressive
```

# 15. 第三维：是否 Action-Conditioned

这是判断一个模型是否真正适合作为机器人 World Model 的重要标准。

仅仅：

$$
z_t\rightarrow z_{t+1}
$$

属于：

> Passive World Prediction

即：

> 世界下一步会发生什么？

而：

$$
z_t+a_t\rightarrow z_{t+1}
$$

属于：

> Action-Conditioned World Model

回答的是：

> 如果**我做这个动作**，世界会发生什么？

机器人规划真正需要的通常是第二种。

因此严格从具身智能/机器人角度看：

$$
P(s_{t+1}|s_t,a_t)
$$

是 World Model 最重要的形式之一。

2026 年机器人 World Model 综述甚至直接采用“action-conditioned predictive system”作为操作性定义，用来区分 World Model 与普通 perception model、policy、reward/value model。

# 16. 当前 World Model 的整体发展趋势

目前 World Model 正在发生一个非常明显的变化：

早期 World Model：

```
Environment
    ↓
Small Encoder
    ↓
Latent State
    ↓
RSSM
    ↓
RL Agent
```

典型：

```
PlaNet
↓
Dreamer
↓
DreamerV2
↓
DreamerV3
```

主要服务于：

> Reinforcement Learning

现在的大型 World Model：

```
大量 Image / Video / Robot Data
              ↓
       Foundation Model
              ↓
     World Representation
              ↓
    Action-conditioned Model
              ↓
     Prediction / Planning
              ↓
           Robot
```

World Model 正逐渐从：

> **RL 中的一个动力学模块**

演变为：

> **智能体理解、预测和规划物理世界的 Foundation Model。**

因此 2026 年的综述将 World Model 描述为一个已经同时涵盖生成模型、自监督预测、机器人控制、自动驾驶和 embodied AI 的快速扩展领域。

# 17. 最终知识框架

如果现在开始系统学习 World Model，可以在脑中建立下面这个框架：

```
                         World Model
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
       预测什么             怎么预测            如何交互
          │                   │                   │
   ┌──────┼──────┐     ┌──────┼──────┐      ┌─────┴─────┐
   │      │      │     │      │      │      │           │
 Pixel  Token  Latent   AR  Diffusion JEPA   无动作     Action
   │      │      │                         Prediction Conditioned
   │      │      │
 Video   Genie Dreamer
 Model         V-JEPA
```

如果从“主要研究路线”角度再简化，则可以记成：

```
World Model
│
├── 1. Latent Dynamics
│      └── Dreamer / RSSM
│
├── 2. Autoregressive World Model
│      └── Transformer / Token Prediction
│
├── 3. Diffusion / Flow World Model
│      └── Video / Physical World Generation
│
└── 4. JEPA-style World Model
       └── I-JEPA
           ↓
         V-JEPA
           ↓
        V-JEPA 2
           ↓
       V-JEPA 2-AC
```

# 18. 一句话理解四条路线

最终可以用四句话把主要路线区分开：

**Latent Dynamics：**

> 把世界压缩成状态，然后学习这个状态如何随动作变化。

**Autoregressive World Model：**

> 把世界变成 Token，然后像 LLM 一样一个一个预测未来 Token。

**Diffusion / Flow World Model：**

> 把未来世界当作一个生成问题，直接生成未来图像或视频。

**JEPA：**

> 不生成未来世界的所有像素，而只预测未来世界最重要的抽象语义表示。

因此，从技术思想上可以进一步概括为：

```
Latent Dynamics：
“世界怎么运动？”

Autoregressive：
“下一个世界 Token 是什么？”

Diffusion：
“未来世界长什么样？”

JEPA：
“未来世界在语义上会变成什么？”
```

这四个问题基本构成了当前 World Model 最重要的技术路线。