# INTACT 技术文档

## Isomorphic Intent-to-Action Learning for Search-Free World Models

------

# 1. 背景：World Model 的 Forward-Action Gap

## 1.1 World Model 基本思想

世界模型（World Model）的目标是让智能体学习环境的动态规律，使其能够在内部预测未来状态，而不需要不断与真实环境交互。

对于视觉控制任务，通常首先利用视觉编码器将环境观测映射到 latent 空间：

\[ z_t=E_\theta(o_t) \]

其中：

- $o_t$ 表示时间 $t$ 的视觉观测；
- $E_\theta$ 表示视觉编码器；
- $z_t$ 表示对应的 latent 状态表示。

传统世界模型主要学习：

\[ (z_t,a_t)\rightarrow z_{t+1} \]

即：

> 已知当前状态和执行动作，预测动作之后的未来状态。

例如机器人：

当前状态：

\[ z_t \]

执行动作：

\[ a_t=\text{机械臂向右移动} \]

世界模型预测：

\[ \hat z_{t+1} \]

表示：

执行该动作之后机器人可能达到的位置。

------

## 1.2 传统 World Model 的问题

虽然 World Model 可以预测未来，但是它并不能直接回答：

> 如果我想达到某个目标状态，需要执行什么动作？

即：

\[ z_g\rightarrow a \]

其中：

- $z_g$ 是目标状态；
- $a$ 是达到目标所需动作。

因此存在：

## Forward-Action Gap

即：

传统 World Model 擅长：

\[ Action\rightarrow Future \]

但是缺少：

\[ Goal\rightarrow Action \]

------

## 1.3 传统解决方式：搜索式规划

为了利用 World Model 完成控制，传统方法通常采用搜索。

例如：

给定当前状态：

\[ z_t \]

随机生成多个动作序列：

\[ A=(a_t,a_{t+1},...,a_{t+H}) \]

然后利用 World Model 预测未来：

\[ (z_t,A)\rightarrow z_{future} \]

计算哪个未来状态距离目标最近，再选择对应动作。

流程：

```
当前状态 z_t
      ↓
生成大量候选动作
      ↓
World Model预测未来状态
      ↓
计算未来状态与目标距离
      ↓
选择最优动作
      ↓
执行
```

这种方法存在：

1. 推理速度慢；
2. 需要大量动作采样；
3. World Model 只负责预测，不直接参与动作生成。

因此 INTACT 希望解决：

> 能否让 World Model 直接根据目标变化生成动作，而不需要搜索？

------

# 2. INTACT 核心思想

INTACT 的核心思想是：

将控制问题转化为：

\[ Intent\rightarrow Action \]

即：

学习：

> 如果希望环境产生某种变化，那么应该执行什么动作。

------

传统 World Model：

\[ Action\rightarrow State\ Change \]

INTACT：

\[ State\ Change\ Intent\rightarrow Action \]

其中 Intent 表示：

> 当前状态到目标状态之间希望发生的变化。

------

整体结构：

```
当前状态 z_t
        ↓
   Intent表示
        ↓
Intent-to-Action Predictor
        ↓
      动作 a
        ↓
 Forward World Model
        ↓
未来状态预测
```

INTACT 包含两个核心模型：

1. Forward Predictor  
   - 学习动作如何改变世界；
2. INTACT Predictor  
   - 学习变化意图如何产生动作。

------

# 3. Latent空间中的Intent表示

## 3.1 为什么需要 Intent？

直接使用：

\[ z_g-z_t \]

作为控制信号存在问题。

因为 latent 差异通常包含大量混合信息。

例如机器人任务：

目标：

“拿起杯子”。

latent变化可能包含：

- 杯子位置变化；
- 机器人姿态变化；
- 手臂运动；
- 环境变化；
- 视觉背景变化。

这些信息全部混合在：

\[ z_g-z_t \]

中。

因此 INTACT 提出：

将状态变化显式表示为：

\[ Intent \]

让 latent transition 与动作之间建立更加明确的联系。

------

# 4. Local Intent 与 Goal Intent

INTACT 定义两类 Intent：

1. Local Intent
2. Goal Intent

二者共享同一个 Intent 空间。

------

# 4.1 Local Intent

Local Intent 来源于真实轨迹。

定义：

\[ m_t^{local}=z_{t+1}-z_t \]

其中：

- $z_t$：当前状态；
- $z_{t+1}$：真实下一状态。

它表示：

> 当前动作实际造成的状态变化。

例如：

机器人：

```
当前：
机器人位于桌子左侧

执行动作：
向右移动

结果：
机器人移动到桌子中间
```

那么：

\[ m_t^{local}=z_{t+1}-z_t \]

表示：

这次动作产生的变化。

------

Local Intent 可以理解为：

\[ Action\rightarrow State\ Change \]

即：

动作导致的变化。

------

# 4.2 Goal Intent

Goal Intent 定义为：

\[ m_t^{goal}=sg(z_g)-z_t \]

其中：

\[ sg(\cdot) \]

表示 stop-gradient 操作。

即：

前向计算：

\[ sg(z_g)=z_g \]

但是反向传播：

\[ \frac{\partial sg(z_g)}{\partial z_g}=0 \]

表示：

目标状态作为固定参考，不参与梯度更新。

------

Goal Intent 表示：

> 当前状态距离目标状态还需要发生什么变化。

例如：

当前：

```
机器人在桌子左侧
```

目标：

```
机器人抓住杯子
```

那么：

\[ m_t^{goal}=z_g-z_t \]

表示：

达到目标需要产生的变化方向。

------

# 5. 为什么 Local Intent 和 Goal Intent 可以共享？

这是 INTACT 名称中：

**Isomorphic**

的含义。

很多人容易误解：

认为：

\[ z_{t+1}=z_g \]

或者：

两者时间过程完全相同。

实际上不是。

Local Intent：

\[ z_t\rightarrow z_{t+1} \]

Goal Intent：

\[ z_t\rightarrow z_g \]

二者可能具有完全不同的时间跨度。

Local：

一步变化。

Goal：

多步变化。

------

INTACT 关注的不是时间长度，而是：

状态变化关系。

二者都满足：

\[ State+Intent\rightarrow Target \]

即：

```
起始状态
    ↓
状态变化Intent
    ↓
目标状态
```

因此作者认为：

二者属于同一个状态变化空间。

------

换句话说：

Local Intent：

表示：

> 已经发生的变化。

Goal Intent：

表示：

> 希望发生的变化。

INTACT 假设：

如果两个变化属于同一种 latent transition 结构，那么它们应该共享同一个：

\[ Intent\rightarrow Action \]

映射。

------

# 6. Action-Aligned Intent Representation

## 6.1 为什么需要 Action-Aligned Intent？

传统 latent world model 中，latent 表示主要用于预测未来：

\[ (z_t,a_t)\rightarrow z_{t+1} \]

latent 的目标是：

> 保留足够的信息，使模型能够预测未来状态。

但是对于控制任务，仅仅能够预测未来是不够的。

智能体需要知道：

> 哪些状态变化与动作存在对应关系。

例如：

机器人执行动作：

\[ a_t=\text{向右移动} \]

产生：

\[ z_{t+1}-z_t \]

这个变化应该和：

“向右移动”

建立对应关系。

因此，INTACT 希望学习一种：

**Action-Aligned Intent Representation**

即：

> 与动作空间对齐的状态变化表示。

------

## 6.2 从 State Representation 到 Intent Representation

传统 World Model：

学习：

\[ z_t \]

表示：

“世界现在是什么状态”。

INTACT进一步关注：

\[ \Delta z \]

表示：

“世界正在如何变化”。

因此：

传统：

\[ State\ Representation \]

变为：

\[ Transition\ Intent\ Representation \]

即：

从：

> 世界是什么样

转变为：

> 世界需要如何变化。

------

# 7. Four-slot Intent Representation

## 7.1 为什么需要 Four-slot？

如果直接使用：

\[ m=z_g-z_t \]

作为 Intent：

存在一个问题：

latent change 是一个高度混合的向量。

例如：

机器人任务：

目标：

“抓取杯子”。

状态变化可能包含：

\[ \Delta z= [ 位置变化, 姿态变化, 物体关系变化, 动作变化 ] \]

但是这些因素全部混合。

模型需要自己学习：

哪些变化对应哪些动作。

------

因此 INTACT 对 Intent 进行结构化表示。

核心思想：

> 将一个混合的 latent transition 分解为多个具有不同作用的子空间。

即：

\[ m= [m_1,m_2,m_3,m_4] \]

其中：

每个 slot 负责描述 Intent 的不同方面。

------

## 7.2 Four-slot 的作用

Four-slot 并不是简单增加四个网络模块，而是一种：

**结构化 latent 表示方式。**

普通 latent：

```
一个整体向量

[xxxxxxxxxxxxxxxx]
```

模型不知道：

每个维度对应什么。

Four-slot：

```
Intent

[slot1]
[slot2]
[slot3]
[slot4]
```

使变化信息具有结构。

------

## 7.3 Four-slot 与 Isomorphic Intent 的关系

INTACT 的目标：

让：

Local Intent

和：

Goal Intent

进入同一个结构化空间。

即：

\[ m^{local} \rightarrow Action \]

以及：

\[ m^{goal} \rightarrow Action \]

共享同一个映射。

因此：

训练阶段：

利用 Local Intent：

\[ z_{t+1}-z_t \]

学习动作规律。

推理阶段：

使用 Goal Intent：

\[ z_g-z_t \]

生成目标动作。

------

# 8. INTACT 整体模型结构

INTACT 包含三个主要部分：

1. Visual Encoder
2. INTACT Predictor
3. Forward Predictor

整体流程：

```
观察 o_t
    |
    ↓
Encoder
    |
    ↓
latent z_t


目标 o_g
    |
    ↓
Encoder
    |
    ↓
latent z_g


z_g-z_t
    |
    ↓
Intent Representation
    |
    ↓
INTACT Predictor
    |
    ↓
动作 a
    |
    ↓
Forward Predictor
    |
    ↓
预测未来状态
```

------

# 9. Visual Encoder

视觉编码器负责将环境观测转换到 latent 空间。

公式：

\[ z_t=E_\theta(o_t) \]

其中：

- $o_t$：视觉输入；
- $E_\theta$：视觉编码器；
- $z_t$：latent 状态。

同理：

\[ z_{t+1}=E_\theta(o_{t+1}) \]\[ z_g=E_\theta(o_g) \]

得到：

当前状态：

\[ z_t \]

下一状态：

\[ z_{t+1} \]

目标状态：

\[ z_g \]

------

# 10. INTACT Predictor

## 10.1 功能

INTACT Predictor 是论文的核心模块。

目标：

学习：

\[ Intent\rightarrow Action \]

即：

给定：

状态变化意图。

输出：

对应动作。

------

数学形式：

\[ \hat a=G_\eta(m) \]

其中：

- $m$：Intent；
- $G_\eta$：INTACT Predictor；
- $\hat a$：预测动作。

------

# 10.2 Local Intent训练

训练阶段：

首先计算：

\[ m^{local}=z_{t+1}-z_t \]

输入：

\[ m^{local} \]

预测：

\[ \hat a_t=G_\eta(m^{local}) \]

由于真实动作：

\[ a_t \]

已知。

因此可以计算：

\[ L_{action} = ||\hat a_t-a_t||^2 \]

作用：

让模型学习：

\[ \boxed{ State\ Change\rightarrow Action } \]

------

# 10.3 为什么不用 Goal Intent 直接监督？

因为：

Goal Intent：

\[ z_g-z_t \]

通常没有唯一动作标签。

例如：

目标：

机器人移动到门口。

可以：

方案1：

向前走。

方案2：

绕路。

方案3：

避开障碍。

因此不存在唯一：

\[ a_g \]

所以：

INTACT 使用：

Local Intent 提供监督。

------

# 11. Forward Predictor

## 11.1 功能

Forward Predictor 是一个 latent world model。

作用：

学习：

\[ (z_t,a_t)\rightarrow z_{t+1} \]

即：

动作如何改变世界。

------

输入：

当前状态：

\[ z_t \]

动作：

\[ a_t \]

输出：

预测未来：

\[ \hat z_{t+1} \]

公式：

\[ \hat z_{t+1} = F_\phi(z_t,a_t) \]

------

# 11.2 为什么需要 Forward Predictor？

因为 INTACT Predictor 只负责：

\[ Intent\rightarrow Action \]

但是它不知道：

这个动作执行以后世界会怎样。

因此需要 Forward Predictor 判断：

动作是否能够产生目标变化。

例如：

Goal Intent：

\[ z_g-z_t \]

↓

INTACT Predictor：

\[ \hat a \]

↓

Forward Predictor：

\[ (z_t,\hat a) \rightarrow \hat z_{t+1} \]

然后判断：

预测状态是否朝目标靠近。

------

# 12. World Model Loss

Forward Predictor 的训练目标：

让预测未来接近真实未来。

公式：

\[ L_{world} = E[ ||\hat z_{t+1}-z_{t+1}||_2^2/d ] + \lambda_{sig}R_{SIG} \]

包含两部分：

------

## 12.1 Future Prediction Loss

第一项：

\[ E[ ||\hat z_{t+1}-z_{t+1}||_2^2/d ] \]

表示：

预测未来 latent 与真实未来 latent 的距离。

作用：

训练：

\[ (z_t,a_t)\rightarrow z_{t+1} \]

------

## 12.2 SIGReg

第二项：

\[ \lambda_{sig}R_{SIG} \]

其中：

- $\lambda_{sig}$：权重；
- $R_{SIG}$：Similarity Invariance Regularization。

作用：

防止 latent 表示退化。

------

# 13. SIGReg 的作用

Latent World Model 存在一个问题：

如果没有约束，encoder 可能学习到无意义表示。

例如：

所有状态：

\[ z=[0,0,0,...] \]

那么：

预测未来：

\[ \hat z_{t+1}=0 \]

虽然 loss 可能降低，但是 latent 丢失信息。

因此 SIGReg 约束：

不同状态应该保持可区分。

目标：

保持 latent 空间具有：

- 信息性；
- 稳定性；
- 可预测性。

------

# 14. INTACT 完整训练流程

前面介绍了各个模块，下面从训练数据开始，将整个训练过程串起来。

INTACT 的训练核心目标：

同时学习两个能力：

1. **World Model 能够预测动作导致的未来变化**

\[ (z_t,a_t)\rightarrow z_{t+1} \]

1. **Intent 能够直接生成动作**

\[ Intent\rightarrow Action \]

最终实现：

\[ Goal\ Intent\rightarrow Action \]

即：

根据目标直接产生动作。

------

# 14.1 训练数据形式

机器人轨迹数据通常表示为：

\[ (o_t,a_t,o_{t+1},...,o_g) \]

其中：

- $o_t$：当前观测；
- $a_t$：当前真实执行动作；
- $o_{t+1}$：下一时刻观测；
- $o_g$：最终目标观测。

例如：

```
时间序列：

t:

机器人当前位置
      |
      | 执行动作 a_t
      ↓

t+1:

机器人移动后状态

      |
      |
      ↓

最终:

机器人完成任务状态
```

------

经过视觉编码器：

\[ z_t=E_\theta(o_t) \]\[ z_{t+1}=E_\theta(o_{t+1}) \]\[ z_g=E_\theta(o_g) \]

得到：

当前 latent：

\[ z_t \]

真实下一状态：

\[ z_{t+1} \]

目标 latent：

\[ z_g \]

------

# 14.2 第一步：构造两个 Intent

## （1）Local Intent

由真实轨迹产生：

\[ m^{local}=z_{t+1}-z_t \]

含义：

> 当前动作实际导致了什么变化。

例如：

动作：

\[ a_t=\text{向右移动} \]

产生：

\[ z_t\rightarrow z_{t+1} \]

所以：

\[ m^{local} \]

就是这个动作对应的变化。

------

## （2）Goal Intent

由目标产生：

\[ m^{goal}=sg(z_g)-z_t \]

含义：

> 当前状态距离最终目标还需要发生什么变化。

例如：

当前：

机器人在桌子左边。

目标：

机器人抓住杯子。

那么：

\[ m^{goal} \]

表示：

需要向目标方向变化。

------

# 14.3 第二步：训练 Intent-to-Action Predictor

INTACT Predictor：

\[ \hat a=G_\eta(m) \]

输入 Intent：

输出动作。

------

## Local Intent 分支

输入：

\[ m^{local} \]

得到：

\[ \hat a_t=G_\eta(m^{local}) \]

由于真实动作：

\[ a_t \]

已知。

因此：

计算动作损失：

\[ L_{action} = ||\hat a_t-a_t||^2 \]

作用：

让模型学习：

\[ \boxed{ z_{t+1}-z_t \rightarrow a_t } \]

即：

知道某种状态变化应该如何产生。

------

# 14.4 第三步：训练 Forward World Model

Forward Predictor：

\[ \hat z_{t+1}=F_\phi(z_t,a_t) \]

输入：

当前状态：

\[ z_t \]

真实动作：

\[ a_t \]

预测：

未来状态：

\[ \hat z_{t+1} \]

然后和真实：

\[ z_{t+1} \]

比较。

Loss：

\[ L_{world} = E[ ||\hat z_{t+1}-z_{t+1}||_2^2/d ] + \lambda_{sig}R_{SIG} \]

第一项：

学习环境动力学。

第二项：

保持 latent 表示质量。

------

# 14.5 第四步：Goal Intent 如何参与训练？

这是 INTACT 最关键的地方。

Goal Intent：

\[ m^{goal}=sg(z_g)-z_t \]

输入：

同一个 Intent Predictor：

\[ \hat a_g=G_\eta(m^{goal}) \]

得到预测动作。

注意：

这里的：

\[ \hat a_g \]

不是数据提供的真实标签。

因为：

目标通常不存在唯一动作。

------

那么如何约束？

将预测动作送入 Forward World Model：

\[ \hat z_{t+1} = F_\phi(z_t,\hat a_g) \]

得到：

执行该动作后的预测未来状态。

然后判断：

该状态是否朝目标方向移动。

即：

希望：

\[ \hat z_{t+1}\approx z_g \]

或者：

\[ \hat z_{t+1}-z_t \approx z_g-z_t \]

因此形成目标一致性约束。

------

# 14.6 完整训练流程总结

整体流程：

```
输入：

当前观测 o_t
下一观测 o_(t+1)
目标观测 o_g
真实动作 a_t


        ↓


Encoder


        ↓


得到:

z_t
z_(t+1)
z_g


        ↓


-----------------------------

        ↓

Local Intent:

m_local = z_(t+1)-z_t


        ↓

Intent Predictor


        ↓

预测动作

        ↓

和真实 a_t 比较


-----------------------------


        ↓

Goal Intent:

m_goal = sg(z_g)-z_t


        ↓

Intent Predictor


        ↓

预测动作


        ↓

Forward World Model


        ↓

预测未来状态


        ↓

和目标状态比较


-----------------------------


同时训练：

Forward Predictor

学习：

(z_t,a_t)->z_(t+1)
```

------

# 15. 推理阶段：Search-Free Control

训练完成后，实际执行任务时：

不存在：

\[ z_{t+1} \]

因为未来未知。

因此无法计算：

\[ z_{t+1}-z_t \]

只能使用：

目标状态。

------

## 推理流程

输入：

当前观测：

\[ o_t \]

目标观测：

\[ o_g \]

编码：

\[ z_t=E(o_t) \]\[ z_g=E(o_g) \]

计算：

\[ m^{goal}=z_g-z_t \]

输入：

\[ \hat a=G_\eta(m^{goal}) \]

直接得到动作：

\[ a \]

执行。

------

流程：

```
当前状态

o_t

↓

Encoder

↓

z_t



目标状态

o_g

↓

Encoder

↓

z_g



计算：

z_g-z_t



↓

INTACT Predictor



↓

动作 a



↓

机器人执行
```

------

# 16. 为什么叫 Search-Free？

传统 World Model：

需要：

\[ Action\rightarrow Future \]

但是不知道：

哪个 action 好。

所以：

搜索：

```
动作1
 ↓
预测未来

动作2
 ↓
预测未来

动作3
 ↓
预测未来

选择最好
```

------

INTACT：

直接学习：

\[ Goal\ Intent\rightarrow Action \]

所以：

一次 forward：

即可得到动作。

不需要：

- CEM；
- MPPI；
- Tree Search。

------

# 17. INTACT 与 JEPA 的关系

INTACT 可以看作：

\[ JEPA+World\ Model+Control \]

------

## JEPA

核心：

学习好的 latent 表示。

目标：

预测：

\[ z_{future} \]

例如：

\[ z_t\rightarrow z_{t+k} \]

重点：

Representation Learning。

------

## INTACT

进一步：

利用 latent 变化控制动作。

增加：

\[ Intent\rightarrow Action \]

重点：

Control。

------

关系：

```
JEPA

学习：

状态空间


        ↓


World Model

学习：

状态如何变化


        ↓


INTACT

学习：

如何让状态按照目标变化
```

------

# 18. INTACT 与 LeWM 的关系

你之前看的 LeWM：

属于：

Latent World Model。

核心：

学习：

\[ (z_t,a_t)\rightarrow z_{t+1} \]

即：

动作预测未来。

------

但是 LeWM 本身：

不知道：

目标对应什么动作。

因此需要：

搜索。

------

INTACT：

在 LeWM 类型 World Model 上增加：

\[ Intent\rightarrow Action \]

形成：

```
LeWM:

Action

↓

Future State



INTACT:

Goal Intent

↓

Action

↓

Future State
```

------

因此：

INTACT 不是替代 World Model。

而是在 World Model 上增加：

**可控制接口。**

------

# 19. 总结

INTACT 的核心贡献可以总结为：

## （1）提出 Intent 作为控制接口

传统：

\[ Action\rightarrow Future \]

INTACT：

\[ Intent\rightarrow Action \]

------

## （2）统一 Local Intent 和 Goal Intent

Local：

\[ z_{t+1}-z_t \]

表示：

真实发生的变化。

Goal：

\[ sg(z_g)-z_t \]

表示：

希望发生的变化。

二者共享同一个 Intent-Aaction 映射。

------

## （3）实现 Search-Free World Model

传统：

World Model + 搜索。

INTACT：

World Model + Intent Predictor。

直接：

\[ Goal\rightarrow Action \]

------

## 一句话总结

> INTACT 是一种面向具身智能控制的 latent world model 方法，它通过将状态变化显式表示为 Intent，并利用真实轨迹中的 Local Intent 学习变化到动作的映射，再将该映射迁移到 Goal Intent，使智能体能够直接根据目标状态生成动作，摆脱传统 World Model 依赖搜索规划的问题。

