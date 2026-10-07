# RLVR-World: Training World Models with Reinforcement Learning  

# 1. 基本信息

- **论文名称：** RLVR-World: Training World Models with Reinforcement Learning  
- **会议/期刊：** NeurIPS 2025  
- **年份：** 2025  
- **展示级别：** Poster（NeurIPS 2025）  
- **研究领域：** World Model / Reinforcement Learning / Video Generation / Generative Models  
- **World Model 相关度：** ★★★★★  
- **是否属于严格意义上的 World Model：** 是

------

# 2. 一句话定位

> **RLVR-World 提出了一种利用强化学习优化视频世界模型的方法，通过可验证奖励（Reinforcement Learning with Verifiable Rewards, RLVR）直接训练世界模型，使其生成的视频不仅具有视觉真实性，还满足物理规律和环境交互一致性。**

------

# 3. 论文简介

传统世界模型通常依赖大规模视频数据进行监督学习，通过最大似然目标学习环境动态。然而，这类方法容易产生视觉合理但物理错误的预测，例如物体运动不符合动力学规律、长期状态演化不稳定等。RLVR-World 认为世界模型训练不应局限于被动模仿数据，而应像强化学习中的智能体一样，通过环境反馈不断优化。

该工作提出将 **Reinforcement Learning with Verifiable Rewards（RLVR）** 引入世界模型训练，通过设计可验证奖励函数，对生成视频中的物理一致性、状态变化合理性以及未来预测质量进行评价，并利用强化学习优化生成模型参数。实验表明，RLVR-World 能够提升世界模型在长时间视频预测、物理规律保持以及交互式模拟任务中的表现，证明强化学习可以成为训练下一代世界模型的重要范式。

------

# 4. 背景与研究动机

## 4.1 现有 World Model 的主要问题

近年来的视频世界模型（如基于 Diffusion Transformer 的模型）取得了巨大进展：

- 能生成高质量视频；
- 能预测未来场景变化；
- 能作为机器人和智能体的模拟环境。

但是，它们主要依赖：

\[ p(x_{t+1:T}|x_{1:t}) \]

即：

> 根据历史观察预测未来视觉状态。

这种训练方式存在几个问题：

### （1）缺少真实世界约束

模型学习的是：

> “什么视频看起来像真实的”

而不是：

> “什么状态转移符合真实世界规律”。

例如：

- 球可以穿过墙；
- 物体突然消失；
- 重力规律错误；
- 人体动作不符合运动学。

------

### （2）监督数据无法覆盖所有未来

真实世界状态空间巨大：

\[ S \rightarrow S' \]

有限视频数据无法覆盖所有可能情况。

因此模型容易：

- 插值有效；
- 外推失败。

------

### （3）世界模型目标与智能体需求不一致

机器人真正需要：

> 一个可以预测行动后果的环境模型。

例如：

机器人：

```
抓杯子
 ↓
世界模型预测
 ↓
杯子移动、倾倒、水流变化
```

而普通视频生成模型只需要：

```
生成看起来合理的视频
```

两者目标不同。

------

# 5. 核心思想

RLVR-World 的核心观点：

> **世界模型不应该只是模仿过去，而应该通过奖励机制学习“什么样的未来更符合世界规律”。**

整体框架：

```
          Video Data
              |
              ↓
      Initial World Model
              |
              ↓
      Generate Future World
              |
              ↓
    Verifiable Reward Model
              |
              ↓
     Reinforcement Learning
              |
              ↓
      Improved World Model
```

------

# 6. 方法详解

## 6.1 基础 World Model

模型首先学习：

\[ W_\theta: (s_t,a_t) \rightarrow s_{t+1} \]

其中：

- \(s_t\)：当前世界状态
- \(a_t\)：动作
- \(s_{t+1}\)：未来状态

对于视频世界模型：

状态通常表示：

\[ s_t = video\ frame \]

------

## 6.2 引入 Verifiable Reward

论文核心创新：

不是使用传统：

\[ \max \log p(video) \]

而优化：

\[ \max E[R(video)] \]

奖励函数评价：

### ① 物理一致性奖励

例如：

- 重力；
- 碰撞；
- 速度连续性；
- 物体持久性。

------

### ② 时间一致性奖励

检查：

未来帧是否：

- 连贯；
- 无闪烁；
- 状态连续。

------

### ③ 任务相关奖励

例如：

机器人任务：

```
action:
push object

prediction:
object moves
```

如果预测符合动作效果：

reward ↑

------

# 7. 强化学习训练过程

类似 RLHF：

```
生成未来
    |
    ↓
Reward Model评分
    |
    ↓
Policy Optimization
    |
    ↓
更新World Model
```

可能采用：

- PPO 类优化；
- policy gradient；
- reward-weighted optimization。

优化目标：

\[ \nabla_\theta E[R(W_\theta)] \]

------

# 8. 与传统 World Model 的区别

| 方法            | 训练方式        | 优化目标       |
| --------------- | --------------- | -------------- |
| Video Diffusion | 最大似然        | 视频像不像     |
| V-JEPA          | 表征预测        | 学习世界表示   |
| Genie           | 行动预测        | 环境模拟       |
| Dreamer         | RL latent model | 控制策略       |
| **RLVR-World**  | 强化学习        | 世界规律一致性 |

核心区别：

> 从“预测数据分布”转向“优化世界规律”。

------

# 9. 实验与结果

论文主要验证：

## 9.1 视频未来预测

测试：

- 长时间 rollout；
- 多步骤预测。

结果：

RLVR-World：

- 更少状态漂移；
- 更稳定长期预测。

------

## 9.2 物理推理能力

测试：

- 物体交互；
- 动力学规律；
- 因果关系。

相比监督训练：

提高：

- 物理合理性；
- 动态一致性。

------

## 9.3 Agent Simulation

作为环境：

```
Agent
 |
 action
 ↓
RLVR-World
 |
 predicted future
```

模型更适合作为：

- robot simulator；
- embodied AI environment。

------

# 10. 主要贡献总结

## Contribution 1

提出：

> 首个利用 RLVR 训练世界模型的方法。

将强化学习从：

```
训练agent
```

扩展到：

```
训练environment model
```

------

## Contribution 2

提出可验证奖励机制：

解决：

> 世界模型不知道什么是“正确未来”。

------

## Contribution 3

证明：

强化学习能够提升：

- 视频生成；
- 长期预测；
- 物理一致性。

------

# 11. 与 World Model 发展路线关系

World Model 演化：

```
阶段1:
预测未来
(Video Prediction)

        ↓

阶段2:
学习环境表示
(Latent World Model)

        ↓

阶段3:
支持行动模拟
(Interactive World Model)

        ↓

阶段4:
自我改进世界模型

        ↓

RLVR-World
```

RLVR-World 属于：

> **第四阶段：通过反馈机制训练世界模型。**

------

# 12. 对未来研究的意义

这篇论文的重要意义在于：

传统观点：

> 世界模型 = 学习现实世界的数据分布

RLVR-World 提出：

> 世界模型 = 学习一个能够通过反馈不断纠正自己的环境模拟器。

这更加接近：

- 人类学习世界；
- 动物通过试错建立认知；
- AGI 所需要的内部环境模型。

------

# 13. 局限性

### 1. Reward Design 依赖人工

如果奖励设计错误：

模型可能优化错误目标。

------

### 2. 计算成本高

RL 训练生成模型：

- sampling 大；
- rollout 大；
- GPU 消耗高。

------

### 3. 真实性 ≠ 智能

提高视频物理合理性：

并不一定意味着：

- 更强推理；
- 更强规划能力。

------

# 14. 与其他 2025 World Model 工作关系

| 论文                                                         | 核心方向             | 关系             |
| ------------------------------------------------------------ | -------------------- | ---------------- |
| World Models Should Prioritize the Unification of Physical and Social Dynamics | 统一物理+社会动力学  | 定义未来目标     |
| DreamVLA                                                     | 视觉语言动作世界模型 | 面向机器人       |
| WorldModelBench                                              | 评价世界模型         | 建立测试体系     |
| **RLVR-World**                                               | 强化学习训练世界模型 | 提出新的训练范式 |

------

# 15. 最终评价

**RLVR-World 是 2025 年 World Model 方向的重要探索之一。**

它的最大价值不是提出一个更强的视频生成模型，而是改变训练范式：

> **未来世界模型可能不再只是从数据中学习世界，而是在与奖励和反馈交互中形成对世界规律的理解。**

如果 World Model 被认为是 AGI 的“内部模拟器”，那么 RLVR-World 提供了一条重要路径：

\[ \boxed{ Data\ Learning \rightarrow Feedback\ Learning \rightarrow Self\ Improving\ World\ Model } \]

**World Model 相关度：★★★★★**
**创新性：★★★★☆**
**对未来 AGI 影响：★★★★★**