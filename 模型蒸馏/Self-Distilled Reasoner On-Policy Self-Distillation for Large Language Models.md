# Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models



> we introduce *On-Policy Self* *Distillation* (OPSD), a learning algorithm where a single LLM acts as both teacher and student with different contexts.we introduce *On-Policy Self* *Distillation* (OPSD), a learning algorithm where a single LLM acts as both teacher and student with different contexts.

==让一个模型同时扮演“老师”和“学生”，利用自身拥有答案信息时的推理能力，去指导没有答案信息时的自己学习。==

## 1. 论文概述

**论文名称：**

Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models

**核心方法：**

On-Policy Self-Distillation（OPSD）

**研究方向：**

- 大语言模型推理能力提升
- 自蒸馏（Self-Distillation）
- 大模型后训练（Post-training）

------

## 2. 研究背景

近年来，大语言模型（Large Language Models, LLMs）在数学推理、代码生成等任务中表现出强大的能力。为了进一步提升模型推理能力，主要采用以下方法：

### 2.1 监督微调（SFT）

SFT通过人工标注数据训练模型，使模型学习输入与答案之间的映射关系。

给定训练数据：

$$
D={(x_i,y_i)}
$$

其中：

- $x_i$ 表示输入问题
- $y_i$ 表示正确答案

优化目标：

$$
\max_\theta
\sum_i
\log P_\theta(y_i|x_i)
$$

缺点：

- 依赖大量高质量标注数据
- 只能学习固定答案，探索能力有限

------

### 2.2 强化学习（RLVR）

强化学习通过奖励函数优化模型输出。

模型生成：

$$
y\sim\pi_\theta(y|x)
$$

根据结果获得奖励：

$$
R(y)
$$

优化目标：

$$
\max_\theta
E_{y\sim\pi_\theta}[R(y)]
$$

缺点：

- 奖励通常只评价最终结果
- 缺少对中间推理过程的指导
- 训练成本较高

------

### 2.3 知识蒸馏（Knowledge Distillation）

传统知识蒸馏利用强大的 Teacher 模型指导 Student 模型。

Teacher：

$$
P_T(y|x)
$$

Student：

$$
P_S(y|x)
$$

目标：

$$
P_S(y|x)\approx P_T(y|x)
$$

通常采用 KL 散度：

$$
L_{KD}=KL(P_T||P_S)
$$

但是传统方法需要额外的大模型作为 Teacher，计算成本较高。

------

# 3. 核心思想

论文提出：

> 利用模型自身已有的推理能力，让同一个模型同时承担 Teacher 和 Student 的角色，通过自蒸馏提升推理能力。

核心思想：

$$
\text{模型自身}
\rightarrow
\text{指导自身学习}
$$

具体来说：

同一个模型：

$$
\pi_\theta
$$

根据输入条件不同，产生两个角色：

------

## 3.1 Student 模型

Student 模拟真实推理场景，只获得问题：

$$
x
$$

生成概率：

$$
P_S(y|x)
$$

------

## 3.2 Teacher 模型

Teacher 获得额外信息，即正确答案：

$$
(x,y^*)
$$

其中 $y^*$ 为真实答案。

Teacher生成概率：

$$
P_T(y|x,y^*)
$$

由于 Teacher 知道答案，因此具有更强的推理能力。

------

# 4. On-Policy Self-Distillation 方法

## 4.1 On-Policy思想

传统蒸馏：

Teacher先生成答案：

$$
y_T\sim P_T(y|x)
$$

然后 Student 学习 Teacher 的输出。

但是：

Teacher生成的数据分布：

$$
P_T(y)
$$

与 Student实际生成分布：

$$
P_S(y)
$$

存在差异。

------

OPSD 方法：

首先让 Student 根据自身策略生成答案：

$$
\hat y\sim P_S(y|x)
$$

然后 Teacher 在 Student 生成的轨迹上提供指导。

因此训练数据符合：

$$
P_{train}=P_{test}
$$

减少分布偏移问题。

------

# 5. 数学形式化

假设 Student 生成推理序列：

$$
\hat y=(\hat y_1,\hat y_2,...,\hat y_T)
$$

其中 $\hat y_t$ 表示第 $t$ 个生成 token。

Student 在第 $t$ 步的预测：

$$
P_S(y_t|x,\hat y_{<t})
$$

Teacher 的预测：

$$
P_T(y_t|x,y^*,\hat y_{<t})
$$

其中：

# $$ \hat y_{<t}

(\hat y_1,\hat y_2,...,\hat y_{t-1})
$$

------

训练目标：

让 Student 学习 Teacher 的 token 分布：

$$
P_S(y_t|x,\hat y_{<t})
\approx
P_T(y_t|x,y^*,\hat y_{<t})
$$

因此采用 KL 散度：

# $$ L_{OPSD}

\sum_t
KL
(
P_T(\cdot|x,y^*,\hat y_{<t})
||
P_S(\cdot|x,\hat y_{<t})
)
$$

该损失函数使 Student 在每一步推理过程中获得更加准确的决策。

------

# 6. 方法流程

OPSD训练过程如下：

### Step 1：Student生成推理过程

输入问题：

$$
x
$$

模型生成：

$$
\hat y
$$

------

### Step 2：构造Teacher

输入：

$$
(x,y^*)
$$

Teacher利用答案信息计算更加准确的预测分布。

------

### Step 3：计算蒸馏损失

比较：

Student：

$$
P_S(y_t|x,\hat y_{<t})
$$

Teacher：

$$
P_T(y_t|x,y^*,\hat y_{<t})
$$

计算：

$$
L_{OPSD}
$$

更新模型参数。

------

# 7. 与传统方法对比

| 方法 | Teacher    | 监督信号    | 特点            |
| ---- | ---------- | ----------- | --------------- |
| SFT  | 人工数据   | 答案        | 依赖标注        |
| RLVR | 无         | 最终奖励    | 反馈稀疏        |
| KD   | 外部大模型 | 概率分布    | 需要额外Teacher |
| OPSD | 自身模型   | Token级概率 | 无需额外Teacher |

------

# 8. 方法创新点

## （1）提出自蒸馏推理框架

传统：

$$
Teacher\rightarrow Student
$$

OPSD：

$$
Model\rightarrow Model
$$

利用模型自身能力提升自身能力。

------

## （2）利用答案作为特权信息

Teacher拥有：

$$
y^*
$$

Student没有：

$$
y^*
$$

通过两者信息差产生监督信号。

------

## （3）结合On-Policy与蒸馏优势

相比强化学习：

- 提供更加密集的token级监督
- 降低训练成本

相比传统蒸馏：

- 不需要额外Teacher模型
- 避免模型分布差异

------

# 9. 方法局限性

## 1. 依赖正确答案

虽然不需要额外Teacher，但仍需要：

$$
y^*
$$

作为监督信息。

------

## 2. Teacher能力受模型限制

如果模型自身推理能力不足，则生成的 Teacher 分布质量有限。

------

## 3. 应用范围仍需扩展

目前主要验证数学推理任务，对于：

- 长文本生成
- Agent任务
- 多模态推理

仍需要进一步研究。

------

# 10. 总结

Self-Distilled Reasoner 提出了一种新的 LLM 后训练方式：

$$
\boxed{
\text{拥有答案信息的模型}
\rightarrow
\text{没有答案信息的模型}
}
$$

通过 On-Policy Self-Distillation，使模型利用自身已有知识增强推理能力。

相比传统方法：

- 不需要额外 Teacher 模型
- 提供 token 级推理监督
- 提高训练效率

其核心贡献可以概括为：

> 让大语言模型通过自身不同状态之间的知识迁移，实现自我增强式推理学习。