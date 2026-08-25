# On Token’s Dilemma：面向大视觉语言模型持续学习的动态 MoE 与漂移感知 Token 分配

## 1. 引言

近年来，大视觉语言模型（Large Vision Language Models, LVLMs）在视觉理解、多模态推理以及机器人操作等任务中展现出了强大的能力。例如：

- LLaVA
- BLIP-2
- MiniGPT-4
- GPT-4V

这些模型通过将视觉编码器与大语言模型结合，使模型能够理解图像内容，并根据自然语言指令完成复杂任务。

然而，现实世界中的智能体并不是一次性训练完成的，而是需要不断学习新的任务和知识。

例如机器人：

第一次学习：

> 抓取杯子

之后学习：

> 打开抽屉

进一步学习：

> 整理桌面

如果每次学习新任务都直接更新整个模型参数，会导致：

[
\theta_{old}\rightarrow\theta_{new}
]

新任务获得提升的同时，旧任务性能下降。

这种现象称为：

**灾难性遗忘（Catastrophic Forgetting）**。

因此，如何让 LVLM 具备持续学习能力，成为当前重要研究方向。

------

# 2. 为什么选择 MoE 解决持续学习？

一种直观方法是：

不要让所有任务共享同一组参数，而是引入多个专家（Expert）。

这就是：

## Mixture of Experts（MoE）

MoE结构如下：

```
             Input Token
                  |
                Router
                  |
       ----------------------
       |          |          |
    Expert1    Expert2    Expert3
       |          |          |
       ----------------------
                  |
              Output
```

其中：

Router负责决定：

> 当前输入应该交给哪个专家处理。

数学表示：

对于输入：

[
x
]

Router计算：

[
p=G(x)
]

其中：

[
p=[p_1,p_2,...,p_n]
]

表示选择每个Expert的概率。

最终：

[
y=
\sum_i p_iE_i(x)
]

------

## MoE用于持续学习的基本思想

假设：

任务1：

训练：

[
Expert_1
]

任务2：

新增：

[
Expert_2
]

任务3：

新增：

[
Expert_3
]

旧专家冻结：

```
Task1
 |
Expert1 (Frozen)


Task2
 |
Expert2 (Train)


Task3
 |
Expert3 (Train)
```

这样可以避免新任务覆盖旧知识。

------

# 3. 传统MoE持续学习存在的问题

直觉上：

只要专家隔离，就不会遗忘。

但是论文提出：

> 真正导致遗忘的问题并不是 Expert 参数变化，而是 Token 路由发生变化。

即：

**Routing Drift（路由漂移）**。

------

# 4. 什么是 Routing Drift？

考虑一个已经学习过的任务：

任务：

> 识别杯子

训练完成后：

Token：

[
x_{cup}
]

Router输出：

# [ G(x_{cup})

[0.9,0.1]
]

表示：

90%概率进入：

Expert1。

之后学习新任务：

> 倒水

增加：

Expert2。

训练过程中：

Router参数发生变化。

再次输入：

cup token：

可能变成：

# [ G'(x_{cup})

[0.4,0.6]
]

此时：

原本属于Expert1的token：

进入了Expert2。

虽然：

Expert1参数没有改变。

但是：

访问路径改变。

最终导致：

旧知识无法被调用。

这就是：

[
Routing\ Drift
]

------

# 5. Token’s Dilemma（Token困境）

论文提出：

在持续学习过程中，Token存在两难问题。

对于一个Token：

它可能：

- 对新任务没有价值
- 但是错误路由会破坏旧知识

例如：

机器人任务：

任务1：

> 拿杯子

任务2：

> 倒水

两个任务共享：

```
cup
hand
table
object
```

这些Token：

对于任务2：

不是新知识。

但是如果训练任务2时：

这些Token全部进入新Expert：

会导致：

旧专家和新专家知识边界混乱。

因此：

论文认为：

持续学习不仅需要：

Expert隔离。

还需要：

Token级别的路由控制。

------

# 6. LLaVA-DyMoE整体框架

论文提出：

## Dynamic MoE with Drift-Aware Token Assignment

整体结构：

```
             Image
               |
        Vision Encoder
               |
          Visual Tokens
               |
              LLM
               |
             MoE Layer
               |
            Router
          /        \
 Old Tokens      New Tokens
    |                |
Old Experts      New Experts
```

核心思想：

动态增加专家，同时根据Token特征决定路由。

------

# 7. Drift-Aware Token Assignment

## 7.1 Token路由概率

对于Token：

[
x
]

Router输出：

# [ p(x)

[p_1,p_2,...,p_n]
]

其中：

[
p_i
]

表示选择第i个专家概率。

------

## 7.2 根据Entropy判断Token稳定性

论文利用：

Routing Entropy

衡量Token的不确定程度。

公式：

# [ H(x)

-\sum_i p_i\log p_i
]

------

当：

### Entropy低

例如：

[
[0.95,0.03,0.02]
]

说明：

Token路由明确。

属于：

稳定Token。

------

### Entropy高

例如：

[
[0.33,0.34,0.33]
]

说明：

Router无法判断。

属于：

Ambiguous Token。

------

这些Token更容易发生：

Routing Drift。

因此需要额外约束。

------

# 8. Token Assignment Loss

为了保持旧Token路由稳定：

论文设计：

Token Assignment约束。

目标：

保持：

[
G_{new}(x)
\approx
G_{old}(x)
]

即：

新模型Router不能随意改变旧Token路径。

损失：

# [ L_{assign}

-\sum_x logP(E_{old}|x)
]

作用：

减少：

旧Token进入新专家。

------

# 9. Routing Regularization

除了Token约束：

论文进一步约束专家之间的分工。

希望：

旧专家：

负责旧知识。

新专家：

负责新知识。

因此：

加入：

[
L_{route}
]

使：

[
P(E_{old}|x_{old})

> 

P(E_{new}|x_{old})
]

同时：

[
P(E_{new}|x_{new})

> 

P(E_{old}|x_{new})
]

形成清晰专家边界。

------

# 10. 总体优化目标

最终训练目标：

# [ L

L_{task}
+
\lambda_1L_{assign}
+
\lambda_2L_{route}
]

其中：

## 任务损失

保证模型完成任务：

[
L_{task}
]

## Token Assignment Loss

保持旧知识路径：

[
L_{assign}
]

## Routing Loss

增强专家分工：

[
L_{route}
]

------

# 11. 方法优势

## 1. 避免参数遗忘

传统方法：

只保护参数。

DyMoE：

保护：

参数 + 路由。

------

## 2. 提升专家利用率

传统：

一个任务对应一个专家。

但是实际任务存在共享知识。

Token级路由：

允许：

多个专家共同处理。

------

## 3. 更适合多模态任务

视觉语言模型中：

大量Token具有共享语义：

例如：

- object
- action
- relation

Token级控制可以更细粒度管理知识。

------

# 12. 与Task-level MoE的区别

需要注意：

DyMoE采用：

## Token-level Routing

即：

每个Token选择专家。

而另一类方法：

## Task-level Routing

例如机器人技能专家：

```
Task
 |
Router
 |
Top2 Expert
```

区别：

|          | Token MoE     | Task MoE        |
| -------- | ------------- | --------------- |
| 路由单位 | Token         | 任务            |
| 粒度     | 细            | 粗              |
| 专家含义 | 知识模块      | 技能模块        |
| 适合     | LVLM知识学习  | 机器人技能学习  |
| 主要问题 | Routing Drift | Task Expert匹配 |

------

# 13. 对机器人终身学习的启发

机器人学习具有明显的任务结构：

例如：

```
Expert1:
抓取

Expert2:
推动

Expert3:
导航

Expert4:
装配
```

输入：

> 把杯子放入盒子

可以：

Router：

选择：

[
Top2(E_{grasp},E_{place})
]

只激活相关技能专家。

因此：

机器人领域更适合：

Task-aware Sparse MoE。

但是：

DyMoE提出的思想仍然重要：

即：

> 持续学习不仅需要增加专家，还需要防止路由边界随训练发生漂移。

------

# 14. 总结

《On Token’s Dilemma: Dynamic MoE with Drift-Aware Token Assignment for Continual Learning of Large Vision Language Models》提出了一个重要观点：

> MoE持续学习中的灾难性遗忘，不仅来自专家参数变化，更来自Token路由变化。

论文提出：

1. 动态扩展专家结构；
2. 分析Token路由漂移问题；
3. 根据Token特征进行漂移感知分配；
4. 通过路由约束保持旧知识路径。

其核心贡献可以概括为：

# [ \boxed{ \text{Continual Learning}

\text{Expert Expansion}
+
\text{Stable Routing}
}
]

对于未来机器人终身学习系统，一个可能的发展方向是：

[
\boxed{
\text{Task-aware Sparse MoE}
+
\text{Router Stability Regularization}
+
\text{Expert Knowledge Aggregation}
}
]

即：

根据任务选择专家，同时保证长期学习过程中专家分工不会发生漂移。