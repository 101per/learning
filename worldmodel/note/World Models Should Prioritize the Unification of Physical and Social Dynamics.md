# World Models Should Prioritize the Unification of Physical and Social Dynamics

# 1. 基本信息

- **论文名称：** *World Models Should Prioritize the Unification of Physical and Social Dynamics*
- **作者：** Xiaoyuan Zhang, Chengdong Ma, Yizhe Huang, Weidong Huang, Siyuan Qi, Song-Chun Zhu, Xue Feng, Yaodong Yang
- **会议/期刊：** NeurIPS 2025，**Position Paper Track**
- **年份：** 2025
- **展示级别：** **Position Paper；官方未标注 Oral / Spotlight / Poster 等级**
- **研究领域：** World Models / Multi-Agent Systems / Social Simulation / Physical-Social Dynamics
- **World Model 相关度：** ★★★★★
- **是否属于严格意义上的 World Model：** **部分相关——论文研究对象完全是 World Model，但它本身不是一个已实现并训练的 World Model，而是一篇提出研究范式、原则和概念框架的 Position Paper。**

NeurIPS 官方将其明确收录在 **Position Paper Track**，论文核心产出是 ACE Principles 和一个称为 \(WM_{P-S}\) 的物理—社会统一世界模型框架，而非可直接复现的具体神经网络。[NeurIPS 会议录](https://proceedings.neurips.cc/paper_files/paper/2025/hash/d0089c70e9fd42ebddd596d11af15e9e-Abstract-Position_Paper_Track.html?utm_source=chatgpt.com)

------

# 2. 一句话定位

> **这篇论文认为现有 World Model 最大的缺口之一，是把“物理世界如何变化”和“智能体为什么采取行动”割裂开来，因此提出 ACE 原则和 Physical-Social World Model 框架，用统一状态与联合动力学显式建模物理状态与社会状态的双向因果耦合。** [NeurIPS 会议录](https://proceedings.neurips.cc/paper_files/paper/2025/file/d0089c70e9fd42ebddd596d11af15e9e-Paper-Position_Paper_Track.pdf)

------

# 3. 论文简介

当前 World Model 大多分别研究物理动力学或社会行为：前者擅长预测物体、环境和机器人状态，却往往忽略意图、信念、关系和社会规范；后者能够模拟人物行为和多智能体互动，却常缺乏真实物理环境约束。本文认为这种割裂限制了长期预测、因果推理和多智能体决策，因此提出 **ACE Principles**，要求模型处理社会抽象、情境依赖因果以及物理—社会共同演化，并给出 \(WM_{P-S}\) 概念框架，将物理状态与社会状态统一到联合状态转移中。**论文没有训练具体模型，也没有定量实验，其贡献主要是重新定义下一代 World Model 应建模的对象和研究路线。** [NeurIPS 会议录](https://proceedings.neurips.cc/paper_files/paper/2025/file/d0089c70e9fd42ebddd596d11af15e9e-Paper-Position_Paper_Track.pdf)

------

# 4. World Model 技术定位

- 状态空间：

   

  Other — Structured Physical-Social State

  - \(s_{\mathrm{phy}}\)：环境及智能体的物理状态
  - \(s_{\mathrm{soc}}\)：个体信念、目标等社会状态 + 智能体间关系

- **动力学形式：** **Other — Joint Conditional Transition / Coupled Dynamics**

- **是否 Action-conditioned：** **可选**；论文明确指出可以加入 joint action \(a\)，但框架本身并不强制

- **输入：** 当前物理状态、社会状态、多模态感知信息；可附加多智能体动作

- **预测对象：** 下一时刻联合物理—社会状态 \(s'=(s'_{\mathrm{phy}},s'_{\mathrm{soc}})\)

- **是否显式生成未来状态：** **Yes**

- **训练方式：** **未规定**；这是概念框架，可由 Self-supervised / Supervised / RL / Hybrid 实例化

- **主要用途：** Prediction / Planning / Control / Simulation / Multi-Agent Decision Making

论文特别强调：真正需要学习的不是两个彼此独立的
\(T_{\rm phy}\) 和 \(T_{\rm soc}\)，而是一个能够表达二者相互影响的**联合状态转移**。[NeurIPS 会议录](https://proceedings.neurips.cc/paper_files/paper/2025/file/d0089c70e9fd42ebddd596d11af15e9e-Paper-Position_Paper_Track.pdf)

技术路线可以概括为：

> **Physical-Social State World Model → Bidirectionally Coupled Dynamics → Joint Physical & Social State Prediction → Planning / Simulation / Multi-Agent Decision Making**

------

# 5. 核心方法与数学公式

需要先说明一点：**这不是典型“提出网络 + Loss + 训练算法”的方法论文。**下面这些公式的作用主要是把作者提出的研究范式数学化，而不是描述一个已经训练好的 architecture。[NeurIPS 会议录](https://proceedings.neurips.cc/paper_files/paper/2025/file/d0089c70e9fd42ebddd596d11af15e9e-Paper-Position_Paper_Track.pdf)

核心变量：

\[ N=\{1,\dots,n\}:\text{智能体集合},\quad s_{\rm phy}:\text{联合物理状态},\quad s_{\rm soc}:\text{联合社会状态} \]\[ s=(s_{\rm phy},s_{\rm soc}):\text{完整世界状态},\quad a:\text{可选联合动作},\quad s':\text{下一世界状态} \]\[ X:\text{视频/音频/文本等多模态观测},\quad c:\text{社会与物理上下文},\quad T:\text{世界动力学} \]

### 公式 1：统一的 Physical-Social 状态空间

\[ \mathcal S= \mathcal S_{\rm phy}^{env} \times \left(\prod_{i=1}^{n}\mathcal S_{\rm phy}^{i}\right) \times \left(\prod_{i=1}^{n}\mathcal S_{\rm soc}^{i}\right) \times \left(\prod_{\substack{i,j\in N\\i\neq j}} \mathcal S_{\rm soc}^{ij}\right) \]

其中：

\[ \mathcal S_{\rm phy}^{env}:\text{环境物理状态空间} \]\[ \mathcal S_{\rm phy}^{i}:\text{第 }i\text{ 个智能体的物理状态} \]\[ \mathcal S_{\rm soc}^{i}:\text{个体社会/心理状态} \]\[ \mathcal S_{\rm soc}^{ij}:\text{智能体 }i,j\text{ 之间的社会关系} \]

**公式作用：**

把传统 World Model 中的“世界状态”扩展成四部分：**环境物理状态 + 个体物理状态 + 个体社会状态 + 智能体间社会关系**。[NeurIPS 会议录](https://proceedings.neurips.cc/paper_files/paper/2025/file/d0089c70e9fd42ebddd596d11af15e9e-Paper-Position_Paper_Track.pdf)

**直观理解：**

传统模型看到的是“人在哪里、车在哪里、物体怎么运动”；作者认为还应该看到“这个人想干什么、相信什么、和另一个人是什么关系”。

**与 World Model 的关系：**

这是论文最重要的**状态建模扩展**。

------

### 公式 2：联合 Physical-Social Dynamics

\[ s=(s_{\rm phy},s_{\rm soc}) \]

并学习

\[ T(s'\mid s) \]

若显式考虑动作，则可推广为

\[ T(s'\mid s,a) \]

其中：

\[ s:\text{当前完整世界状态},\quad a:\text{多智能体联合动作} \]\[ s':\text{下一世界状态},\quad T:\text{联合状态转移模型} \]

关键在于：

\[ T\neq T_{\rm phy}+T_{\rm soc} \]

即不能简单把两个独立模型拼起来。论文要求

\[ s_{\rm phy}\leftrightarrow s_{\rm soc} \]

形成持续双向影响。[NeurIPS 会议录](https://proceedings.neurips.cc/paper_files/paper/2025/file/d0089c70e9fd42ebddd596d11af15e9e-Paper-Position_Paper_Track.pdf)

**公式作用：**

定义真正需要 World Model 学习的动力学。

**直观理解：**

司机“赶时间”这个社会/心理状态会改变车速；交通事故这个物理事件又可能改变司机情绪和其他司机行为。因此不能先预测车，再单独预测人的心理。

**与 World Model 的关系：**

对应论文最核心的**动力学预测机制**。

------

### 公式 3：社会状态抽象

论文将从感知数据到社会概念的建模写为：

\[ f:X\rightarrow \mathcal S_{\rm soc} \]

其中

\[ X=\{x_v,x_a,x_t,\ldots\} \]

分别表示视频、音频、文本等多模态信息。

训练可抽象为：

\[ \min_f \mathcal L \big( f(x),s_{\rm true} \big) \]

其中：

\[ f:\text{社会状态抽象模型},\quad s_{\rm true}:\text{社会状态监督/目标} \]

**公式作用：**

解决 ACE 中的第一个 A：

> **Abstraction of Social Complexity and Heterogeneity**

也就是如何从表情、语言、动作等观测中得到“信任、意图、目标、规范”等抽象社会变量。[NeurIPS 会议录](https://proceedings.neurips.cc/paper_files/paper/2025/file/d0089c70e9fd42ebddd596d11af15e9e-Paper-Position_Paper_Track.pdf)

**直观理解：**

模型不能只看到“一个人皱眉”；它需要进一步形成“这个人可能不信任对方”这样的社会状态。

**与 World Model 的关系：**

对应 **state representation / social state grounding**。

------

### 公式 4：社会动力学的 Contingent Causality

社会状态变化表示为：

\[ P(s'_{\rm soc}\mid s_{\rm soc},c) \]

其中：

\[ c:\text{文化、场景、关系历史等上下文} \]

作者进一步指出 OOD 情况下社会预测不稳定：

\[ \operatorname{Var} \left[ P(s'_{\rm soc}\mid c_{\rm OOD}) \right] \gg \operatorname{Var} \left[ P(s'_{\rm soc}\mid c_{\rm in}) \right] \] :chatgpt-content-reference{index="8"}  **公式作用：** 形式化 ACE 的第二个原则： > **C — Contingent Causality** **直观理解：** 物理规律通常比较稳定： > 松手 → 物体下落。 但社会规律不是： > 道歉 → 对方原谅。 结果取决于文化、关系、历史、语气和环境。 因此社会 Dynamics 不能被当作固定物理规律学习。 **与 World Model 的关系：** 对应**非平稳、上下文条件化的社会动力学预测**。 --- ### 公式 5：物理—社会纠缠与长期误差放大 论文用互信息表达两个世界不是独立的： \[ I(s_{\rm phy};s_{\rm soc})>0 \]

联合动力学为：

\[ T( s'_{\rm phy}, s'_{\rm soc} \mid s_{\rm phy}, s_{\rm soc} ) \]

同时指出非线性共同演化可能产生：

\[ \Delta s_{t+1} \approx e^{\lambda\Delta t}\Delta s_t, \qquad \lambda>0 \] :chatgpt-content-reference{index="9"}  **公式作用：** 对应 ACE 的第三个原则： > **E — Entangled System Emergence and Co-evolution** **直观理解：** 社会和物理世界互相反馈，小预测错误可能经过多轮 interaction 不断放大。 例如： \[ \text{错误判断驾驶意图} \rightarrow \text{错误预测车辆动作} \rightarrow \text{改变其他司机行为} \rightarrow \text{整体交通预测偏离} \]

**与 World Model 的关系：**

对应 **长期 rollout、联合动力学、emergent behavior 和误差累积**。

------

# 6. 整体方法流程

## 训练阶段

如果未来真正实例化 \(WM_{P-S}\)，论文设想的过程可以概括为：

\[ \text{Multimodal Observation} \rightarrow \begin{cases} \text{Physical Representation}\\ \text{Social Abstraction} \end{cases} \]\[ \rightarrow s_t= (s^{phy}_t,s^{soc}_t) \rightarrow T \rightarrow (\hat s^{phy}_{t+1},\hat s^{soc}_{t+1}) \rightarrow \mathcal L \]

其中社会部分尤其需要从视频、语言、音频等信息抽象出 belief、goal、trust、relation 等变量；随后模型不是分别训练两个独立 Dynamics，而是学习物理与社会状态共同变化的联合转移。ACE 三个原则分别约束**社会表示、社会因果关系以及两个系统的联合演化**。[NeurIPS 会议录](https://proceedings.neurips.cc/paper_files/paper/2025/file/d0089c70e9fd42ebddd596d11af15e9e-Paper-Position_Paper_Track.pdf)

**但必须注意：论文没有真正实现以上训练 pipeline，也没有给出统一 Loss、网络 architecture、optimizer 或训练数据。**

------

## 推理阶段

概念上的推理过程为：

\[ (s_t^{phy},s_t^{soc}) + a_t \rightarrow T \rightarrow (\hat s_{t+1}^{phy},\hat s_{t+1}^{soc}) \]

进一步 rollout：

\[ \hat s_{t+1} \rightarrow \hat s_{t+2} \rightarrow \cdots \rightarrow \hat s_{t+H} \]

再用于：

\[ \text{Prediction / Planning / Policy / Simulation} \]

例如自动驾驶中，不只是预测：

\[ \text{车辆当前位置} \rightarrow \text{未来车辆位置} \]

而是：

\[ \text{车辆状态} + \text{驾驶意图} + \text{其他驾驶者行为} + \text{交通规范} \rightarrow \text{未来交通状态} \]

这正是作者所说的 physical-social bidirectional dynamics。[NeurIPS 会议录](https://proceedings.neurips.cc/paper_files/paper/2025/file/d0089c70e9fd42ebddd596d11af15e9e-Paper-Position_Paper_Track.pdf)

------

# 7. 关键实验结果

这一部分非常重要：

> **这篇论文没有实验。**

它是一篇 **NeurIPS Position Paper，而非 empirical method paper**。[NeurIPS 会议录](https://proceedings.neurips.cc/paper_files/paper/2025/hash/d0089c70e9fd42ebddd596d11af15e9e-Abstract-Position_Paper_Track.html?utm_source=chatgpt.com)

- **主要数据集 / Benchmark：** 无
- **主要对比方法：** 无定量 baseline；主要通过综述比较 Physical WM 与 Social WM
- **核心评价指标：** 无
- **最重要的结果：** 提出 ACE Principles + \(WM_{P-S}\) conceptual framework
- **相比强基线提升：** N/A
- **消融实验结论：** 无

因此阅读这篇论文时**不要寻找 SOTA 数字**。

它的价值不在：

\[ +5\% \text{ performance} \]

而在提出：

\[ \boxed{ \text{World Model 的“World”是不是定义得太窄了？} } \]

------

# 8. 核心创新点

### 创新点 1：问题层面

论文把一个长期被分开研究的问题显式提出来：

\[ \boxed{ \text{Physical World Model} \neq \text{完整 World Model} } \]

因为真实世界中智能体具有：

\[ \text{belief} + \text{goal} + \text{intention} + \text{emotion} + \text{social relation} + \text{norm} \]

这些因素会反过来影响物理动力学。作者把这种割裂称为一个根本性的 **integration gap**。[NeurIPS 会议录](https://proceedings.neurips.cc/paper_files/paper/2025/file/d0089c70e9fd42ebddd596d11af15e9e-Paper-Position_Paper_Track.pdf)

------

### 创新点 2：方法层面

提出 **ACE Principles**：

\[ \boxed{ A+C+E } \]

分别为：

**A — Abstraction of Social Complexity and Heterogeneity**

如何建立多层级社会状态表示。

**C — Contingent Causality of Social Dynamics**

如何处理随时间、文化、关系和上下文变化的社会因果。

**E — Entangled System Emergence and Co-evolution**

如何联合建模物理和社会系统双向反馈与涌现。[NeurIPS 会议录](https://proceedings.neurips.cc/paper_files/paper/2025/file/d0089c70e9fd42ebddd596d11af15e9e-Paper-Position_Paper_Track.pdf)

------

### 创新点 3：理论/机制层面

传统 World Model 常关注：

\[ p(s_{t+1}^{phy}\mid s_t^{phy},a_t) \]

论文希望最终转向：

\[ p( s_{t+1}^{phy}, s_{t+1}^{soc} \mid s_t^{phy}, s_t^{soc}, a_t ) \]

核心变化是：

> **从单一物理动力学预测，转向 Physical-Social Joint Dynamics。**

而且 Social Dynamics 不能简单复制物理动力学的 Markov / stationary 假设。

------

### 创新点 4：应用层面

作者认为统一后的 World Model 可以覆盖：

- Smart Urban Mobility
- Human-AI Teaming
- Government Policy Simulation
- Personalized Well-being
- Crisis Response
- Advanced Game AI [NeurIPS 会议录](https://proceedings.neurips.cc/paper_files/paper/2025/file/d0089c70e9fd42ebddd596d11af15e9e-Paper-Position_Paper_Track.pdf)

其意义在于把 World Model 从：

> **“机器人/视频中的物理预测器”**

向：

> **“有人、有意图、有规则、有关系的真实社会世界模拟器”**

扩展。

------

# 9. 论文真正的亮点

> **这篇论文为什么值得关注？**

我认为它真正值得关注的地方不是 architecture，而是它提出了一个很有价值的 **World Model scope shift**：

\[ \boxed{ \text{Physical World} \rightarrow \text{Physical + Agent + Society World} } \]

当前很多 World Model 研究实际上解决的是：

\[ \text{What will physically happen next?} \]

例如 Dreamer、视频预测、V-JEPA、机器人 dynamics。

而本文进一步问：

\[ \boxed{ \text{Why will an intelligent agent make that happen?} } \]

也就是：

\[ \text{intent} \rightarrow \text{action} \rightarrow \text{physical consequence} \rightarrow \text{new social state} \rightarrow \text{next action} \]

形成闭环。

因此它的价值主要体现在三点：

**第一，它扩展了“世界状态”的定义。**

World 不再只是 pixel、object、geometry、physics，还包括 belief、goal、relationship、norm。

**第二，它挑战了统一 Dynamics 的简单 Markov 假设。**

社会规律具有：

\[ \text{Context-dependent} + \text{Non-stationary} + \text{Strategic} \]

这是传统物理 World Model 很少认真建模的东西。[NeurIPS 会议录](https://proceedings.neurips.cc/paper_files/paper/2025/file/d0089c70e9fd42ebddd596d11af15e9e-Paper-Position_Paper_Track.pdf)

**第三，它提出了一个很可能在未来越来越重要的方向：Social World Model。**

尤其随着机器人从纯 manipulation 进入：

- human-robot interaction
- autonomous driving
- household robot
- multi-agent system
- AI society

仅预测物理状态是不够的。

不过必须严格区分：

> **它不是一次“已经实现的新 World Model 范式突破”，而更像是在提出“下一种 World Model 范式应该长什么样”。**

所以它是**研究议程创新 > 工程/算法创新**。

------

# 10. 局限性

### 1. 最大问题：只有 framework，没有 implementation

论文提出：

\[ WM_{P-S} \]

但没有告诉我们具体应该使用：

- Transformer？
- JEPA？
- Diffusion？
- RSSM？
- LLM？
- Neuro-symbolic architecture？

也没有给出完整 Loss 和训练算法。

所以其核心论点目前更多是**研究假设**，尚未得到实验验证。[NeurIPS 会议录](https://proceedings.neurips.cc/paper_files/paper/2025/file/d0089c70e9fd42ebddd596d11af15e9e-Paper-Position_Paper_Track.pdf)

### 2. Social State 本身极难定义和监督

物理变量：

\[ 位置、速度、质量 \]

通常具有明确含义。

但：

\[ 信任、意图、价值观、关系、规范 \]

往往是隐变量，而且文化依赖、主体依赖。

论文自己也承认存在严重的 symbol grounding 和数据稀缺问题。[NeurIPS 会议录](https://proceedings.neurips.cc/paper_files/paper/2025/file/d0089c70e9fd42ebddd596d11af15e9e-Paper-Position_Paper_Track.pdf)

### 3. 联合系统的长期预测可能更加不稳定

加入 social state 后，World Model 不但没有解决传统 rollout error，反而可能增加：

\[ \text{non-stationarity} + \text{feedback loop} + \text{state explosion} \]

论文甚至明确讨论了类似 Lyapunov error amplification 的问题。[NeurIPS 会议录](https://proceedings.neurips.cc/paper_files/paper/2025/file/d0089c70e9fd42ebddd596d11af15e9e-Paper-Position_Paper_Track.pdf)

### 4. Evaluation 会非常困难

什么叫：

> “正确预测一个人的信任？”

甚至：

> “正确预测一个社会群体五分钟后的关系状态？”

缺乏类似 Atari score、robot success rate 或 FVD 那样统一、客观的 benchmark，而且还涉及文化偏差与伦理问题。论文也把数据、evaluation 和 scaling 列为未来核心研究问题。[NeurIPS 会议录](https://proceedings.neurips.cc/paper_files/paper/2025/file/d0089c70e9fd42ebddd596d11af15e9e-Paper-Position_Paper_Track.pdf)

------

# 11. 与已有工作的区别

| 方法                  | 状态空间                           | 动力学形式                           | Action                     | 预测目标                             | 核心区别                                                     |
| --------------------- | ---------------------------------- | ------------------------------------ | -------------------------- | ------------------------------------ | ------------------------------------------------------------ |
| **本文 \(WM_{P-S}\)** | Structured Physical + Social State | Joint coupled transition             | Optional                   | Physical + Social future state       | 显式把 belief / goal / relation 与物理状态统一，并建模双向因果耦合。[NeurIPS 会议录](https://proceedings.neurips.cc/paper_files/paper/2025/file/d0089c70e9fd42ebddd596d11af15e9e-Paper-Position_Paper_Track.pdf) |
| **DreamerV3**         | Latent categorical state           | Recurrent latent dynamics            | Yes                        | Latent state / reward / continuation | 典型 action-conditioned **physical/control WM**，通过 imagination 学 policy，不显式建模社会认知。[arXiv](https://arxiv.org/abs/2301.04104?utm_source=chatgpt.com) |
| **V-JEPA 2-AC**       | Latent visual representation       | JEPA Predictive                      | Yes                        | Future latent representation         | 从大规模视频学物理表征，再利用少量 robot trajectories 获得 action-conditioned planning；核心仍是 physical dynamics。[AI.Meta](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/?utm_source=chatgpt.com) |
| **Generative Agents** | Text / Memory / Agent states       | LLM autoregressive social simulation | 非显式                     | Human-like behavior / plans          | 很强的社会行为模拟，但物理环境高度抽象，缺少显式 physical dynamics。[数字对象识别](https://doi.org/10.1145/3586183.3606763?utm_source=chatgpt.com) |
| **SWM-AP**            | Agent traits + social system state | Learned social response model        | Yes，mechanism-conditioned | Agent responses                      | 真正实现了 Social World Model，但重点是机制设计和 agent response，而非完整 physical-social co-evolution。[NIPS 会议论文集](https://proceedings.nips.cc/paper_files/paper/2025/hash/a21db07ccc247b6383f78939c8f894c7-Abstract-Conference.html?utm_source=chatgpt.com) |

最关键的区别可以画成：

\[ \text{Dreamer / V-JEPA} = \boxed{\text{Physical Dynamics}} \]\[ \text{Generative Agents / SWM} = \boxed{\text{Social Dynamics}} \]

而本文主张：

\[ \boxed{ \text{Physical Dynamics} \leftrightarrow \text{Social Dynamics} } \]

所以：

> **本文相比已有 World Model，真正新增的不是一种新的 decoder 或 loss，而是要求 World Model 显式拥有“社会状态”以及 Physical ↔ Social 的双向联合动力学。**

------

# 12. 在 World Model 发展路线中的位置

可以把这篇论文放在这样一条路线里：

\[ \text{Ha \& Schmidhuber World Models} \]\[ \downarrow \]\[ \text{Dreamer / MuZero / TD-MPC} \]\[ \text{Latent Physical Dynamics + Planning} \]\[ \downarrow \]\[ \text{Video WM / V-JEPA / 3D WM} \]\[ \text{Large-scale Physical World Understanding} \]

同时另一条线是：

\[ \text{ToM / Generative Agents / Multi-Agent Models} \]\[ \text{Social Dynamics} \]

两条路线在本文汇合：

\[ \boxed{ \text{Unified Physical-Social World Model} } \]\[ \downarrow \]\[ \text{Grounded Social Causality} + \text{Multi-Agent Dynamics} + \text{Physical Simulation} \]\[ \downarrow \]\[ \text{Human-Robot / AI Society / Complex Real-world Simulation} \]

论文自己也把现有研究划分为 physical WM 和 social WM，并指出两者目前几乎总是独立发展。[NeurIPS 会议录](https://proceedings.neurips.cc/paper_files/paper/2025/file/d0089c70e9fd42ebddd596d11af15e9e-Paper-Position_Paper_Track.pdf)

### 它继承了哪一类 World Model 思想

它继承的是最原始的定义：

\[ \boxed{ \text{学习世界状态及其 Dynamics，用于预测和决策} } \]

而不是限定 World Model 必须等于视频生成模型。

### 它改变了哪个关键环节

最主要改变：

\[ \boxed{\text{State}} \]

从：

\[ s_t=s_t^{physical} \]

变成：

\[ s_t=(s_t^{physical},s_t^{social}) \]

然后自然导致 Dynamics 也必须改变：

\[ T_{physical} \rightarrow T_{physical-social} \]

### 它可能影响后续哪类研究

论文给出的路线包括：

\[ \text{Multimodal Data} \]\[ + \text{Neuro-symbolic Social Representation} \]\[ + \text{Physics Simulator} \]\[ + \text{Multi-Agent RL} \]\[ + \text{Context-dependent Causal Learning} \]

最终形成真正能够持续更新的 Physical-Social World Model。[NeurIPS 会议录](https://proceedings.neurips.cc/paper_files/paper/2025/file/d0089c70e9fd42ebddd596d11af15e9e-Paper-Position_Paper_Track.pdf)

------

# 13. 阅读价值判断

- **重要程度：** ★★★★☆
- **创新程度：** ★★★★☆ **（概念创新强，算法创新弱）**
- **技术难度：** ★★★☆☆
- **与 World Model 主线相关度：** ★★★★★
- **是否建议精读：** **建议**

### 精读建议

如果你的目标是梳理 **2025–2026 World Model 技术路线**，这篇论文值得读，但**不用逐页精读参考文献和 appendix**。

最值得看的顺序是：

**第一重点：Section 3.1**

> *The Inextricable Link Between Physical and Social World*

先弄明白作者为什么认为 Physical-only WM 不完整。

**第二重点：Section 3.2 ACE Principles**

这是整篇文章真正的核心：

\[ \boxed{A\rightarrow C\rightarrow E} \]

尤其重点理解：

\[ \text{Abstraction} \rightarrow \text{Contingent Causality} \rightarrow \text{Entangled Co-evolution} \] :chatgpt-content-reference{index="28"}  **第三重点：Figure 3 + Section 3.3** 这是最值得你记住的一张图。 它把论文思想浓缩成： \[ s_{\rm phy} \]\[ \updownarrow \]\[ \boxed{ s=(s_{\rm phy},s_{\rm soc}) \xrightarrow{T} s' } \]\[ \updownarrow \]\[ s_{\rm soc} \] :chatgpt-content-reference{index="29"}  **第四重点：Section 4** 如果你准备自己研究 World Model，这一节其实比前面的 survey 更重要，因为这里直接指出了未来可能形成论文的问题： - Social representation - Non-stationary causal dynamics - Physical-social joint dynamics - Error propagation - Multimodal social datasets - Neuro-symbolic modeling - Evaluation :chatgpt-content-reference{index="30"}  --- # 14. 最终总结 1. **这篇论文认为当前 World Model 最大的结构性缺陷之一，是物理动力学和社会动力学长期被分开建模，导致模型能预测“物体怎么动”，却不一定理解“智能体为什么让它这么动”。** 2. **核心方法不是新的神经网络，而是 ACE Principles + \(WM_{P-S}\) 概念框架：把 physical state 和 social state 合并为完整 world state，并学习二者双向耦合的联合状态转移。** 3. **相比 Dreamer、V-JEPA 等 physical WM，以及 Generative Agents、Social WM 等 social modeling 工作，它真正新增的是“Physical ↔ Social Dynamics 应成为同一个 World Model”的研究范式。** 4. **论文没有 benchmark、没有 SOTA、没有消融实验，因此它并没有实验证明 \(WM_{P-S}\) 已经有效；它是一篇提出问题、形式化问题并给出未来路线的 Position Paper。** :chatgpt-content-reference{index="31"} 5. **它对 World Model 后续发展的潜在意义很大：如果 2025 年的主线是从 pixel generation 向 latent prediction、action-conditioned planning 和 physical foundation models 扩展，那么本文提出的下一步是——让 World Model 不仅理解“物理世界”，还开始理解“生活在这个物理世界里的智能体与社会”。** \]