#  *DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge*

# 1. 基本信息

- **论文名称：** *DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge*
- **作者：** Wenyao Zhang, Hongsi Liu, Zekun Qi, Yunnan Wang, Xinqiang Yu, Jiazhao Zhang, Runpei Dong, Jiawei He, He Wang, Zhizheng Zhang, Li Yi, Wenjun Zeng, Xin Jin
- **会议/期刊：** NeurIPS 2025 Main Conference
- **年份：** 2025
- **展示级别：** **Poster**
- **研究领域：** Vision-Language-Action / Robot Manipulation / Predictive World Modeling / Embodied AI
- **World Model 相关度：** ★★★★☆
- **是否属于严格意义上的 World Model：** **部分相关**

NeurIPS 官方 proceedings 将其列为 2025 Main Conference；NeurIPS 虚拟会场对应 poster 条目。论文代码与模型也已公开。[NeurIPS 会议录](https://proceedings.neurips.cc/paper_files/paper/2025/hash/22d4f952efa13970f0b1ffb22170d416-Abstract-Conference.html?utm_source=chatgpt.com)

这里先给一个很重要的判断：

> **DreamVLA 更准确地说是“World-Model-enhanced VLA”，而不是经典意义上的独立 World Model。**

原因是经典 World Model 往往学习

\[ p(s_{t+1}\mid s_t,a_t), \]

即给定**当前状态和动作**预测世界如何变化；而 DreamVLA 的核心方向恰恰相反：

\[ \text{当前观测 + 任务} \rightarrow \text{预测未来世界知识} \rightarrow \text{反推动作}. \]

它更接近 **Predictive Inverse Dynamics Model**。世界预测是控制策略内部的“想象/推理中间变量”。[NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)

------

# 2. 一句话定位

> **DreamVLA 不再让 VLA 直接从视觉和语言映射到动作，也不生成完整未来图像，而是先“梦想”未来的运动区域、深度和语义知识，再利用这些紧凑的未来世界表示通过 diffusion inverse dynamics 生成机器人动作。**

------

# 3. 论文简介

现有 VLA 通常直接从视觉和语言预测动作，而引入 world prediction 的方法又常预测完整未来 RGB 图像，既包含大量无关背景，又缺少显式空间与语义结构。DreamVLA 因此提出 **Comprehensive World Knowledge Forecasting**：不重建整张未来图像，而预测未来动态区域、深度几何和高层语义，并通过结构化 attention 防止不同知识相互污染，再使用 Diffusion Transformer 根据未来 world embedding 生成动作。实验中其 CALVIN ABC-D 平均连续完成任务数达到 **4.44**，LIBERO 平均成功率 **92.6%**，真实 Franka 机器人平均成功率 **76.7%**，说明“预测任务相关世界知识”比完整像素预测更适合机器人控制。[NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)

------

# 4. World Model 技术定位

- **状态空间：** **Hybrid — Latent + Structured World Knowledge**

- 动力学形式：

   

  Other / Hybrid

  - World prediction：Transformer-based predictive latent representation
  - Action generation：Diffusion Transformer

- **是否 Action-conditioned：** **No（World Prediction 本身不是 action-conditioned）**

- 输入：

  - Language instruction \(l\)
  - 当前 RGB observation \(o_t\)
  - Robot proprioceptive state \(s_t\)

- 预测对象：

  - Future dynamic region \(\hat f_{t+n}\)
  - Future depth \(\hat d_{t+n}\)
  - Future semantic feature \(\hat c_{t+n}\)
  - Future action chunk \(\hat a_{t:t+n-1}\)

- **是否显式生成未来状态：** **Yes（训练阶段）/ Latent Only（推理阶段）**

- **训练方式：** Supervised / Self-supervised pseudo-label / Imitation Learning Hybrid

- **主要用途：** Prediction / Control / Representation Learning

这里最值得注意的是：推理时并不会真的运行三个 decoder 去生成 depth、semantic、dynamic map。训练时这些预测任务迫使 `<dream>` queries 学出具有未来信息的 world embedding；**部署时 decoder 被移除，只保留内部“未来世界表示”来指导动作**，因此预测能力主要以内隐 latent 的形式存在。[NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)

一句话技术路线：

\[ \boxed{ \text{Current Observation} \rightarrow \text{Latent Future World Knowledge} \rightarrow \text{Inverse Dynamics} \rightarrow \text{Robot Action} } \]

更具体地说：

\[ \text{RGB + Language + State} \rightarrow \text{Dynamic / Depth / Semantic Forecast} \rightarrow \text{World Embedding} \rightarrow \text{Diffusion Action} \]

------

# 5. 核心方法与数学公式

先定义核心变量：

\[ o_t:\text{当前视觉观测},\quad s_t:\text{机器人本体状态},\quad l:\text{语言指令} \]\[ w_{t+n}:\text{未来世界 latent embedding},\quad f_{t+n}:\text{动态区域} \]\[ d_{t+n}:\text{未来深度},\quad c_{t+n}:\text{未来语义表示} \]\[ a_{t:t+n-1}:\text{未来动作序列},\quad M:\text{统一 Transformer 模型} \]

------

### 公式 1：Future World Embedding

论文首先定义：

\[ w_{t+n} = M \left( l,o_t,s_t\mid \langle dream\rangle \right) \]

其中：

\[ l:\text{语言任务},\quad o_t:\text{当前图像},\quad s_t:\text{机器人状态} \]\[ \langle dream\rangle:\text{可学习的未来预测 Query} \]

**公式作用：**

让模型根据当前观测和任务目标产生一个**面向未来的 latent world representation**。

**直观理解：**

机器人看到当前桌面以后，不马上决定机械臂怎么动，而是先在内部形成：

> “如果我要完成这个任务，未来几步环境中哪些东西应该变化？”

这样的 latent “想象”。

**与 World Model 的关系：**

这是 DreamVLA 最核心的 **future-state representation**，相当于整个方法里的隐式 World Model state。[NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)

------

### 公式 2：Comprehensive World Knowledge Prediction

\[ \hat p_{t+n} = P(w_{t+n}) = \left[ \hat f_{t+n}, \hat d_{t+n}, \hat c_{t+n} \right] \]

其中：

\[ \hat f_{t+n}:\text{未来动态区域} \]\[ \hat d_{t+n}:\text{未来深度} \]\[ \hat c_{t+n}:\text{未来高层语义特征} \]

**公式作用：**

把一个抽象的 world embedding 用三个不同视角进行监督：

\[ \boxed{ \text{What moves} + \text{Where it is} + \text{What it is} } \]

也就是：

- Dynamic → **什么会动**
- Depth → **空间在哪里**
- Semantic → **它是什么/有什么意义**

**直观理解：**

与其让机器人预测未来整张照片，不如只告诉它：

> 哪些区域会动？
> 它们在什么深度？
> 它们是什么物体？

这三类信息对动作规划更直接。

**与 World Model 的关系：**

对应**未来状态预测**，也是 DreamVLA 与传统 pixel-world-model 最大的区别。[NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)

------

### 公式 3：Dynamic Region Prediction

论文利用 CoTracker 从视频中找出会运动的区域，并主要监督这些区域：

\[ \mathcal L_{\rm dyn} = \frac{1}{|\mathcal D|} \sum_{x_i\in\mathcal D} \mathbb E_{z\sim Q_\phi(z|x_i)} \left[ -\log P_\psi \left( (x_i)_M\mid z \right) \right] \]

其中：

\[ M:\text{运动区域 mask},\quad z:\text{视觉 latent token} \]\[ Q_\phi:\text{视觉 tokenizer/encoder},\quad P_\psi:\text{重建 decoder} \]

**公式作用：**

模型只需要重点预测未来**与动作有关、会发生变化的区域**，而不是花容量重建背景。

**直观理解：**

假设机器人正在拿杯子：

完整 frame prediction 要预测：

\[ \text{桌子 + 墙 + 相机背景 + 杯子 + 机械臂} \]

DreamVLA 更关心：

\[ \boxed{ \text{杯子 + 机械臂下一步会在哪里} } \]

论文的实验甚至发现：**Dynamic Region 是所有世界知识中贡献最大的部分。**[NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)

**与 World Model 的关系：**

属于 **task-centric dynamics prediction**：不是预测所有变化，而是预测与控制相关的变化。

------

### 公式 4：Future Depth Prediction

\[ \mathcal L_{\rm depth} = \frac{1}{HW} \sum_{i,j} \left( \hat d_{t+n}^{(i,j)} - \alpha d_{t+n}^{(i,j)} \right)^2 \]

其中：

\[ \alpha = \frac{ \sum_{i,j} \hat d_{t+n}^{(i,j)} d_{t+n}^{(i,j)} }{ \sum_{i,j} \left(d_{t+n}^{(i,j)}\right)^2 } \]\[ d_{t+n}:\text{目标未来深度},\quad \hat d_{t+n}:\text{预测深度} \]

**公式作用：**

让 world embedding 包含未来三维空间结构，同时利用 \(\alpha\) 消除 monocular depth 的全局尺度歧义。

**直观理解：**

只知道：

> “杯子会移动”

还不够。

机器人还需要知道：

> “杯子在我前面多远？机械臂应该往哪个空间方向移动？”

所以 depth 为 World Model 注入 geometry。

没有真实 depth 时，作者用 Depth Anything V2 产生 pseudo-ground-truth。[NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)

**与 World Model 的关系：**

对应 **spatial state / geometry prediction**。

------

### 公式 5：Diffusion Inverse Dynamics

最终动作不是 autoregressive token，而是通过 DiT 预测：

\[ \mathcal L_{\rm DiT} = \mathbb E_{\tau,\epsilon} \left[ \left\| \epsilon- \epsilon_\theta \left( \sqrt{\bar\alpha_\tau}a_{t:t+n-1} + \sqrt{1-\bar\alpha_\tau}\epsilon, \tau,c \right) \right\|_2^2 \right] \]

其中：

\[ \epsilon\sim\mathcal N(0,I) \]\[ c:\text{由 Action Query 得到的 latent action embedding} \]\[ \epsilon_\theta:\text{Diffusion Transformer denoiser} \]

**公式作用：**

从 Gaussian noise 逐步去噪生成一个连续的未来 action chunk。

**直观理解：**

World Model 先回答：

> “未来世界应该变成什么样？”

然后 inverse dynamics 回答：

> “那我现在应该怎么动，才能让世界变成那个样子？”

因此整体实际上是：

\[ \boxed{ \text{Desired / Predicted Future} \rightarrow \text{Action} } \]

而不是经典 World Model：

\[ \boxed{ \text{Current State + Action} \rightarrow \text{Future} } \]

这是理解 DreamVLA 最重要的区别之一。[NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)

最后总 Loss 为：

\[ \mathcal L = \lambda_{\rm dyn}\mathcal L_{\rm dyn} + \lambda_{\rm depth}\mathcal L_{\rm depth} + \lambda_{\rm sem}\mathcal L_{\rm sem} + \lambda_{\rm DiT}\mathcal L_{\rm DiT} \]

对应权重：

\[ \lambda_{\rm dyn}=0.1,\quad \lambda_{\rm depth}=0.001,\quad \lambda_{\rm sem}=0.1,\quad \lambda_{\rm DiT}=1. \]

论文在 8 张 A800 上训练，并将大型 SAM 等视觉特征提前预计算，以降低训练时 GPU 开销。[NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)

------

# 6. 整体方法流程

## 训练阶段

DreamVLA 的完整训练流程可以压缩成：

\[ (l,o_t,s_t) \]\[ \downarrow \]\[ \text{Text / Visual / State Encoder} \]\[ \downarrow \]\[ \text{GPT-2-based Unified Transformer} + \langle dream\rangle + \langle action\rangle \]\[ \downarrow \]\[ w_{t+n} \]\[ \downarrow \]\[ \begin{cases} \hat f_{t+n} & \text{Dynamic}\\ \hat d_{t+n} & \text{Depth}\\ \hat c_{t+n} & \text{Semantic} \end{cases} \]

同时：

\[ w_{t+n} \rightarrow \text{Action Query} \rightarrow \text{DiT} \rightarrow \hat a_{t:t+n-1} \]

最终：

\[ \mathcal L_{\rm world} + \mathcal L_{\rm action}. \]

视觉由 MAE 编码，语言使用 CLIP text embedding，本体状态由卷积/全连接网络编码；随后加入 `<dream>` 和 `<action>` queries。三个 dream sub-query 分别处理 dynamic、depth、semantic。论文进一步使用 **Block-wise Structured Attention** 禁止三种 dream query 直接互相 attention，避免 motion、geometry 和 semantics 相互泄漏和产生 gradient noise。[NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)

这点很关键，可以理解成：

\[ q_{\rm dynamic} \not\leftrightarrow q_{\rm depth} \not\leftrightarrow q_{\rm semantic} \]

但它们都可以访问：

\[ \text{Vision + Language + Robot State}. \]

因此最终得到的是**共享输入、分离预测头、统一 world embedding**。

------

## 推理阶段

推理时：

\[ (l,o_t,s_t) \rightarrow \langle dream\rangle \rightarrow w_{t+n} \rightarrow \langle action\rangle \rightarrow \text{DiT} \rightarrow a_{t:t+n-1}. \]

非常重要的是：

\[ \boxed{ \text{Dynamic / Depth / Semantic Decoder 在推理时全部删除} } \]

也就是说机器人并不会真的先生成三张图再执行动作。

训练阶段通过：

\[ \text{future knowledge supervision} \]

迫使 latent \(w_{t+n}\) 学会“想象未来”；部署时直接使用这种 latent knowledge，因此预测带来的 inference overhead 很小。论文测得 RTX 4090 上总推理时间约 **91 ms**，而去掉 dream query 后约 **88 ms**。[NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)

------

# 7. 关键实验结果

- 主要数据集 / Benchmark：
  - CALVIN ABC-D
  - LIBERO-Spatial / Object / Goal / Long
  - DROID 用于机器人预训练
  - 自建 Franka Panda real-world manipulation tasks
- 主要对比方法：
  - OpenVLA
  - \(\pi_0\)
  - UP-VLA
  - Seer
  - VPP
  - RoboVLM
  - Diffusion Policy
  - Octo
  - SpatialVLA
  - CoT-VLA
- 核心评价指标：
  - Success Rate
  - CALVIN Avg. Len.
  - 连续完成 1～5 个任务的成功率

### CALVIN ABC-D

| 方法         | Avg. Len. | 连续完成 5 个任务 |
| ------------ | --------- | ----------------- |
| OpenVLA      | 3.27      | 43.5%             |
| UP-VLA       | 4.08      | 69.9%             |
| Seer         | 4.28      | 74.0%             |
| VPP          | 4.29      | 75.0%             |
| **DreamVLA** | **4.44**  | **78.1%**         |

DreamVLA 相比当时最强的 VPP，Avg. Len. 从 **4.29 → 4.44**，约为 **3.5% 相对提升**；连续完成五任务成功率由 75.0% 提升到 78.1%。[NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)

### LIBERO

\[ \boxed{\text{DreamVLA Average}=92.6\%} \]

相比：

\[ \text{CoT-VLA}=81.1\% \]\[ \text{SpatialVLA}=78.1\% \]\[ \text{OpenVLA}=76.5\%. \]

尤其 LIBERO-Long：

\[ 69.0\% \rightarrow \boxed{89.5\%} \]

对长期控制提升尤其明显。[NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)

### Real Robot

真实 Franka Panda 环境：

\[ \text{DreamVLA}=76.7\% \]

对比：

\[ \text{Diffusion Policy}=50.8\% \]\[ \text{Octo}=45.0\% \]\[ \text{OpenVLA}=35.0\%. \]

也就是说，相比这里最强的 Diffusion Policy：

\[ +25.9\text{ percentage points}. \]

真实机器人实验包括 pick、place 和 drawer manipulation，每类下游任务仅使用 100 条 task-specific demonstration 微调。[NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)

### 最关键的消融

这一组实验其实比 SOTA 数字更值得看：

\[ \text{Vanilla VLA}=3.64 \]

加入 Dynamic Region：

\[ 3.64 \rightarrow 4.32 \]

再加入 Depth：

\[ 4.32 \rightarrow 4.40 \]

再加入 Semantic：

\[ 4.40 \rightarrow 4.44. \]

说明：

\[ \boxed{ \text{Dynamic Region 是最大贡献来源} } \]

而 depth / semantics 更多是补充。[NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)

另外：

\[ \text{Current-state Auxiliary Reconstruction}=4.14 \]

而：

\[ \text{Future Prediction}=4.44 \]

证明提升并不仅仅来自“多任务辅助监督”，而来自**真正预测未来**。[NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)

还有两个很重要的结果：

\[ \text{Optical Flow Prediction}=4.23 \]\[ \text{Dynamic Region Prediction}=4.44 \]

以及：

\[ \text{Vanilla Causal Attention}=3.75 \]\[ \text{Structured Attention}=4.44. \]

这支持论文两个核心观点：**不是未来信息越详细越好，而是越 task-relevant 越好；不同类型的世界知识也不能简单混在一起。**[NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)

------

# 8. 核心创新点

### 创新点 1：问题层面

论文挑战了机器人 World Model 一个很自然但未必正确的假设：

\[ \boxed{ \text{预测未来} = \text{生成未来完整 RGB} } \]

作者认为，对于控制任务：

\[ \text{Future Pixels} \]

包含太多与动作无关的信息。

机器人真正需要的是：

\[ \boxed{ \text{Action-relevant Future Knowledge} } \]

即动态、几何、语义。[NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)

------

### 创新点 2：方法层面

提出 **Comprehensive World Knowledge Forecasting**：

\[ \boxed{ \text{Dynamic} + \text{Depth} + \text{Semantic} } \]

替代完整图像预测。

特别是 Dynamic Region：

\[ \text{Future Image Prediction} \rightarrow \text{Future Change Prediction} \]

这是论文最有效、也是最有方法论意义的设计。

------

### 创新点 3：理论/机制层面

DreamVLA 实际重新定义了机器人中的“World Model representation”。

传统：

\[ z_{t+1} \approx \operatorname{Encode}(o_{t+1}) \]

或者：

\[ \hat o_{t+1} = G(o_t,a_t) \]

DreamVLA 更像：

\[ w_{t+n} = \{ \text{task-relevant future dynamics}, \text{geometry}, \text{semantics} \}. \]

换句话说：

> **好的 World Model state 不一定需要重建整个世界，只需要保存对决策足够的未来信息。**

这是这篇论文最值得关注的思想。

------

### 创新点 4：应用层面

它把：

\[ \text{World Prediction} \]

真正嵌入：

\[ \text{VLA Policy} \]

内部，形成：

\[ \boxed{ \text{Perception} \rightarrow \text{Prediction} \rightarrow \text{Action} } \]

而且 inference 不需要显式运行三个 world decoder，因此相比真正生成未来 RGB，部署效率更友好。[NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)

------

# 9. 论文真正的亮点

> **这篇论文为什么值得关注？**

我认为真正值得记住的不是“4.44 CALVIN”本身，而是它对一个问题给出了很清晰的答案：

\[ \boxed{ \text{机器人到底应该预测未来世界中的什么？} } \]

此前大致有三条路线：

\[ \text{VLA} : (o_t,l)\rightarrow a_t \]

完全不显式预测未来；

或者：

\[ (o_t,l) \rightarrow \hat o_{t+n} \rightarrow a_t \]

生成未来完整图像；

DreamVLA 则提出第三种：

\[ (o_t,l) \rightarrow \underbrace{ (\text{motion},\text{geometry},\text{semantics}) }_{\text{task-relevant future abstraction}} \rightarrow a_t. \]

这实际上是在推动：

\[ \boxed{ \text{Pixel World Model} \rightarrow \text{Task-centric Latent World Model} } \]

对机器人来说，这很可能是一条比“把视频生成模型做得越来越大”更有效率的路线。

另外，消融给出了非常有意思的证据：

> **预测的信息更多，并不必然更好。**

单独预测 Depth、DINO 或 SAM feature 有时甚至会比 Vanilla VLA 更差；最有效的是和行动高度相关的 **Dynamic Region**。论文因此暴露出一个未来很重要的问题：

\[ \boxed{ \text{World Model 应该预测所有可预测的信息， 还是只预测对决策有价值的信息？} } \]

DreamVLA 明显站在后者。[NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)

所以我会把它判断为：

- **不是新的通用 World Model 范式**
- **是机器人 World Model representation 设计上的重要推进**
- **方法论价值明显高于单纯工程调参**

------

# 10. 局限性

### 1. 并不是严格的 action-conditioned dynamics model

DreamVLA 的世界预测是：

\[ (l,o_t,s_t) \rightarrow w_{t+n} \]

而不是：

\[ (s_t,a_t) \rightarrow s_{t+1}. \]

因此它不能像 Dreamer、TD-MPC、V-JEPA 2-AC 一样自然地做：

\[ a^{(1)},a^{(2)},\ldots,a^{(K)} \]

多个候选动作的 imagined rollout。

这也是我将其判断为“部分属于 World Model”的核心原因。

### 2. 未来世界监督仍然高度依赖外部模型

Dynamic region 来源于 CoTracker；无真实深度时使用 Depth Anything V2；semantic supervision 来自 SAM / DINOv2。

因此 DreamVLA 的所谓 comprehensive world knowledge，在一定程度上依赖已有 foundation model 提供的 pseudo-label，而且作者也发现 DINO 特征并不稳定，后续实验甚至去掉了它。[NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)

### 3. World representation 仍以 RGB 为中心

作者明确承认当前主要处理：

- parallel gripper manipulation
- RGB-centric observations
- 有限 geometry / material diversity

目前没有真正做到：

\[ \text{RGB + 3D + tactile + contact + force} \]

统一的 physical world state。[NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)

### 4. 长期规划能力仍未真正验证

CALVIN 最多考察连续 5 个任务。作者自己也承认更复杂的：

- long-horizon dependency
- delayed reward
- sequential tool use
- hierarchical planning
- memory

仍是未来工作。[NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)

所以：

> **DreamVLA 擅长 short-to-medium horizon predictive control，但距离能够长期 rollout 和 planning 的通用机器人 World Model 仍有明显距离。**

------

# 11. 与已有工作的区别

| 方法            | 状态空间                        | 动力学形式                                | Action               | 预测目标                       | 核心区别                                                    |
| --------------- | ------------------------------- | ----------------------------------------- | -------------------- | ------------------------------ | ----------------------------------------------------------- |
| **DreamVLA**    | Latent + Dynamic/Depth/Semantic | Transformer prediction + Diffusion action | World Prediction: No | Task-relevant future knowledge | 不生成完整未来 RGB，预测紧凑世界知识后反推动作              |
| **OpenVLA**     | VLM latent                      | Direct policy                             | Yes                  | Action tokens                  | 基本没有显式 future world prediction                        |
| **Seer / PIDM** | Latent + future RGB             | Transformer predictive model              | Inverse dynamics     | Future RGB + Action            | DreamVLA 的直接前身之一，但 Seer 仍预测完整 RGB             |
| **VPP**         | Video diffusion latent          | Video predictive representation           | Indirect             | Future visual representation   | 利用 VDM 内部 future representation，而非显式结构化世界知识 |
| **UP-VLA**      | Multimodal latent + pixels      | Autoregressive prediction                 | Yes                  | Future visual content + action | 同时做理解、未来预测与控制，但未来预测仍偏视觉像素级        |

OpenVLA 是典型的 observation-to-action VLA；Seer 则提出端到端 Predictive Inverse Dynamics，用未来 RGB foresight 指导 action；VPP 使用 video diffusion model 内部 predictive representation；UP-VLA 联合多模态理解和 future prediction。DreamVLA 在这条路线上的主要新增点，就是把 **dense future image** 替换为 **dynamic + spatial + semantic structured future knowledge**。[arXiv](https://arxiv.org/abs/2406.09246?utm_source=chatgpt.com)

所以重点回答：

> **本文相比已有 World Model，真正新增了什么能力？**

不是：

\[ \text{更强的视频生成} \]

而是：

\[ \boxed{ \text{选择性地预测“对动作真正有用”的未来世界因素} } \]

即从：

\[ \text{forecast everything} \]

变成：

\[ \boxed{ \text{forecast what matters for control} } \]

------

# 12. 在 World Model 发展路线中的位置

针对机器人这一条线，我建议你这样记：

\[ \text{Behavior Cloning VLA} \]\[ \text{RT-2 / OpenVLA} \]\[ \downarrow \]\[ \text{Future Prediction + Action} \]\[ \text{UP-VLA} \]\[ \downarrow \]\[ \text{Predictive Inverse Dynamics} \]\[ \text{Seer / VPP} \]\[ \downarrow \]\[ \boxed{ \text{DreamVLA} } \]\[ \text{Task-centric Future World Knowledge} \]\[ \downarrow \]\[ \text{Action-conditioned Latent World Model} + \text{Planning} + \text{3D / Tactile / Memory} \]

DreamVLA 继承的是 **“先预测未来，再根据未来决定动作”** 的 Predictive Inverse Dynamics 思路，尤其和 Seer 联系非常直接；论文代码本身也建立在 Seer 代码基础上。[GitHub](https://github.com/Zhangwenyao1/DreamVLA?utm_source=chatgpt.com)

它改变的关键环节是：

\[ \boxed{\text{Prediction Target}} \]

从：

\[ \text{Future RGB} \]

变成：

\[ \boxed{ \text{Future Dynamic} + \text{Future Geometry} + \text{Future Semantics} } \]

再进一步，我认为它最自然的后续研究方向应该是：

\[ \text{task-centric prediction} \]\[ + \]\[ \text{action-conditioned dynamics} \]\[ + \]\[ \text{latent rollout} \]

最终得到：

\[ \boxed{ p(z_{t+1}\mid z_t,a_t,l) } \]

但这个 \(z_t\) 不再是完整 pixel reconstruction latent，而是一个**面向控制的 compact world state**。

这可能是 DreamVLA 对真正 World Model 主线最重要的启发。

------

# 13. 阅读价值判断

- **重要程度：** ★★★★☆
- **创新程度：** ★★★★☆
- **技术难度：** ★★★★☆
- **与 World Model 主线相关度：** ★★★★☆
- **是否建议精读：** **强烈建议——如果你的重点包含 Embodied World Model / Robot World Model**

### 精读建议

最值得看的第一处是 **Figure 1**。它实际上把机器人 World Model 的路线分成三代：

\[ \text{Direct VLA} \]\[ \rightarrow \]\[ \text{Two-stage Future Generation + Policy} \]\[ \rightarrow \]\[ \text{Unified Prediction + Action} \]

然后 DreamVLA 进一步提出 comprehensive knowledge prediction。

第二重点是 **Figure 2 + Section 3.1/3.2**。一定要真正搞明白：

\[ \boxed{ \langle dream\rangle \quad\text{vs.}\quad \langle action\rangle } \]

分别承担什么功能，以及为什么 inference 时预测 decoder 可以去掉。[NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)

第三重点是 **Section 3.3**：

> Comprehensive World Knowledge Prediction

重点理解：

\[ \boxed{ \text{Dynamic Region} + \text{Depth} + \text{Semantics} } \]

尤其是为什么作者宁可预测 dynamic mask，也不直接预测 optical flow。[NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)

第四重点是 **Figure 4：Block-wise Structured Attention**。

要理解：

\[ q_{\rm dyn}, q_{\rm depth}, q_{\rm sem} \]

为什么需要 disentangle，以及为什么 naive causal attention 会从：

\[ 4.44 \rightarrow 3.75. \]

这是 architecture 上最值得仔细看的地方。[NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)

第五重点反而不是主结果表，而是 **Section 4.4 Ablation**。

特别是牢记这三个结果：

\[ \boxed{ 3.64 \rightarrow 4.32 \rightarrow 4.40 \rightarrow 4.44 } \]

以及：

\[ \text{Reconstruction}=4.14 < \text{Prediction}=4.44 \]

和：

\[ \text{Optical Flow}=4.23 < \text{Dynamic Region}=4.44. \]

这三个实验比“又一个 CALVIN SOTA”更能说明这篇论文真正的研究价值。[NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)

------

# 14. 最终总结

1. **DreamVLA 解决的是机器人 VLA 中“未来到底应该预测什么”的问题：它认为生成完整未来 RGB 冗余且昂贵，因此只预测动态区域、深度和语义等与控制相关的世界知识。**
2. **核心方法是 `<dream>` queries 学习 future world embedding，再通过 Dynamic / Depth / Semantic supervision 约束这一表示，最后用 Diffusion Transformer 做 inverse dynamics，把预测未来转化为机器人动作。**
3. **相比 OpenVLA 的直接 observation-to-action，以及 Seer 等 future-RGB predictive policy，DreamVLA 真正的新意是从“像素级未来预测”转向“task-centric structured future prediction”。**
4. **实验支持很充分：CALVIN ABC-D 达到 4.44，LIBERO 平均 92.6%，真实机器人 76.7%；尤其消融证明 Dynamic Region、Future Prediction 和 Structured Attention 确实是性能来源，而不是简单堆辅助任务。** [NeurIPS Papers](https://papers.neurips.cc/paper_files/paper/2025/file/22d4f952efa13970f0b1ffb22170d416-Paper-Conference.pdf)
5. **从 World Model 发展路线看，它还不是完整的 action-conditioned forward model，但提出了一个很重要的方向：未来机器人 World Model 未必需要“完整模拟整个世界”，更可能需要学习一种专门服务于规划与控制的 compact、task-relevant world representation。**

**如果你在梳理 2025 World Model 主线，我建议把 DreamVLA 和 Seer、VPP、UP-VLA 放在一起看。它们恰好形成了一条非常清晰的机器人路线：**

\[ \boxed{ \text{Direct VLA} \rightarrow \text{Future Image Prediction} \rightarrow \text{Predictive Latent Representation} \rightarrow \text{Task-Centric World Knowledge} } \]

这条线和 Dreamer / V-JEPA 那条“经典 action-conditioned latent dynamics”路线最终很可能会汇合。