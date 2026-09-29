下面把你今天围绕 **LeWM** 产生的几个问题串起来，整理成一份比较完整的技术笔记。

# LeWM 今日知识总结

## 1. LeWM 到底在做什么

LeWM 的核心思想可以概括为：

> **不直接预测未来图像，而是在 latent space 中预测执行动作之后的未来状态表示。**

当前观测先经过 Encoder：

\[ z_t = E(o_t) \]

然后 Predictor 根据当前 latent 和动作预测下一时刻 latent：

\[ \hat z_{t+1}=P(z_t,a_t) \]

真实的下一帧也经过 Encoder：

\[ z_{t+1}=E(o_{t+1}) \]

训练时让：

\[ \hat z_{t+1}\approx z_{t+1} \]

最基本的预测损失就是：

\[ L_{\text{pred}} = \|\hat z_{t+1}-z_{t+1}\|_2^2 \]

因此 LeWM 学习的不是：

\[ \text{现在长什么样} \rightarrow \text{未来图像长什么样} \]

而是：

\[ \text{当前状态表示}+\text{动作} \rightarrow \text{未来状态表示} \]

------

# 2. LeWM 是一种 latent World Model

世界模型可以先问一个问题：

> **它到底预测什么？**

常见的有：

| 类型                   | 预测对象                         |
| ---------------------- | -------------------------------- |
| Pixel World Model      | 未来图像像素                     |
| Token World Model      | 离散视觉 Token                   |
| Latent World Model     | 连续 latent representation       |
| Structured World Model | 位置、速度、物体关系等结构化状态 |

LeWM 属于：

\[ \boxed{\text{Latent World Model}} \]

它没有必要把未来重新解码成 RGB 图像。

因此它更关心：

> **未来状态“是什么”，而不是未来画面“长什么样”。**

------

# 3. LeWM 的 rollout 是自回归的

LeWM 在预测多步未来的时候，会进行 latent rollout：

\[ z_t \overset{a_t}{\longrightarrow} \hat z_{t+1} \]

接下来再把预测结果继续输入 Predictor：

\[ \hat z_{t+1} \overset{a_{t+1}}{\longrightarrow} \hat z_{t+2} \]

再继续：

\[ \hat z_{t+2} \overset{a_{t+2}}{\longrightarrow} \hat z_{t+3} \]

所以从 rollout 的角度，它属于：

\[ \boxed{\text{latent-space autoregressive rollout}} \]

这里的“自回归”不是语言模型那种：

\[ token_1\rightarrow token_2\rightarrow token_3 \]

而是：

\[ z_t\rightarrow z_{t+1}\rightarrow z_{t+2} \]

即**潜状态自回归**。

------

# 4. Predictor 并不是 LeWM 独有的

普通世界模型同样有类似 Predictor / Dynamics Model。

区别主要在于它预测什么。

像素世界模型：

\[ (o_t,a_t) \rightarrow P \rightarrow \hat o_{t+1} \]

LeWM：

\[ (z_t,a_t) \rightarrow P \rightarrow \hat z_{t+1} \]

因此：

> **LeWM 的关键并不是“用了 Predictor”，而是 Predictor 工作在 latent space 中。**

------

# 5. LeWM 和 JEPA 的关系

JEPA 的核心思想可以简单概括为：

\[ \boxed{\text{predict representations rather than pixels}} \]

即：

> 不要求重建像素，而是在 representation space 中预测目标表示。

因此 LeWM 和 JEPA 的思想非常接近：

\[ \text{当前 latent} \rightarrow \text{预测未来 latent} \]

但要注意：

> **JEPA ≠ latent 自回归。**

JEPA 是一个更大的思想：

\[ \boxed{\text{Latent Prediction}} \]

而自回归只是其中一种预测方式。

例如 I-JEPA：

\[ z_{\text{context}} \rightarrow z_{\text{target}} \]

通常并不是：

\[ z_1\rightarrow z_2\rightarrow z_3 \]

而 LeWM 做多步 rollout 时才形成：

\[ z_t \rightarrow \hat z_{t+1} \rightarrow \hat z_{t+2} \rightarrow\cdots \]

所以：

> **JEPA = 在 latent 中预测；LeWM = 把这种思想进一步用于 action-conditioned dynamics prediction。**

------

# 6. 为什么 LeWM 会出现 collapse

这是今天比较重要的一个点。

如果训练目标只有：

\[ L_{\text{pred}} = \|\hat z_{t+1}-z_{t+1}\|^2 \]

那么 Encoder 和 Predictor 可能找到一个非常“作弊”的解。

假设 Encoder 学成：

\[ E(o)=c \]

不管输入什么图片：

\[ o_1,o_2,o_3,\cdots \]

全部变成：

\[ z_1=z_2=z_3=\cdots=c \]

那么 Predictor 只需要：

\[ P(c,a)=c \]

于是：

\[ \hat z_{t+1}=z_{t+1}=c \]

所以：

\[ L_{\text{pred}}=0 \]

模型看起来预测得完美。

但是实际上：

\[ \boxed{\text{latent 已经完全没有信息了}} \]

比如：

- 球在左边
- 球在右边
- 球在运动
- 球停止

理论上都应该有不同表示，但 collapse 后：

\[ z_{\text{left}} = z_{\text{right}} = z_{\text{moving}} = z_{\text{stop}} \]

这就是 **representation collapse**。

所以 collapse 的本质不是：

> Predictor 只喜欢预测某一个答案。

而是：

> **Encoder 把不同世界状态都映射成了几乎相同的表示，从而让预测任务变得毫无意义。**

------

# 7. 为什么 LeWM 要训练 Encoder

如果 Encoder 完全冻结，比如使用一个已经训练好的 CLIP：

\[ o\rightarrow E_{\text{fixed}}(o) \]

那么不同输入本身已经有不同 latent。

此时 Predictor 很难通过让 Encoder collapse 来作弊，因为 Encoder 根本不能改。

但 LeWM 希望：

> **让表示本身也适应物理世界的动态预测任务。**

所以 Encoder 也参与学习。

这样有好处：

\[ \text{Encoder} \rightarrow \text{learn dynamics-friendly representation} \]

但同时也带来了 collapse 风险。

所以：

\[ \boxed{\text{Encoder 可训练}} \]

和

\[ \boxed{\text{需要防 collapse}} \]

通常是联系在一起的。

------

# 8. 为什么有些 JEPA 使用 EMA，而 LeWM 使用 SIGReg

一种常见的防 collapse 方法是 teacher-student / EMA。

例如：

\[ z_{\text{target}} = E_{\text{EMA}}(o_{t+1}) \]

target encoder 不直接反向传播，而是：

\[ \theta_{\text{target}} \leftarrow \tau\theta_{\text{target}} + (1-\tau)\theta_{\text{online}} \]

这样预测目标比较稳定。

但是你看的 LeWM 并没有走这条路线。

它采用的是：

\[ \boxed{\text{SIGReg}} \]

------

# 9. SIGReg 的核心逻辑

可以把 SIGReg 理解成一句话：

> **你可以优化 latent，但不允许所有 latent 挤到同一个地方。**

只有 prediction loss 时：

\[ z_1=z_2=\cdots=c \]

也是合法最优解。

SIGReg 就会额外要求：

\[ z \]

在 latent space 中保持足够的**分散性、方差和结构**。

直觉上希望 latent distribution 类似一个展开的、各向比较均匀的分布，而不是：

```
正常：

      •       •
   •      •
       •       •
 •        •
      •

Collapse：

          •••••
           •••
```

因此两个 loss 各司其职：

\[ L = L_{\text{pred}} + \lambda L_{\text{SIGReg}} \]

其中：

### Prediction Loss

负责：

> **未来必须可以预测。**

\[ \hat z_{t+1}\approx z_{t+1} \]

### SIGReg

负责：

> **但不许通过所有 latent 都变成一样来作弊。**

最终逼迫 Encoder 学到：

\[ \boxed{\text{有区分度 + 可以预测}} \]

的 latent representation。

这其实是 LeWM 非常核心的一点。

------

# 10. LeWM 和像素生成 World Model 的区别

传统像素世界模型可能做：

\[ (o_t,a_t) \rightarrow \hat o_{t+1} \]

它必须预测：

- 颜色
- 纹理
- 光照
- 背景
- 物体细节
- 噪声

很多信息其实与物理决策没有直接关系。

而 LeWM：

\[ (z_t,a_t) \rightarrow \hat z_{t+1} \]

只需要在 latent 中保留对未来动力学有用的信息。

例如一个球滚动：

像素模型可能需要预测：

> 球的颜色、阴影、地板纹理、边缘像素……

LeWM 更希望 latent 抓住：

> 球在哪里、往哪里运动、动作以后状态怎么改变。

所以它的核心理念是：

\[ \boxed{\text{Predict what matters, rather than reconstruct everything}} \]

------

# 11. LeWM 和 Diffusion World Model 的区别

LeWM 的 rollout 更像：

\[ z_t \rightarrow z_{t+1} \rightarrow z_{t+2} \rightarrow z_{t+3} \]

属于逐步预测。

Diffusion World Model 更像：

\[ \text{noise} \rightarrow \text{denoise} \rightarrow \text{denoise} \rightarrow \text{future} \]

所以：

**LeWM：**

\[ \boxed{\text{Autoregressive latent dynamics}} \]

**Diffusion WM：**

\[ \boxed{\text{Diffusion-based future generation}} \]

Diffusion 一个重要优势是比较容易表示：

> **未来存在多种可能性。**

而 LeWM 更强调学习一个可用于预测、规划和理解动态的 latent space。

------

# 12. 今天形成的世界模型分类框架

你今天最后形成的这个分类思路其实很有用，但需要分成几个独立维度。

### 第一维：预测什么

\[ \boxed{\text{Pixel / Token / Latent / Structured State}} \]

其中 Token 本质上也可以看作：

\[ \text{Discrete Latent} \]

------

### 第二维：怎么预测

\[ \boxed{ \text{Autoregressive / Diffusion / Direct Prediction} } \]

例如：

| 表示   | 方法              |
| ------ | ----------------- |
| Pixel  | Autoregressive    |
| Pixel  | Diffusion         |
| Token  | Autoregressive    |
| Latent | Autoregressive    |
| Latent | Diffusion         |
| Latent | Direct Prediction |

LeWM 大致处于：

\[ \boxed{ \text{Continuous Latent} + \text{Autoregressive Rollout} } \]

------

# 13. 生成式 World Model 和预测式 World Model

还有一个更宏观的区分。

### Generative World Model

目标是：

> **把未来生成出来。**

例如：

\[ o_t,a_t \rightarrow o_{t+1} \]

最终可以得到未来图像、视频或 token。

------

### Predictive Representation World Model

目标是：

> **理解未来状态，而不一定把未来画出来。**

例如：

\[ z_t,a_t \rightarrow z_{t+1} \]

JEPA、LeWM 更接近这条路线。

所以不能简单说：

> “所有 World Model 都是生成模型。”

更准确的是：

\[ \boxed{ \text{World Model} \supset \text{Generative} + \text{Predictive Representation} } \]

------

# 14. 最后用一条链把 LeWM 串起来

你现在可以这样理解整篇 LeWM 的核心逻辑：

\[ o_t \overset{Encoder}{\longrightarrow} z_t \]

↓

加入动作：

\[ (z_t,a_t) \overset{Predictor}{\longrightarrow} \hat z_{t+1} \]

↓

用真实未来观测得到：

\[ o_{t+1} \overset{Encoder}{\longrightarrow} z_{t+1} \]

↓

要求：

\[ \hat z_{t+1}\approx z_{t+1} \]

↓

但是单独这么训练会出现：

\[ z_1=z_2=\cdots=c \]

↓

所以加入：

\[ L_{\text{SIGReg}} \]

防止 representation collapse。

↓

然后 Predictor 可以连续 rollout：

\[ z_t \rightarrow \hat z_{t+1} \rightarrow \hat z_{t+2} \rightarrow\cdots \]

↓

最终得到：

\[ \boxed{\text{能够预测动作后果的 latent world model}} \]

------

## 一句话总结今天对 LeWM 的理解

> **LeWM 是一种 JEPA 风格的潜空间世界模型：它训练 Encoder 将视觉观测映射到 latent space，再利用 action-conditioned Predictor 自回归地预测未来 latent；为了避免仅靠 prediction loss 导致所有表示塌缩到同一点，LeWM 使用 SIGReg 约束 latent 表示的分布，使模型最终学到既具有区分性、又能够描述世界动态变化的表示，而无需直接生成未来像素。**