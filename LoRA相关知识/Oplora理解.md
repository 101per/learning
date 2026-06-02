# 以下以形式化语言（定义–命题–定理风格）介绍该论文。

---

> ## 1. 问题设定与符号系统
>
> **定义 1（预训练权重矩阵）**  
> 设预训练模型的某层权重矩阵为  
> $$W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}},$$  
> 其在微调过程中保持冻结。
>
> **定义 2（LoRA 参数化）**  
> Low-Rank Adaptation（LoRA）将任务适配建模为低秩更新：  
> $$\Delta W = \frac{\alpha}{r} BA, \quad B \in \mathbb{R}^{d_{\text{out}} \times r},\; A \in \mathbb{R}^{r \times d_{\text{in}}},$$  
> 其中 $r \ll \min(d_{\text{out}}, d_{\text{in}})$，$\alpha$ 为缩放因子。微调后的前向传播为  
> $$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} BAx.$$
>
> **定义 3（预训练知识的谱分解）**  
> 对 $W_0$ 施行奇异值分解（SVD）：  
> $$W_0 = U\Sigma V^\top = \sum_{i=1}^{\min(d_{\text{out}}, d_{\text{in}})} \sigma_i u_i v_i^\top,$$  
> 其中奇异值按降序排列 $\sigma_1 \geq \sigma_2 \geq \cdots$。记前 $k$ 个左、右奇异向量构成的矩阵分别为  
> $$U_k = [u_1, \dots, u_k] \in \mathbb{R}^{d_{\text{out}} \times k}, \quad V_k = [v_1, \dots, v_k] \in \mathbb{R}^{d_{\text{in}} \times k}.$$
>
> ---
>
> ## 2. 灾难性遗忘的量化
>
> **定义 4（子空间对齐度量）**  
> 令 $Q_k = U_k U_k^\top$ 为到 $\operatorname{span}\{U_k\}$ 的正交投影算子。将 LoRA 更新分解为  
> $$\Delta W = Q_k \Delta W + (I - Q_k)\Delta W.$$  
> 由正交投影的勾股定理，Frobenius 范数满足  
> $$\|\Delta W\|_F^2 = \|Q_k \Delta W\|_F^2 + \|(I - Q_k)\Delta W\|_F^2.$$  
> 定义子空间对齐度量 $\rho_k$ 为  
> $$\rho_k := \frac{\|Q_k \Delta W\|_F^2}{\|\Delta W\|_F^2} \in [0,1].$$
>
> **解释**  
> $\rho_k \to 1$ 表明更新主要落在预训练主导子空间内，存在覆盖预训练知识的风险；$\rho_k \to 0$ 则表明更新与主导方向正交，有利于保留先验知识。
>
> ---
>
> ## 3. OPLoRA：双侧正交投影
>
> **定义 5（双侧正交投影算子）**  
> 构造左、右正交投影矩阵：  
> $$P_L := I - U_k U_k^\top, \quad P_R := I - V_k V_k^\top.$$  
> $P_L$ 与 $P_R$ 分别为到 $\operatorname{span}\{U_k\}^\perp$ 与 $\operatorname{span}\{V_k\}^\perp$ 的正交投影算子。
>
> **定义 6（OPLoRA 更新规则）**  
> OPLoRA 将低秩更新约束于主导奇异子空间的正交补：  
> $$\Delta W_{\text{OP}} := P_L B A P_R,$$  
> 其中 $B, A$ 仍为可训练的低秩矩阵。前向传播修正为  
> $$h = W_0 x + \frac{\alpha}{r} P_L B A P_R \, x.$$
>
> ---
>
> ## 4. 理论保证
>
> **命题 1（左投影的零化性质）**  
> 对任意输入 $x \in \mathbb{R}^{d_{\text{in}}}$，有  
> $$U_k^\top (P_L B A x) = 0, \quad \forall x.$$  
> 即更新输出恒位于 $\operatorname{span}\{U_k\}$ 的正交补内。
>
> **命题 2（主导奇异三元组的精确保持）**  
> 设微调后权重为 $W' = W_0 + \Delta W_{\text{OP}}$。则对 $i = 1, \dots, k$，有  
> $$W' v_i = \sigma_i u_i, \quad (W')^\top u_i = \sigma_i v_i,$$  
> 或等价地  
> $$U_k^\top W' V_k = \Sigma_k = \operatorname{diag}(\sigma_1, \dots, \sigma_k).$$  
> **推论**：前 $k$ 个奇异值及对应的左、右奇异向量在更新后严格保持不变。
>
> ---
>
> ## 5. 实验验证（形式化摘要）
>
> 在 LLaMA-2 7B 与 Qwen2.5 7B 上，于常识推理、数学推理与代码生成三类任务中，OPLoRA 满足：
>
> 1. **低子空间对齐**：$\rho_k$ 在所有基线中最低，表明更新与主导奇异方向几乎无干涉；
> 2. **遗忘抑制**：在 held-out 任务（MathQA、MBPP、RACE 等）上的平均遗忘评分显著优于 LoRA、PiSSA、MiLoRA 与 LoRA-Null；
> 3. **适配性能保持**：在领域内任务上的准确率与最佳基线持平或更优。
>
> ---
>
> ## 6. 总结
>
> OPLoRA 通过双侧正交投影 $\Delta W = (I - U_k U_k^\top) B A (I - V_k V_k^\top)$，将参数高效微调的更新严格限制在预训练主导子空间的正交补内，并以命题 2 保证了前 $k$ 个奇异三元组的精确不变性，从而在数学上提供了防止灾难性遗忘的严格机制。



# 为什么要对W~0~进行分解？

> 对 $W_0$ 进行分解的根本动机在于**识别并隔离预训练知识所占据的线性子空间**，从而为更新施加几何约束。具体而言，可归纳为以下三个层次：
>
> ---
>
> ### 1. 谱分解揭示知识的"能量集中"现象
>
> **定义（能量集中）**  
> 设 $W_0$ 的奇异值谱为 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_m$，则对任意 $k \ll m$，有  
> $$\frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^m \sigma_i^2} \approx 1 - \varepsilon, \quad \varepsilon \ll 1.$$  
> 即预训练权重的大部分"能量"（Frobenius 范数意义下的信息）集中在前 $k$ 个奇异值对应的子空间中。
>
> **解释**  
> 预训练过程通过海量数据优化，使得 $W_0$ 的统计结构呈现低有效秩（low effective rank）。前 $k$ 个奇异向量对 $(u_i, v_i)$ 构成了模型完成预训练任务时**最依赖的变换方向**。这些方向编码了语法、语义、世界知识等核心能力。
>
> ---
>
> ### 2. 主导子空间与灾难性遗忘的因果关联
>
> **命题（更新方向与遗忘的定量关系）**  
> 设微调更新为 $\Delta W$，则预训练知识损失（以 $W_0$ 的 Frobenius 距离度量）可分解为  
> $$\|W_0 - (W_0 + \Delta W)\|_F^2 = \|\Delta W\|_F^2 = \|Q_k \Delta W\|_F^2 + \|(I-Q_k)\Delta W\|_F^2,$$  
> 其中 $Q_k = U_k U_k^\top$ 为到主导左子空间的投影。
>
> **推论**  
> 若 $\Delta W$ 在 $\operatorname{span}\{U_k\}$ 上有非零分量（即 $\|Q_k \Delta W\|_F > 0$），则更新直接修改了预训练知识所在的方向。当 $\|Q_k \Delta W\|_F$ 占 $\|\Delta W\|_F$ 的主要比例时，发生**灾难性遗忘**。
>
> 因此，**只有先通过 SVD 明确 $\operatorname{span}\{U_k\}$ 和 $\operatorname{span}\{V_k\}$ 的几何边界，才能定义"哪些方向不可触碰"**。
>
> ---
>
> ### 3. 为正交投影提供计算基
>
> **定义（正交补空间的显式构造）**  
> SVD 提供了 $\mathbb{R}^{d_{\text{out}}}$ 和 $\mathbb{R}^{d_{\text{in}}}$ 的标准正交基：
> - 左空间：$\{u_1, \dots, u_k\} \cup \{u_{k+1}, \dots, u_{d_{\text{out}}}\}$
> - 右空间：$\{v_1, \dots, v_k\} \cup \{v_{k+1}, \dots, v_{d_{\text{in}}}\}$
>
> **作用**  
> 基于这些基向量，可显式构造正交投影矩阵：
> $$P_L = I - U_k U_k^\top, \quad P_R = I - V_k V_k^\top.$$  
> 没有 SVD，主导子空间没有显式坐标表示，正交补空间无法被直接计算，OPLoRA 的约束 $\Delta W = P_L B A P_R$ 将无法实施。
>
> ---
>
> ### 总结
>
> 对 $W_0$ 进行 SVD 分解并非为了修改 $W_0$ 本身，而是为了：
> 1. **诊断**：通过奇异值谱识别预训练知识的"栖息地"（主导子空间）；
> 2. **隔离**：将更新严格限制在这些栖息地的正交补空间内；
> 3. **保证**：利用投影算子的幂等性与自伴性，从线性代数层面证明预训练核心知识在微调后保持不变。





# 一些理解的问题

我现在理解的内容是ΔW会影响冻结的预训练权重W0,而W0实际上的能量都在一个较小的空间内（UV所决定的），现在的目的就是让ΔW的改变在这个空间外部？（即ΔW影响正交补空间？）

对于数学公式我有个疑问，PL，PR这两个正交投影矩阵不都是零矩阵了吗？

还有左子空间投影里面Uk，代表的就是W0权重比较大的方向对吧？而右子空间投影Vk代表的是某几个权重大的维度？



> 你的理解方向是对的，但有几个关键细节需要澄清。让我逐一回答：
>
> ---
>
> ## 1. 关于"ΔW 影响 W0"的澄清
>
> **W0 是冻结的，ΔW 不会修改 W0 本身。** 微调后的实际权重是：
>
> $$W' = W_0 + \Delta W$$
>
> 你的核心直觉是对的：**让 ΔW 的"作用"避开 W0 的能量集中区域**，即让 ΔW 的输入/输出都落在主导子空间的**正交补**上。
>
> ---
>
> ## 2. PL 和 PR 为什么不是零矩阵？（这是你最核心的疑问）
>
> 这是最关键的一点，你的直觉反了。
>
> **定义回顾：**
>
> - $U_k \in \mathbb{R}^{d_{\text{out}} \times k}$：前 $k$ 个左奇异向量，$k \ll d_{\text{out}}$
> - $V_k \in \mathbb{R}^{d_{\text{in}} \times k}$：前 $k$ 个右奇异向量，$k \ll d_{\text{in}}$
>
> **$U_k U_k^\top$ 是什么？**
>
> 它是到 $\operatorname{span}\{u_1, \dots, u_k\}$ 的**投影矩阵**，秩为 $k$。
>
> - 当 $k \ll d_{\text{out}}$ 时，$U_k U_k^\top$ 是一个**低秩矩阵**（秩只有 $k$），远不是单位矩阵 $I$。
> - 因此 $P_L = I - U_k U_k^\top$ 的秩是 $d_{\text{out}} - k$，是一个**巨大的非零矩阵**，投影到 $U_k$ 张成空间的正交补上。
>
> **直观例子：**
>
> 假设 $d_{\text{out}} = 4096$，$k = 32$：
> - $U_k U_k^\top$ 把向量投影到一个 32 维的超平面上
> - $P_L = I - U_k U_k^\top$ 把向量投影到这个 32 维超平面的**正交补**（一个 4064 维的子空间）
> - $P_L$ 的秩是 4064，几乎"满秩"，绝不是零矩阵
>
> 只有当 $k = d_{\text{out}}$（即保留所有奇异向量）时，$U_k U_k^\top = I$，才有 $P_L = 0$。但那样就失去了低秩适配的意义。
>
> ---
>
> ## 3. Uk 和 Vk 的精确含义
>
> 你的理解基本正确，但需要更精确：
>
> | 符号  | 维度                      | 所在空间                                   | 含义                                                         |
> | ----- | ------------------------- | ------------------------------------------ | ------------------------------------------------------------ |
> | $U_k$ | $d_{\text{out}} \times k$ | **输出空间** $\mathbb{R}^{d_{\text{out}}}$ | W0 的**前 k 个主导输出方向**。即：当输入沿着 $V_k$ 的方向时，W0 的输出主要落在 $U_k$ 的方向上，且增益最大（由奇异值 $\sigma_i$ 决定） |
> | $V_k$ | $d_{\text{in}} \times k$  | **输入空间** $\mathbb{R}^{d_{\text{in}}}$  | W0 的**前 k 个主导输入方向**。即：W0 对这些方向的输入最敏感，变换后能量最大 |
>
> **不是"某几个权重大的维度"，而是"某几个重要的模式方向"。**
>
> - $U_k$ 的每一列 $u_i$ 是输出空间中的一个**单位方向向量**（比如 4096 维空间中的一个方向）
> - $V_k$ 的每一列 $v_i$ 是输入空间中的一个**单位方向向量**
>
> 它们描述的是 W0 的**线性变换结构**：W0 最喜欢把哪些输入方向映射到哪些输出方向，且映射强度由 $\sigma_i$ 决定。
>
> ---
>
> ## 4. 用一个极简例子验证
>
> 设 $W_0 \in \mathbb{R}^{3 \times 3}$，取 $k=1$：
>
> $$W_0 = \begin{bmatrix} 10 & 0 & 0 \\ 0 & 0.1 & 0 \\ 0 & 0 & 0.1 \end{bmatrix}$$
>
> SVD 后：
> - $U_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}$，$V_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}$，$\sigma_1 = 10$
>
> 计算投影：
> $$U_1 U_1^\top = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}$$
>
> $$P_L = I - U_1 U_1^\top = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} \neq 0$$
>
> 显然 $P_L$ 不是零矩阵——它保留了第 2、3 维的信息，只去掉了第 1 维（W0 能量最大的方向）。
>
> ---
>
> ## 总结
>
> | 你的理解              | 修正                                       |
> | --------------------- | ------------------------------------------ |
> | ΔW 影响 W0            | ΔW 与 W0 叠加，W0 本身冻结                 |
> | ΔW 在主导空间外部     | ✅ 正确，ΔW 的输入/输出都被约束在正交补上   |
> | PL、PR 是零矩阵       | ❌ 错误，$k \ll d$ 时它们是高秩非零投影矩阵 |
> | Uk、Vk 是权重大的方向 | ✅ 基本正确，更精确说是"主导输入/输出方向"  |
>
> 核心公式再写一遍：
>
> $$\Delta W_{\text{OP}} = \underbrace{(I - U_k U_k^\top)}_{P_L} \cdot B A \cdot \underbrace{(I - V_k V_k^\top)}_{P_R}$$
>
> - $P_L$：把 BA 的输出"过滤掉"W0 的主导输出方向
> - $P_R$：把输入"过滤掉"W0 的主导输入方向
>
> 这样 ΔW 就只能影响 W0 的"次要方向"，从而保护预训练的核心知识。





## 数学推导

这句话触及了 OPLoRA 最核心的数学洞察。我用**定义 → 反例 → 正例**的结构来讲。

---

## 1. 什么是"奇异三元组"？

对 $W_0$ 做 SVD：$W_0 = U\Sigma V^\top$。第 $i$ 个**奇异三元组** $(u_i, \sigma_i, v_i)$ 不是三个独立的零件，而是一组**耦合的变换关系**：

$$
W_0 v_i = \sigma_i u_i \quad \text{（正向：输入方向 } v_i \text{ 被映射到输出方向 } u_i \text{，增益 } \sigma_i\text{）}
$$

$$
W_0^\top u_i = \sigma_i v_i \quad \text{（反向：输出方向 } u_i \text{ 的梯度/伴随映射回输入方向 } v_i\text{）}
$$

**关键点**：$u_i$、$v_i$、$\sigma_i$ 三者通过上述两个等式**锁死在一起**。你不能只保留其中一个而扔掉另外两个，否则这组变换关系就断裂了。

---

## 2. 为什么单独保留某一个是不行的？

假设微调后权重为 $W' = W_0 + \Delta W$。我们看看只保留单侧会发生什么。

### 情况 A：只保留左奇异向量 $u_i$（输出端）

假设我们要求 $W'^\top u_i = W_0^\top u_i = \sigma_i v_i$，即 $u_i^\top \Delta W = 0$。

**问题**：这并不保证 $W' v_i = \sigma_i u_i$。  
因为 $\Delta W$ 虽然**输出端**避开了 $u_i$，但**输入端**仍可能接受 $v_i$ 方向的分量。即 $\Delta W v_i \neq 0$，导致：

$$
W' v_i = W_0 v_i + \Delta W v_i = \sigma_i u_i + \underbrace{\Delta W v_i}_{\neq 0} \neq \sigma_i u_i
$$

**物理意义**：预训练知识"从 $v_i$ 方向输入，应该从 $u_i$ 方向输出"的映射被破坏了。

---

### 情况 B：只保留右奇异向量 $v_i$（输入端）

假设我们要求 $W' v_i = W_0 v_i = \sigma_i u_i$，即 $\Delta W v_i = 0$。

**问题**：这并不保证 $W'^\top u_i = \sigma_i v_i$。  
因为 $\Delta W$ 虽然**输入端**避开了 $v_i$，但**输出端**仍可能在 $u_i$ 方向有分量。即 $u_i^\top \Delta W \neq 0$，导致：

$$
(W')^\top u_i = W_0^\top u_i + \Delta W^\top u_i = \sigma_i v_i + \underbrace{\Delta W^\top u_i}_{\neq 0} \neq \sigma_i v_i
$$

**物理意义**：反向传播时，$u_i$ 方向的梯度不再正确映射回 $v_i$ 方向。

---

### 情况 C：只保留奇异值 $\sigma_i$

这更荒谬。$\sigma_i$ 是一个标量，它描述的是**沿着特定方向对 $(v_i, u_i)$ 的增益**。如果 $v_i$ 和 $u_i$ 都被改变了，单独保留一个数字 $\sigma_i$ 没有任何几何意义。

---

## 3. OPLoRA 如何同时保留完整三元组？

OPLoRA 的更新：

$$
\Delta W = P_L B A P_R = (I - U_k U_k^\top) B A (I - V_k V_k^\top)
$$

**双侧零化验证**：

**输入端**：对任意 $i \leq k$，
$$
P_R v_i = (I - V_k V_k^\top) v_i = v_i - v_i = 0 \quad \Rightarrow \quad \Delta W v_i = P_L B A \cdot 0 = 0
$$

因此：
$$
W' v_i = W_0 v_i + \Delta W v_i = \sigma_i u_i + 0 = \sigma_i u_i \quad \checkmark
$$

**输出端**：对任意 $i \leq k$，
$$
u_i^\top P_L = u_i^\top (I - U_k U_k^\top) = u_i^\top - u_i^\top = 0 \quad \Rightarrow \quad u_i^\top \Delta W = 0 \cdot B A P_R = 0
$$

因此：
$$
(W')^\top u_i = W_0^\top u_i + \Delta W^\top u_i = \sigma_i v_i + 0 = \sigma_i v_i \quad \checkmark
$$

---

## 4. 形式化总结

**命题（OPLoRA 的奇异三元组保持）**  
设 $W' = W_0 + \Delta W_{\text{OP}}$，则对 $i = 1, \dots, k$：

$$
W' v_i = \sigma_i u_i \quad \text{且} \quad (W')^\top u_i = \sigma_i v_i
$$

即前 $k$ 个奇异三元组 $(u_i, \sigma_i, v_i)$ 在微调后**作为完整关系**被严格保留。

**单侧投影的失效定理**  
若仅施加单侧投影（如仅左投影 $\Delta W = P_L B A$ 或仅右投影 $\Delta W = B A P_R$），则上述两个等式中**必有一个被破坏**，奇异三元组无法完整保持。

---

## 一句话直觉

> 预训练知识不是"存在 $u_i$ 里"或"存在 $v_i$ 里"的，而是**存在 "$v_i$ 进、$u_i$ 出、增益 $\sigma_i$" 这个完整映射关系里**。OPLoRA 用双侧投影同时守住了这个映射的入口和出口，因此能保住整个三元组；只守一侧，知识就会从另一侧漏掉。