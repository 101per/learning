# Note

## 25年NeurIPS:

### World Models Should Prioritize the Unification of Physical and Social 

> ​	当前 World Model 大多分别研究物理动力学或社会行为：前者擅长预测物体、环境和机器人状态，却往往忽略意图、信念、关系和社会规范；后者能够模拟人物行为和多智能体互动，却常缺乏真实物理环境约束。本文认为这种割裂限制了长期预测、因果推理和多智能体决策，因此提出 **ACE Principles**，要求模型处理社会抽象、情境依赖因果以及物理—社会共同演化，并给出 \(WM_{P-S}\) 概念框架，将物理状态与社会状态统一到联合状态转移中。**论文没有训练具体模型，也没有定量实验，其贡献主要是重新定义下一代 World Model 应建模的对象和研究路线。**
>
> Physical-Social State World Model → Bidirectionally Coupled Dynamics → Joint Physical & Social State Prediction → Planning / Simulation / Multi-Agent Decision Making
>
> ![image-20261006173711563](./Note.assets/image-20261006173711563.png)
>
> ![image-20261006173814505](./Note.assets/image-20261006173814505.png)

### DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge

> 现有 VLA 通常直接从视觉和语言预测动作，而引入 world prediction 的方法又常预测完整未来 RGB 图像，既**包含大量无关背景，又缺少显式空间与语义结构**。DreamVLA 因此提出 **Comprehensive World Knowledge Forecasting**：不重建整张未来图像，而预测未来动态区域、深度几何和高层语义，并通过结构化 attention 防止不同知识相互污染，再使用 Diffusion Transformer 根据未来 world embedding 生成动作。实验中其 CALVIN ABC-D 平均连续完成任务数达到 **4.44**，LIBERO 平均成功率 **92.6%**，真实 Franka 机器人平均成功率 **76.7%**，说明“预测任务相关世界知识”比完整像素预测更适合机器人控制。
>
> **是否 Action-conditioned：** **No**
> 通过instruction，和图像，机器人物理状态的输入，直接预测未来动态区域、深度几何和高层语义这三个部分，这三个部分构成w(diffusion的输入)**未来世界的表示**，w经过diffusion transformer(Decoder)直接得到动作预测a
>
> VLA之前有直接预测动作的，有显式预测未来的（预测出未来图像），而现在是预测出w，未来世界的latent表示，通过这个解码出动作。
>
> ![image-20261006182735082](./Note.assets/image-20261006182735082.png)
>
> ![image-20261006214945023](./Note.assets/image-20261006214945023.png)