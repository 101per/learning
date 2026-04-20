# deformable之前理解有错误！！



之前说的是encoder固定K个参考点P，每个参考点加上一个偏移量offset得到采样点。

decoder中是不固定p.

完全是错误的。



在encoder中总共输入HW个像素点，每一个像素点都是一个参考点p,==每一个p都有K个offset,而每一个offset确定一个采样点。==
在decoder中是不固定的有N(这个N是query数量)个参考点p。每个参考点p都有K个offset，同样每个offset都会确定一个采样点。



现在明确一下这个参考点＋offset得到的采样点的作用：

在普通DETR中一个像素会去看所有像素点位。

而在deformable DETR中,一个参考点p，会看他自己的所有采样点位置。

也就是说：

每个参考点 **只采样 K 个位置**

**不是全局 attention**

Encoder**复杂度从 O(H²W²) → O(HW·K)** 这里是encoder特征编码（特征提取）的复杂度



Decoder复杂度由**O(N⋅HW)→O(N⋅K)** 这里是Decoder中cross-attention，就是query和feature map的





> ![image-20260417220720662](C:\Users\URL\AppData\Roaming\Typora\typora-user-images\image-20260417220720662.png)
>
> where *m* indexes the attention head, *k* indexes the sampled keys, and *K* is the total sampled key number (*K* <<*HW*).



下面是DETR复杂度说明

> For the Transformer encoder in DETR, both query and key elements are of pixels in the feature maps. The inputs are of ResNet feature maps (with encoded positional embeddings). Let H and W denote the feature map height and width, respectively. The computational complexity of self-attention is of ==O(H^2^W^2^C)==, which grows quadratically with the spatial size.
>
>  For the Transformer decoder in DETR, the input includes both feature maps from the encoder, and N object queries represented by learnable positional embeddings (e.g., N = 100). There are two types of attention modules in the decoder, namely, cross-attention and self-attention modules. In the cross-attention modules, object queries extract features from the feature maps. The query elements are of the object queries, and key elements are of the output feature maps from the encoder. In it, Nq = N, Nk = H × W and the complexity of the cross-attention is of ==O(HW C^2^ + NHWC)==. The complexity grows linearly with the spatial size of feature maps. In the self-attention modules, object queries interact with each other, so as to capture their relations. The query and key elements are both of the object queries. In it, N~q~ = N~k~ = N, and the complexity of the self-attention module is of ==O(2NC^2^ + N^2^C)==. The complexity is acceptable with moderate number of object queries.



C是输入输出维度，线性投影（QKV）计算就是个C^2^

