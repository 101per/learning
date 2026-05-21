今天你实际上已经把整个现代 3D Vision 的主线串起来了，从传统几何 → Learning-based MVS → Neural Rendering → Spatial Intelligence。下面我给你做一个高密度总结。

3D 重建的核心问题是：如何从多张二维图像恢复三维世界。最早的方法属于传统几何范式，核心是 SfM（Structure from Motion）+ MVS（Multi-View Stereo）。SfM 负责恢复相机位姿与稀疏点云；MVS 在已知相机位姿基础上恢复稠密几何。典型 pipeline：

```text
Images → SfM/COLMAP → Camera Poses → MVS → Dense Point Cloud/Mesh
```

MVS 的本质是“多视图对应关系搜索”。核心思想：对于参考图像中的一个像素，枚举多个深度假设，如果深度正确，那么其它视图 warp 到参考视图后会对齐；如果深度错误，则会错位。因此：

```text
cost(x,y,d)
```

表示像素 `(x,y)` 在深度 `d` 下的匹配代价。所有 `(x,y,d)` 组成：

```text
H × W × D
```

的三维张量，这就是 Cost Volume。它本质上是“显式 correspondence search 的张量化表示”。

MVSNet 是第一个真正成功的 Learning-based MVS。核心流程：

```text
多视图图像 → CNN特征 → Homography Warping → Cost Volume → 3D CNN → Soft Argmin → Depth
```

关键点：

1. 对每个 depth hypothesis，把 source features warp 到 reference view。
2. 正确深度会导致多视图 feature 对齐。
3. MVSNet 使用 variance 构建 cost volume：variance 小表示多视图一致。
4. 3D CNN 在 `(x,y,d)` 上 regularize volume，学习真实表面在 volume 中的结构。
5. Soft Argmin：
   depth=\sum_d d\cdot P(d)
   实现可导深度回归。

MVSNet 最大问题是显存爆炸，因为：

```text
Volume Size = H×W×D×C
```

再加 3D CNN，计算量极大。

CasMVSNet（Cascade MVSNet）解决了这个问题。核心思想是 coarse-to-fine：

```text
低分辨率全局搜索 → 中分辨率局部 refinement → 高分辨率精细优化
```

它不再一次性搜索全部深度，而是逐层缩小搜索范围，因此显存大幅下降。这奠定了后续大量 MVS 方法的基础。

之后，3D Vision 出现了范式变化：从“显式几何恢复”转向“神经场表示”。NeRF 是这个转折点。

MVS 的核心对象是：

```text
depth / correspondence
```

而 NeRF 的核心对象是：

```text
ray
```

NeRF（Neural Radiance Fields）学习的是一个连续场景函数：
F(x,y,z,\theta,\phi)\rightarrow(c,\sigma)
输入：

- 空间位置 `(x,y,z)`
- viewing direction `(θ,φ)`
  输出：
- color `c`
- density `σ`

这里的 density（σ）可以理解为“这里是不是物体”。空气区域：

```text
σ≈0
```

物体表面：

```text
σ 很大
```

NeRF 不使用 mesh、point cloud 或 depth supervision，而是只用：

```text
多视图RGB + Camera Poses
```

监督来自真实照片像素颜色。训练过程：

1. 对每个像素发射 ray：
   r(t)=o+td
2. 沿 ray 采样大量点。
3. MLP 预测每个点的 `(color,density)`。
4. 使用 Volume Rendering 合成最终像素颜色。
5. 与真实 RGB 做 loss。

核心 loss：
L=|C_{pred}-C_{gt}|^2

NeRF 最关键的地方：
它从来没人告诉“哪里是物体”，而是：

> 为了让所有视角都渲染正确，网络被迫学习一致的三维结构。
> 因此 NeRF 本质上是：

```text
多视图一致性 + 可微渲染
```

逼出了隐式几何。

NeRF 与 MVS 的本质区别：
MVS：

```text
找对应关系 → 三角化 → depth
```

NeRF：

```text
学习连续空间 → volume rendering → 图像
```

MVS 是 geometry-first；NeRF 是 rendering-first。

NeRF 的重大问题是太慢，因为：

```text
每像素 → 一条ray
每ray → 数十~数百采样点
每采样点 → MLP
```

于是出现大量加速工作：

- Instant-NGP：hash grid encoding
- Plenoxels：直接 voxel 表示
- 3D Gaussian Splatting：显式高斯 primitives

3D Gaussian Splatting 可以理解为：

```text
NeRF质量 + 实时渲染速度
```

它不用 MLP，而是直接用大量 3D Gaussian 表示场景。

再后来，3D Vision 又进入 Spatial Intelligence 阶段。代表方法：

- DUSt3R
- MASt3R
- VGGT

DUSt3R 已经不再显式：

- cost volume
- stereo matching
- triangulation

而是 Transformer 直接：

```text
image pair → aligned pointmaps
```

即“直接理解空间”。

整个发展主线：

```text
传统几何（SfM/MVS）
→ Learning-based MVS（MVSNet/CasMVSNet）
→ Neural Rendering（NeRF）
→ Explicit Neural Representation（Gaussian Splatting）
→ Spatial Intelligence（DUSt3R/MASt3R）
```

你今天真正学到的最核心思想有三个：

1. Cost Volume 本质：

> 把 correspondence search 显式张量化。

1. NeRF 本质：

> 学习一个可微分的连续三维场景表示，并通过 volume rendering 生成图像。

1. 现代 3D Vision 的范式迁移：

```text
显式几何恢复
→ 神经场表示
→ 通用空间理解
```

如果继续深入，建议学习顺序：

```text
相机模型
→ Epipolar Geometry
→ Homography
→ Triangulation
→ MVSNet/CasMVSNet
→ Volume Rendering
→ NeRF
→ Gaussian Splatting
→ DUSt3R/MASt3R
```

这条路线基本就是当前 3D Vision 与 Spatial Intelligence 的核心知识树。