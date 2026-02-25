# CSIRO - Image2Biomass Prediction（牧草生物量预测）解决方案报告

------

## 1) 比赛理解

### 1.1 背景与目标

“基于图像、地面实测（ground truth）数据和公开可用的数据集构建模型，以预测牧草生物量（pasture biomass）。农民将使用这些模型来确定何时以及如何放牧其牲畜。”

“牧草生物量——即可用饲草的总量——决定着动物何时能放牧、田块何时需要休牧，以及如何在一季又一季中保持牧场的生产力……剪割称重法准确但缓慢……遥感可以实现大范围监测，但仍需人工验证，且无法按物种区分生物量……本次竞赛挑战你……构建一个模型，利用图像、地面真值测量以及公开可用的数据集来预测牧草生物量。”

------

### 1.2 竞赛数据

**比赛概览
在本次比赛中，你需要利用牧场图像来预测与放牧与饲料管理密切相关的五个关键生物量组成：

- 干燥绿色植被（不含苜蓿）
- 干燥枯死物质
- 干燥苜蓿生物量
- 绿色干物质（GDM）
- 总干生物量

**Files**

- `test.csv`
  - `sample_id`：每个预测行的唯一标识符（每个图像–目标对一行）
  - `image_path`：图像相对路径（例如：`test/IDxxxx.jpg`）
  - `target_name`：本行需要预测的生物量组件名称。取值之一：Dry_Green_g、Dry_Dead_g、Dry_Clover_g、GDM_g、Dry_Total_g。
- `train/`：训练图像（JPEG）
- `test/`：测试图像（评分时隐藏）
- `train.csv`
  - `sample_id`, `image_path`
  - `Sampling_Date` — 采样日期。
  - `State` — 采样所在的澳大利亚州。
  - `Species` — 牧草物种，按生物量排序（以下划线分隔）。
  - `Pre_GSHH_NDVI` — 归一化植被指数（NDVI，GreenSeeker 读数）。
  - `Height_Ave_cm` — 通过落板法（falling plate）测量的牧草平均高度（cm）。
  - `target_name` — 本行的生物量组件名称（Dry_Green_g、Dry_Dead_g、Dry_Clover_g、GDM_g 或 Dry_Total_g）。
  - `target` — 与该图像在对应 target_name 下的真实生物量值（克）。
- `sample_submission.csv`
  - `sample_id`，`target`

------

### 1.3 评估指标

模型性能通过对五个输出维度的 (R^2) 分数进行加权平均来评估：
$$
\text{Final Score} = \sum_{i=1}^{5} (w_{i} \times R^{2}_{i})
$$
权重：

- `Dry_Green_g`: 0.1
- `Dry_Dead_g`: 0.1
- `Dry_Clover_g`: 0.1
- `GDM_g`: 0.2
- `Dry_Total_g`: 0.5

单个目标的 (R^2)：
$$
R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}
$$
$$
SS_{\text{res}} = \sum_{j} (y_j - \hat{y}*j)^2,\quad
SS*{\text{tot}} = \sum_{j} (y_j - \bar{y})^2
$$

------

### 1.4 领域知识入门

这道题本质是“**从顶视图图像估计可用饲草的干物质量**”。输出的 5 个量不是独立的，这意味着：如果模型能先学好更基础的组成（Green/Dead/Clover），再用确定性规则组合出 GDM/Total，通常能减少不一致预测带来的评分损失（尤其是 Total 权重最高）。

**术语词汇表（入门版）**

1. **Biomass（生物量）**：单位面积可用饲草的质量；本赛以“克”为单位输出。
2. **Dry Matter（干物质）**：去除水分后的质量，更能反映真正“可消耗的物质”。
3. **Dry_Green_g**：干燥的绿色植被质量（不含苜蓿）。
4. **Dry_Dead_g**：干燥枯死物质质量（偏黄/褐色的枯草）。
5. **Dry_Clover_g**：干燥苜蓿/三叶草类（豆科）质量。
6. **GDM_g（Green Dry Matter）**：绿色干物质，常近似为 Green + Clover。
7. **Dry_Total_g**：总干生物量，常近似为 GDM + Dead。
8. **NDVI**：归一化植被指数，反映“绿度/植被活性”的遥感/传感指标。
9. **Falling Plate（落板法）**：用落板测草高，间接反映草量，但受密度/物种影响。
10. **(R^2)**：拟合优度，越接近 1 越好（也可能为负）。
11. **Long format / Wide format**：一张图多行目标 vs 一张图一行多目标。
12. **Cross Validation（交叉验证）**：用多折划分估计泛化能力，减少偶然性。

------

## 2) 解决方案解析

### 2.1 方案流程概览

**A. 数据预处理**

- 将 `train.csv` 从 long format pivot 成 wide format：每个 `image_id` 对应 5 个目标列。
- 用 `StratifiedGroupKFold` 做 4 折：
  - **分层**：按 `Dry_Total_g` 做 5 分位分箱（`qcut`）
  - **分组**：按 `image_id` 分组，避免同一图像泄漏到不同折

**B. 模型设计与训练（train.py）**

- Backbone：`vit_huge_plus_patch16_dinov3`（timm），**1.1B 参数级别**的视觉 Transformer。
- 输入处理：每张图像从中间切成左右两半，分别过 backbone 提特征，再进行融合。
- 输出方式：只显式回归 **Green/Dead/Clover**，并用确定性组合得到 **GDM/Total**（保证一致性）。
- 损失：`SmoothL1Loss(beta=5.0)`；优化器：AdamW（backbone/head 不同学习率）；调度：warmup + cosine。

**C. 推理与后处理（inference.py）**

- DINO 4 折权重集成：对测试图像推理后取均值。
- 启发式后处理：
  - `Clover *= 0.8`
  - `Dead` 在阈值区间做比例调整
  - 重新计算 (GDM) 与 (Total)，并 clip 到非负
- SigLIP 分支：提取图像 embedding + “语义概念”相似度特征，用多种 GBDT 回归；与 DINO 做加权融合（除 Clover 只用 DINO）。

------

### 2.2 关键技术点

#### 2.2.1 模型选择与原因

- **ViT Huge（DINOv3）**：适合从复杂自然图像中抽取全局纹理/结构特征，迁移到回归任务。
- **多目标回归 + 权重 (R^2)**：评分对 Total 权重高（0.5），用“基础分量 → 组合指标”的结构，能把学习重点放在可分辨的视觉信号上，同时让高权重目标受益于一致性约束。

#### 2.2.2 特征工程/数据增强（图像侧）

- 训练增强（Albumentations）：翻转、90°旋转、轻量仿射、ColorJitter、Normalize。
- **左右切分输入**：把宽图切成两个视角，等价于增加有效视野覆盖，并降低单次输入的长宽比压力。

#### 2.2.3 训练策略

- **损失**：`SmoothL1Loss(beta=5.0)`（Huber），兼顾稳定性与对离群点的鲁棒性。
- **差分学习率**：backbone (1\times 10^{-5})，head (5\times 10^{-4})，减少大模型微调时的灾难性遗忘。
- **学习率调度**：warmup + cosine（按 step 更新）。
- **混合精度**：`autocast + GradScaler`。
- **早停**：patience=15，避免长训练过拟合。

#### 2.2.4 创新点与为何有效（机制解释）

1. **“质量守恒式”多任务头（Hard Constraint）**

   - 模型只预测 Green/Dead/Clover，随后：
     $$
     \widehat{GDM}=\widehat{Green}+\widehat{Clover},\quad
     \widehat{Total}=\widehat{GDM}+\widehat{Dead}
     $$

   - 机制：减少输出空间自由度，避免出现“Total 与分量不一致”的不可解释预测，从而在 Total 权重很高时更稳健。

2. **LocalMambaBlock 做左右特征融合**

   - 将左右 backbone 特征拼接后，用带 gating 的 1D depthwise conv 处理序列特征。
   - 机制：在不引入很重注意力成本的情况下，让左右半幅的局部模式发生交互，提升对“密度/枯黄比例/局部纹理”的表达。

3. **SigLIP + 语义概念特征 + GBDT（推理侧的稳健补偿）**

   - 用概念 prompt（如 bare/sparse/dense/green/dead/clover）计算相似度，形成可解释的语义特征（例如 `ratio_greenness`）。
   - 机制：当纯回归模型对极端场景泛化不稳时，树模型可利用“语义分段 + 非线性组合”做补偿；再与 DINO 加权融合降低方差。

------

### 2.3 代码解析

#### 2.3.1 数据读取与折划分（train.py）

- `load_train_data()`
  - 输入：`train.csv`（long format）
  - 处理：提取 `image_id`，pivot 成 wide；用 `Dry_Total_g` 分箱后做 `StratifiedGroupKFold`
  - 输出：每张图一行，含 5 目标列与 `fold`

#### 2.3.2 Dataset：左右切分（train.py / inference.py）

- `BiomassDataset.__getitem__`（训练）
  - `cv2.imread` 读取图像 → 从中间切成 left/right → 各自做增强
- `TestDataset.__getitem__`（推理）
  - PIL 读取 → 从左右各裁一个正方形（`left: (0,0,h,h)`；`right: (w-h,0,w,h)`）→ resize/normalize

#### 2.3.3 模型结构：DINO Backbone + 融合 + 三头输出（两份脚本一致）

- `BiomassModel`
  - Backbone：timm 创建 `num_classes=0` 的特征提取器
  - Fusion：2 个 `LocalMambaBlock`
  - Heads：三个回归头 + Softplus 保证非负
  - 组合输出：`gdm = green + clover`；`total = gdm + dead`

（核心结构的“最小摘录”，避免长代码）

```python
green = head_green(x_pool); dead = head_dead(x_pool); clover = head_clover(x_pool)
gdm = green + clover
total = gdm + dead
```

#### 2.3.4 训练循环（train.py）

- `build_optimizer`：AdamW，backbone/head 两组参数不同 LR
- `build_scheduler`：warmup + cosine
- `train_one_epoch`：AMP + grad clip
- `validate`：计算 weighted (R^2) 与每目标 (R^2)
- `train_fold`：保存每折 best checkpoint，patience 早停

#### 2.3.5 推理与集成（inference.py）

- **DINO**：
  - 加载 `fold{0..3}_best.pth` → 各折预测 → 平均
  - 后处理：Clover * 0.8；Dead 分段缩放；重算 GDM/Total；clip
- **SigLIP**：
  - `compute_siglip_embeddings`：切 patch → `get_image_features` → patch 均值
  - `generate_semantic_features`：图像 embedding 与概念文本向量点积相似度
  - `SupervisedEmbeddingEngine`：StandardScaler + PCA(80%) + PLS(8) + GMM(6) + 语义特征拼接
  - `train_gbdt_cv`：对每个目标训练 GBDT（HistGB/GB/Cat/LGBM），并做模型平均
- **最终融合**：
  - 除 Clover 外：`0.7 * DINO + 0.3 * SigLIP`
  - Clover：仅用 DINO
  - 最后做“质量平衡”重算与非负裁剪



------

## 3) 简历项目模板（可直接粘贴）

**CSIRO - Image2Biomass Prediction（Kaggle，2025.10–2026.01）｜LB 0.64（weighted (R^2)）**

- 背景/挑战：基于牧场顶视图图像预测 5 个生物量组分，评分为多目标加权 (R^2)，其中 **Dry_Total_g 权重高达 0.5**，且目标间存在强耦合关系。
- 方案亮点：
  - 使用 **DINOv3 ViT-Huge（1.1B）**进行回归微调，并将宽图 **左右切分**形成双视角输入，通过 **LocalMambaBlock**实现低成本特征交互融合。
  - 引入“质量守恒式”结构化输出，显式预测 Green/Dead/Clover，并通过强化多目标一致性以提升高权重目标稳定性。
  - 推理阶段实现 **4 折模型集成 + 非负约束 + 质量平衡重算**，并集成 **SigLIP 图像嵌入 + 语义概念相似度特征 + GBDT** 作为鲁棒性补偿。
- 技术栈：PyTorch / timm / Albumentations / Transformers（SigLIP）/ scikit-learn（PCA, PLS, GMM）/ LightGBM / CatBoost / AMP 混合精度训练。