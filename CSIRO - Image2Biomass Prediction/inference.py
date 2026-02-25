# %% [markdown]
# # CSIRO BIOMASS - INFERENCE (VIT_HUGE_PLUS + SIGLIP ENSEMBLE)
# # CSIRO 生物量预测 - 推理代码 (VIT_HUGE_PLUS + SIGLIP 集成模型)
# 本代码用于 CSIRO 牧草生物量预测比赛的推理阶段。
# 比赛目标：利用牧场图像预测 5 个关键生物量指标（Green, Dead, Clover, GDM, Total）。
# 方法概述：
# 1. DINOv2 (ViT Huge) 模型：基于图像块（Patch）的深度学习模型，用于直接回归预测。
# 2. SigLIP 模型：提取图像语义特征，结合 GBDT (梯度提升树) 进行回归预测。
# 3. 集成 (Ensemble)：将两种模型的预测结果加权融合，以获得更稳健的结果。

# %%


# %%
import os
import gc
import random
import warnings
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path
from dataclasses import dataclass

# 引入 sklearn 和 boosting 库，用于 SigLIP 部分的特征处理和回归
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer

import timm

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# configs
# 配置类：定义路径、模型参数、目标变量等
@dataclass
class Config:
    # Competition data
    # 比赛数据路径
    DATA_PATH: Path = Path("/kaggle/input/csiro-biomass")
    # 数据划分文件路径
    SPLIT_PATH: Path = Path("/kaggle/input/csiro-datasplit/csiro_data_split.csv")
    
    # 训练好的 DINO 模型权重存放路径
    MODELS_DIR: Path = Path("/kaggle/input/dino-huge-retrain-checkpoints-zul/models_trained")
    
    # SigLIP
    # SigLIP 预训练模型路径
    SIGLIP_PATH: str = "/kaggle/input/google-siglip-so400m-patch14-384/transformers/default/1"
    
    #model name
    # DINO 模型使用的 Backbone 名称
    MODEL_NAME: str = "vit_huge_plus_patch16_dinov3.lvd1689m"
    
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE: int = 512
    BATCH_SIZE: int = 2  
    N_FOLDS: int = 4 # 使用 4 折交叉验证的模型
    DROPOUT: float = 0.2
    
    # Targets
    # 比赛要求的 5 个预测目标：
    # Dry_Green_g: 干燥绿色植被（不含苜蓿）
    # Dry_Dead_g: 干燥枯死物质
    # Dry_Clover_g: 干燥苜蓿生物量
    # GDM_g: 绿色干物质 (Green + Clover)
    # Dry_Total_g: 总干生物量 (GDM + Dead)
    TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    # 模型输出的目标顺序（可能与 TARGETS 列表顺序不同，需注意对应）
    TARGET_NAMES = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    
    # 目标变量的最大值，用于归一化或反归一化
    TARGET_MAX = {
        "Dry_Clover_g": 71.7865,
        "Dry_Dead_g": 83.8407,
        "Dry_Green_g": 157.9836,
        "Dry_Total_g": 185.70,
        "GDM_g": 157.9836,
    }
    
    # Ensemble weights
    # 集成权重：DINO 模型占 70%，SigLIP 模型占 30%
    W_DINO: float = 0.70
    W_SIGLIP: float = 0.30

cfg = Config()

# 设置随机种子，保证结果可复现
def seed_everything(seed=cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()

print(f"Device: {cfg.DEVICE}")
print(f"Model: {cfg.MODEL_NAME}")
print(f"Models Dir: {cfg.MODELS_DIR}")

# 自定义 Mamba 模块，用于特征融合
# Mamba 是一种高效的状态空间模型 (SSM)，这里用于处理图像特征序列
class LocalMambaBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # 深度可分离卷积
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(x)
        # Gating 机制
        g = torch.sigmoid(self.gate(x))
        x = x * g
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.drop(x)
        return shortcut + x

# 生物量预测主模型 (DINO Backbone)
class BiomassModel(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = False):
        super().__init__()
        # 使用 timm 加载预训练的 ViT 模型作为 Backbone
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=''
        )
        nf = self.backbone.num_features
        
        # 融合层：使用 LocalMambaBlock 融合左右两张图像的特征（或者同一张图切分的左右两部分）
        self.fusion = nn.Sequential(
            LocalMambaBlock(nf, kernel_size=5, dropout=cfg.DROPOUT),
            LocalMambaBlock(nf, kernel_size=5, dropout=cfg.DROPOUT)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # 定义三个独立的预测头 (Head)，分别预测 Green, Dead, Clover
        self.head_green = nn.Sequential(
            nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(nf // 2, 1), nn.Softplus() # Softplus 保证输出非负
        )
        self.head_dead = nn.Sequential(
            nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(nf // 2, 1), nn.Softplus()
        )
        self.head_clover = nn.Sequential(
            nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(nf // 2, 1), nn.Softplus()
        )

    def forward(self, left, right):
        # 提取左右图像的特征
        x_l = self.backbone(left)
        x_r = self.backbone(right)
        # 拼接特征
        x_cat = torch.cat([x_l, x_r], dim=1)
        # 融合特征
        x_fused = self.fusion(x_cat)
        # 池化
        x_pool = self.pool(x_fused.transpose(1, 2)).flatten(1)
        
        # 分别预测三个基础分量
        green = self.head_green(x_pool)
        dead = self.head_dead(x_pool)
        clover = self.head_clover(x_pool)
        
        # 根据物理关系计算复合指标
        gdm = green + clover      # 绿色干物质 = 绿草 + 苜蓿
        total = gdm + dead        # 总生物量 = 绿色干物质 + 枯死物质
        
        # 返回所有 5 个预测值
        return torch.cat([green, dead, clover, gdm, total], dim=1)

# dino ds
# DINO 模型的数据集类
class TestDataset(Dataset):
    def __init__(self, df, image_root, img_size=512):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.img_size = img_size
        
        # ImageNet 归一化参数
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_root / row["image_path"]
        
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        
        # Split in middle
        # 将图像从中间切分为左右两部分（这是一种常见的数据增强或处理长宽比不一致图像的方法）
        # 这里假设图像是宽图，切成两个正方形
        left = img.crop((0, 0, h, h))
        right = img.crop((w - h, 0, w, h))
        
        # Resize and normalize
        # 调整大小并归一化
        left = left.resize((self.img_size, self.img_size))
        right = right.resize((self.img_size, self.img_size))
        
        left = np.array(left).astype(np.float32) / 255.0
        right = np.array(right).astype(np.float32) / 255.0
        
        left = (left - self.mean) / self.std
        right = (right - self.mean) / self.std
        
        # 转换为 PyTorch Tensor (C, H, W)
        left = torch.from_numpy(left.transpose(2, 0, 1)).float()
        right = torch.from_numpy(right.transpose(2, 0, 1)).float()
        
        return left, right, row.to_dict()

# DataLoader 的 collate 函数
def collate_fn(batch):
    lefts = torch.stack([b[0] for b in batch])
    rights = torch.stack([b[1] for b in batch])
    infos = [b[2] for b in batch]
    return lefts, rights, infos

# DINO 推理函数
@torch.no_grad()
def predict_dino(model, loader, device):
    model.eval()
    preds_all = []
    
    for lefts, rights, _ in tqdm(loader, desc="DINO Inference"):
        lefts = lefts.to(device)
        rights = rights.to(device)
        
        # 使用混合精度推理加速
        with autocast():
            pred = model(lefts, rights)
        
        preds_all.append(pred.cpu().numpy())
    
    return np.vstack(preds_all)

#siglip()
# SigLIP 部分：图像切片辅助函数
def split_image(image, patch_size=520, overlap=16):
    h, w, c = image.shape
    stride = patch_size - overlap
    patches = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y2 = min(y + patch_size, h)
            x2 = min(x + patch_size, w)
            y1 = max(0, y2 - patch_size)
            x1 = max(0, x2 - patch_size)
            patches.append(image[y1:y2, x1:x2, :])
    return patches

# 计算 SigLIP 图像嵌入 (Embeddings)
def compute_siglip_embeddings(model_path, df, img_dir):
    print(f"Computing SigLIP embeddings for {len(df)} images...")
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(cfg.DEVICE)
    processor = AutoImageProcessor.from_pretrained(model_path)
    
    embeddings = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            img_path = img_dir / row['image_path']
            # 使用 OpenCV 读取图像
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 切分图像为 Patches
            patches = split_image(img)
            images = [Image.fromarray(p) for p in patches]
            
            # 提取特征
            inputs = processor(images=images, return_tensors="pt").to(cfg.DEVICE)
            with torch.no_grad():
                features = model.get_image_features(**inputs)
            
            # 对所有 Patches 的特征取平均，得到整图的 Embedding
            embeddings.append(features.mean(dim=0).cpu().numpy())
        except Exception as e:
            print(f"Error: {e}")
            embeddings.append(np.zeros(1152)) # 异常处理，返回零向量
    
    del model
    torch.cuda.empty_cache()
    return np.stack(embeddings)

# 生成语义特征 (Semantic Features)
# 利用 SigLIP (类似 CLIP) 的图文对齐能力，计算图像与预定义文本概念的相似度
def generate_semantic_features(embeddings, model_path):
    model = AutoModel.from_pretrained(model_path).to(cfg.DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 预定义的牧场相关概念文本
    concepts = {
        "bare": ["bare soil", "dirt ground", "sparse vegetation", "exposed earth"], # 裸土
        "sparse": ["low density pasture", "thin grass", "short clipped grass"],     # 稀疏牧草
        "medium": ["average pasture cover", "medium height grass", "grazed pasture"], # 中等覆盖
        "dense": ["dense tall pasture", "thick grassy volume", "high biomass"],     # 茂密牧草
        "green": ["lush green vibrant pasture", "photosynthesizing leaves", "fresh growth"], # 绿色植被
        "dead": ["dry brown dead grass", "yellow straw", "senesced material"],      # 枯死植被
        "clover": ["white clover", "trifolium repens", "broadleaf legume"],         # 三叶草/苜蓿
        "grass": ["ryegrass", "blade-like leaves", "fescue", "grassy sward"]        # 禾本科草
    }
    
    concept_vectors = {}
    with torch.no_grad():
        for name, prompts in concepts.items():
            inputs = tokenizer(prompts, padding="max_length", return_tensors="pt").to(cfg.DEVICE)
            emb = model.get_text_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            # 对每个概念的多个描述取平均，得到该概念的文本向量
            concept_vectors[name] = emb.mean(dim=0, keepdim=True)
    
    img_tensor = torch.tensor(embeddings, dtype=torch.float32).to(cfg.DEVICE)
    img_tensor = img_tensor / img_tensor.norm(p=2, dim=-1, keepdim=True)
    
    # 计算图像 Embedding 与各个概念向量的点积（相似度）
    scores = {}
    for name, vec in concept_vectors.items():
        scores[name] = torch.matmul(img_tensor, vec.T).cpu().numpy().flatten()
    
    # 构造特征 DataFrame
    df_scores = pd.DataFrame(scores)
    # 计算衍生特征：绿度比例、苜蓿比例
    df_scores['ratio_greenness'] = df_scores['green'] / (df_scores['green'] + df_scores['dead'] + 1e-6)
    df_scores['ratio_clover'] = df_scores['clover'] / (df_scores['clover'] + df_scores['grass'] + 1e-6)
    
    del model
    torch.cuda.empty_cache()
    return df_scores.values

# 监督式嵌入引擎
# 对提取的特征进行进一步降维和转换，用于 GBDT 训练
class SupervisedEmbeddingEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.80, random_state=42) # PCA 保留 80% 方差
        self.pls = PLSRegression(n_components=8, scale=False) # 偏最小二乘回归
        self.gmm = GaussianMixture(n_components=6, covariance_type='diag', random_state=42) # 高斯混合模型
        self.pls_fitted_ = False

    def fit(self, X, y=None, X_semantic=None):
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)
        self.gmm.fit(X_scaled)
        if y is not None:
            self.pls.fit(X_scaled, y)
            self.pls_fitted_ = True
        return self

    def transform(self, X, X_semantic=None):
        X_scaled = self.scaler.transform(X)
        feats = [self.pca.transform(X_scaled)]
        if self.pls_fitted_:
            feats.append(self.pls.transform(X_scaled))
        feats.append(self.gmm.predict_proba(X_scaled)) # 添加 GMM 概率特征
        if X_semantic is not None:
            # 标准化语义特征
            sem_norm = (X_semantic - np.mean(X_semantic, axis=0)) / (np.std(X_semantic, axis=0) + 1e-6)
            feats.append(sem_norm)
        return np.hstack(feats)

# GBDT 交叉验证训练函数
def train_gbdt_cv(model_cls, params, train_data, test_data, sem_tr, sem_te, emb_cols):
    target_max_arr = np.array([cfg.TARGET_MAX[t] for t in cfg.TARGET_NAMES])
    y_pred_test = np.zeros([len(test_data), len(cfg.TARGET_NAMES)])
    n_splits = int(train_data['fold'].nunique())
    
    X_train = train_data[emb_cols].values.astype(np.float32)
    X_test = test_data[emb_cols].values.astype(np.float32)
    y_train = train_data[cfg.TARGET_NAMES].values.astype(np.float32)
    
    # 逐折训练
    for fold in range(n_splits):
        train_mask = train_data['fold'] != fold
        X_tr = X_train[train_mask]
        # 归一化目标值
        y_tr = y_train[train_mask] / target_max_arr
        sem_tr_fold = sem_tr[train_mask]
        
        # 特征工程
        eng = SupervisedEmbeddingEngine()
        eng.fit(X_tr, y=y_tr, X_semantic=sem_tr_fold)
        
        x_tr_eng = eng.transform(X_tr, X_semantic=sem_tr_fold)
        x_te_eng = eng.transform(X_test, X_semantic=sem_te)
        
        # 对每个目标单独训练模型
        for k, target in enumerate(cfg.TARGET_NAMES):
            if target == 'Dry_Clover_g': # SigLIP 部分跳过 Clover 预测（可能因为 DINO 效果更好）
                continue
            model = model_cls(**params)
            model.fit(x_tr_eng, y_tr[:, k])
            # 预测并反归一化
            y_pred_test[:, k] += model.predict(x_te_eng) * target_max_arr[k]
    
    return y_pred_test / n_splits # 平均多折结果


print("="*60)
print("CSIRO BIOMASS INFERENCE - VIT_HUGE_PLUS + SIGLIP")
print("="*60)

# Load test data
# 1. 加载测试数据
print("\n[1/6] Loading test data...")
test_df_raw = pd.read_csv(cfg.DATA_PATH / 'test.csv')
# 去重，得到唯一的图像列表（test.csv 是 long format，每个图有多行）
test_wide = test_df_raw[["image_path"]].drop_duplicates().reset_index(drop=True)
print(f"Test images: {len(test_wide)}")


print("\n[2/6] Running DINO HUGE inference...")
# 2. 运行 DINO 模型推理

test_dataset = TestDataset(test_wide, cfg.DATA_PATH, cfg.IMG_SIZE)
test_loader = DataLoader(
    test_dataset, 
    batch_size=cfg.BATCH_SIZE, 
    shuffle=False, 
    num_workers=0,
    collate_fn=collate_fn
)

all_fold_preds = []

# 对每个折叠的模型进行推理
for fold in range(cfg.N_FOLDS):
    model_path = cfg.MODELS_DIR / f"fold{fold}_best.pth"
    if not model_path.exists():
        print(f"  Fold {fold} not found, skipping...")
        continue
    
    print(f"  Loading fold {fold}...")
    model = BiomassModel(cfg.MODEL_NAME, pretrained=False).to(cfg.DEVICE)
    state_dict = torch.load(model_path, map_location=cfg.DEVICE)
    
    # Handle DataParallel
    # 处理 DataParallel 保存的权重键名
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    
    fold_preds = predict_dino(model, test_loader, cfg.DEVICE)
    all_fold_preds.append(fold_preds)
    
    del model, state_dict
    gc.collect()
    torch.cuda.empty_cache()

# Ensemble folds
# 平均所有折叠的预测结果
dino_preds = np.mean(all_fold_preds, axis=0)
print(f"DINO predictions shape: {dino_preds.shape}")

# Post-processing (from original 0.73 solution)
# DINO 后处理策略
# Clover * 0.8, Dead adjustments
dino_df = test_wide.copy()
dino_df['Dry_Green_g'] = dino_preds[:, 0]
dino_df['Dry_Dead_g'] = dino_preds[:, 1]
dino_df['Dry_Clover_g'] = dino_preds[:, 2] * 0.8  # 后处理：Clover 预测值乘以 0.8
# 注意：DINO 模型直接输出了 5 个值，这里取前 3 个基础分量，后面重新计算 GDM 和 Total

# Dead adjustment from 0.73 solution
# 对 Dead 分量进行启发式调整
for i in range(len(dino_df)):
    if dino_df.loc[i, 'Dry_Dead_g'] > 20:
        dino_df.loc[i, 'Dry_Dead_g'] *= 1.1
    elif dino_df.loc[i, 'Dry_Dead_g'] < 10:
        dino_df.loc[i, 'Dry_Dead_g'] *= 0.9

# 重新计算组合指标
dino_df['GDM_g'] = dino_df['Dry_Green_g'] + dino_df['Dry_Clover_g']
dino_df['Dry_Total_g'] = dino_df['GDM_g'] + dino_df['Dry_Dead_g']

# =============================================================================
# SIGLIP INFERENCE
# =============================================================================
# 3. 运行 SigLIP 模型推理
print("\n[3/6] Running SigLIP inference...")

train_split = pd.read_csv(cfg.SPLIT_PATH)
cols_keep = [c for c in train_split.columns if not c.startswith('emb')]
train_split = train_split[cols_keep]

# 修正训练集图片路径
if not str(train_split['image_path'].iloc[0]).startswith('/'):
    train_split['image_path'] = train_split['image_path'].apply(
        lambda p: str(cfg.DATA_PATH / 'train' / os.path.basename(p))
    )

# Prepare test paths
test_siglip = test_wide.copy()
test_siglip['image_path'] = test_siglip['image_path'].apply(lambda p: str(cfg.DATA_PATH / p))

# Embeddings
# 计算训练集和测试集的 Embeddings
print("  Computing train embeddings...")
train_emb = compute_siglip_embeddings(cfg.SIGLIP_PATH, train_split, cfg.DATA_PATH)
print("  Computing test embeddings...")
test_emb = compute_siglip_embeddings(cfg.SIGLIP_PATH, test_siglip, cfg.DATA_PATH)

emb_cols = [f"emb{i}" for i in range(train_emb.shape[1])]
train_feat = pd.concat([train_split, pd.DataFrame(train_emb, columns=emb_cols)], axis=1)
test_feat = pd.concat([test_siglip.reset_index(drop=True), pd.DataFrame(test_emb, columns=emb_cols)], axis=1)

# Semantic features
# 生成语义特征
print("  Generating semantic features...")
all_emb = np.vstack([train_emb, test_emb])
all_sem = generate_semantic_features(all_emb, cfg.SIGLIP_PATH)
sem_train = all_sem[:len(train_split)]
sem_test = all_sem[len(train_split):]

# GBDT
# 4. 训练 GBDT 模型并预测
print("\n[4/6] Training GBDT models...")
params_hist = {'max_iter': 300, 'learning_rate': 0.05, 'max_depth': 5, 'random_state': 42}
params_gb = {'n_estimators': 1354, 'learning_rate': 0.01, 'max_depth': 3, 'random_state': 42}
params_cat = {'iterations': 1900, 'learning_rate': 0.045, 'depth': 4, 'verbose': 0, 'random_state': 42, 'allow_writing_files': False}
params_lgbm = {'n_estimators': 807, 'learning_rate': 0.014, 'num_leaves': 48, 'verbose': -1, 'random_state': 42}

print("  HistGB...")
pred_hist = train_gbdt_cv(HistGradientBoostingRegressor, params_hist, train_feat, test_feat, sem_train, sem_test, emb_cols)
print("  GB...")
pred_gb = train_gbdt_cv(GradientBoostingRegressor, params_gb, train_feat, test_feat, sem_train, sem_test, emb_cols)
print("  CatBoost...")
pred_cat = train_gbdt_cv(CatBoostRegressor, params_cat, train_feat, test_feat, sem_train, sem_test, emb_cols)
print("  LightGBM...")
pred_lgbm = train_gbdt_cv(LGBMRegressor, params_lgbm, train_feat, test_feat, sem_train, sem_test, emb_cols)

# 平均 4 个 GBDT 模型的预测结果
siglip_pred = (pred_hist + pred_gb + pred_cat + pred_lgbm) / 4.0

siglip_df = test_siglip.copy()
siglip_df[cfg.TARGET_NAMES] = siglip_pred
# SigLIP 不预测 Clover (设置为 0，最后集成时完全依赖 DINO)
siglip_df['Dry_Clover_g'] = 0.0
siglip_df['GDM_g'] = siglip_df['Dry_Green_g']
siglip_df['Dry_Total_g'] = siglip_df['GDM_g'] + siglip_df['Dry_Dead_g']


# ENSEMBLE
# 5. 模型集成

print("\n[5/6] Creating ensemble...")
print(f"Weights: DINO={cfg.W_DINO}, SigLIP={cfg.W_SIGLIP}")

ALL_TARGETS = ['Dry_Green_g', 'Dry_Clover_g', 'Dry_Dead_g', 'GDM_g', 'Dry_Total_g']

final_df = test_wide.copy()

for target in ALL_TARGETS:
    if target == 'Dry_Clover_g':
        # Clover 仅使用 DINO 的预测
        final_df[target] = dino_df[target]  
    else:
        # 其他目标使用加权平均
        final_df[target] = dino_df[target] * cfg.W_DINO + siglip_df[target] * cfg.W_SIGLIP

# Mass balance
# 质量平衡检查与修正
final_df['Dry_Clover_g'] = final_df['Dry_Clover_g'].clip(lower=0.0)
final_df['GDM_g'] = final_df['Dry_Green_g'] + final_df['Dry_Clover_g']
final_df['Dry_Total_g'] = final_df['GDM_g'] + final_df['Dry_Dead_g']

# Clip all to non-negative
# 保证所有预测值为非负
for col in ALL_TARGETS:
    final_df[col] = final_df[col].clip(lower=0.0)

print("\n[6/6] Creating submission...")
# 6. 生成提交文件

submission_rows = []
for _, row in final_df.iterrows():
    # 提取 Image ID
    image_id = os.path.basename(row['image_path']).replace('.jpg', '')
    for target in cfg.TARGETS:
        # 构建 sample_id: ImageID__TargetName
        submission_rows.append({
            'sample_id': f"{image_id}__{target}",
            'target': row[target]
        })

submission = pd.DataFrame(submission_rows)
submission.to_csv('submission.csv', index=False)

print("\n" + "="*60)
print("COMPLETE!")
print("="*60)
print(f"\nSubmission saved: submission.csv")
print(submission.head(10))
print(f"\nStats:\n{submission['target'].describe()}")

# %%


