"""
CSIRO 牧草生物量预测：DINOv3 ViT-Huge 微调训练脚本。
输入：左右半幅图像；输出：5 个生物量分量（含派生的 GDM/Total）。
训练：4 折交叉验证 + 早停，AMP 混合精度，分组分层划分。
"""

# %%
import os
import gc
import math
import random
import warnings
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm.auto import tqdm
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedGroupKFold
from PIL import Image
import timm

warnings.filterwarnings('ignore')  # 屏蔽常见告警，避免输出干扰日志

torch.backends.cuda.matmul.allow_tf32 = True  # 允许 TF32 矩阵乘（加速，数值略有差异）
torch.backends.cudnn.allow_tf32 = True  # 允许 cuDNN 使用 TF32
torch.backends.cudnn.benchmark = True  # 固定尺寸输入时启用最优卷积算法搜索
torch.set_float32_matmul_precision("high")  # matmul 精度偏好设置（PyTorch 2.x）

print("✓ Imports complete")  # 导入完成提示
print(f"PyTorch: {torch.__version__}")  # 打印 PyTorch 版本
print(f"CUDA: {torch.cuda.is_available()}")  # 是否可用 CUDA
if torch.cuda.is_available():  # 若有 GPU 则打印硬件信息
    print(f"GPU: {torch.cuda.get_device_name(0)}")  # GPU 型号
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")  # 显存容量
print(f"timm version: {timm.__version__}")  # timm 版本（影响模型实现/权重命名）

class CFG:
    BASE_PATH = ''  # 数据根目录（Kaggle 通常为空或指定输入路径）
    TRAIN_CSV = os.path.join(BASE_PATH, 'train.csv')  # 训练标注 CSV（长表：每图 5 行）
    TRAIN_IMAGE_DIR = os.path.join(BASE_PATH, 'train')  # 训练图像目录
    TEST_CSV = os.path.join(BASE_PATH, 'test.csv')  # 测试 CSV（长表：每图 5 行）
    TEST_IMAGE_DIR = os.path.join(BASE_PATH, 'test')  # 测试图像目录
    
    MODEL_DIR = ''  # 模型权重保存目录
    OUTPUT_DIR = ''  # 训练摘要等输出目录
    
    MODEL_NAME = 'vit_huge_plus_patch16_dinov3.lvd1689m'  # timm 模型名（DINOv3 ViT-Huge）
    
    SEED = 42  # 随机种子
    N_FOLDS = 4  # 交叉验证折数
    FOLDS_TO_TRAIN = [0, 1, 2, 3]  # 需要训练的折（可只训练部分）
    
    IMG_SIZE = 512  # 训练输入分辨率（左右半幅各自 resize 到该尺寸）
    BATCH_SIZE = 6  # batch 大小（显存敏感，ViT-Huge 需要较小 batch）
    NUM_WORKERS = 0  # DataLoader 进程数（Kaggle/Windows 常设 0）
    
    EPOCHS = 210  # 最大 epoch 数（配合早停）
    WARMUP_EPOCHS = 2  # warmup 轮数（按 step 计算）
    LR_BACKBONE = 1e-5  # 主干网络学习率（较小以稳定微调）
    LR_HEAD = 5e-4  # 头部/融合层学习率（较大以更快拟合任务）
    WD = 1e-2  # AdamW 权重衰减
    
    CLIP_GRAD_NORM = 1.0  # 梯度裁剪阈值（防止梯度爆炸）
    DROPOUT = 0.2  # Dropout 概率（正则化）
    
    EARLY_STOPPING_PATIENCE = 15  # 早停耐心值（连续不提升 epoch 数）
    
    TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]  # 目标名称（含派生）
    TARGET_COLS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']  # 宽表列顺序
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 训练设备

os.makedirs(CFG.MODEL_DIR, exist_ok=True)  # 创建模型保存目录（若为空字符串则为当前目录）

def seed_everything(seed=CFG.SEED):
    random.seed(seed)  # Python 随机数种子
    np.random.seed(seed)  # NumPy 随机种子
    torch.manual_seed(seed)  # CPU 侧 torch 随机种子
    torch.cuda.manual_seed_all(seed)  # 所有 GPU 的随机种子
    os.environ["PYTHONHASHSEED"] = str(seed)  # 哈希随机性固定（影响 dict/set 次序）

seed_everything()  # 固定随机性，提升可复现性

print(f"\n{'='*60}")
print("CONFIGURATION - VIT_HUGE_PLUS")  # 配置概览标题
print(f"{'='*60}")
print(f"Device: {CFG.DEVICE}")  # 设备信息
print(f"Model: {CFG.MODEL_NAME}")  # 模型名称
print(f"Image Size: {CFG.IMG_SIZE}")  # 输入尺寸
print(f"Batch Size: {CFG.BATCH_SIZE}")  # batch 大小
print(f"Epochs: {CFG.EPOCHS}")  # 最大 epoch
print(f"Folds: {CFG.N_FOLDS}")  # 交叉验证折数

print(f"\n{'='*60}")
print("STEP 1: Loading Data")  # 数据加载阶段
print(f"{'='*60}")

def load_train_data():
    df = pd.read_csv(CFG.TRAIN_CSV)  # 读取训练长表：每行对应（图像，target_name）
    df['image_id'] = df['sample_id'].str.split('__').str[0]  # 从 sample_id 提取图像 ID（用于分组）
    
    # 将长表 pivot 成宽表：每张图像一行，5 个 target 列并列
    df_wide = df.pivot_table(
        index=['image_id', 'image_path'],  # 保留图像 ID 与路径
        columns='target_name',  # 每个 target_name 变为一列
        values='target',  # 真实标签值（克）
        aggfunc='first'  # 每个（图像，target）应唯一，取 first 即可
    ).reset_index()
    
    # 若某些目标列缺失则补 0（防止 pivot 后列不齐导致下游报错）
    for col in CFG.TARGET_COLS:
        if col not in df_wide.columns:
            df_wide[col] = 0.0
    
    # 以 Dry_Total_g 分位数分箱做分层；用于 StratifiedGroupKFold 的 stratify 标签
    df_wide['total_bin'] = pd.qcut(df_wide['Dry_Total_g'], q=5, labels=False, duplicates='drop')
    
    # 分层 + 分组 K 折：分层依据 total_bin；分组依据 image_id（避免同图泄漏）
    sgkf = StratifiedGroupKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    df_wide['fold'] = -1  # 初始化 fold 标记
    
    # 为每条宽表样本分配 fold（val_idx 为当前折验证集索引）
    for fold, (_, val_idx) in enumerate(sgkf.split(df_wide, df_wide['total_bin'], groups=df_wide['image_id'])):
        df_wide.loc[val_idx, 'fold'] = fold
    
    print(f"✓ Loaded {len(df_wide)} training images")  # 训练图像数量（宽表行数）
    print(f"Fold distribution:\n{df_wide['fold'].value_counts().sort_index()}")  # 各折样本分布
    return df_wide  # 返回宽表（每图一行）

def load_test_data():
    df = pd.read_csv(CFG.TEST_CSV)  # 读取测试长表（每图 5 行）
    df['image_id'] = df['sample_id'].str.split('__').str[0]  # 提取图像 ID
    df_unique = df.drop_duplicates('image_id')[['image_id', 'image_path']].reset_index(drop=True)  # 每图保留一行路径
    print(f"✓ Loaded {len(df_unique)} test images")  # 测试图像数量
    return df_unique  # 返回每图唯一列表

train_df = load_train_data()  # 加载并构造训练宽表
test_df = load_test_data()  # 加载测试图像列表（仅用于推理阶段，训练不使用）

def get_train_transforms():
    return A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),  # 统一尺寸（左右半幅各自 resize）
        A.HorizontalFlip(p=0.5),  # 水平翻转增强
        A.VerticalFlip(p=0.5),  # 垂直翻转增强
        A.RandomRotate90(p=0.5),  # 随机 90 度旋转
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),  # 平移/缩放/小角度旋转
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),  # 颜色抖动以增强光照鲁棒性
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet 归一化
        ToTensorV2()  # 转为 torch.Tensor，形状 [C, H, W]
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),  # 验证仅 resize
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 同训练的归一化
        ToTensorV2()  # 转 Tensor
    ])

class BiomassDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)  # 重置索引，确保按 0..N-1 访问
        self.img_dir = img_dir  # 图像根目录
        self.transform = transform  # Albumentations 增强
        self.paths = df['image_path'].values  # 图像相对路径列表
        self.labels = df[CFG.TARGET_COLS].values.astype(np.float32)  # 标签矩阵 [N, 5]，float32

    def __len__(self):
        return len(self.df)  # 数据集大小（图像数）

    def __getitem__(self, idx):
        img_name = os.path.basename(self.paths[idx])  # 取文件名（去掉目录前缀）
        path = os.path.join(self.img_dir, img_name)  # 拼接为实际文件路径
        
        img = cv2.imread(path)  # BGR 读取，形状 [H, W, 3]
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)  # 读图失败时用全零占位，避免训练中断
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转 RGB 以匹配常见预训练规范
        
        h, w, _ = img.shape  # 获取原始高宽
        mid = w // 2  # 沿宽度一分为二的分割点
        left = img[:, :mid]  # 左半幅，形状 [H, W/2, 3]
        right = img[:, mid:]  # 右半幅，形状 [H, W/2, 3]
        
        if self.transform:
            left = self.transform(image=left)['image']  # 增强并转 Tensor，[3, IMG_SIZE, IMG_SIZE]
            right = self.transform(image=right)['image']  # 同上
        
        label = torch.from_numpy(self.labels[idx])  # 标签 Tensor，[5]
        return left, right, label  # 返回（左图，右图，5 维标签）

print("✓ Dataset class defined (NO image cleaning)")  # 数据集定义完成（不做额外清洗）

class LocalMambaBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 对最后一维 C 做 LN，输入预期为 [B, L, C]
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)  # 深度可分离 1D 卷积（沿 token 维 L）
        self.gate = nn.Linear(dim, dim)  # 门控线性层，输出与输入同维
        self.proj = nn.Linear(dim, dim)  # 投影回 dim，用于残差融合
        self.drop = nn.Dropout(dropout)  # Dropout 正则

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # 残差支路，保持形状 [B, L, C]
        x = self.norm(x)  # 归一化稳定训练
        g = torch.sigmoid(self.gate(x))  # 门控系数 g，[B, L, C]，范围 (0,1)
        x = x * g  # 门控调制（逐元素）
        x = x.transpose(1, 2)  # 变为 [B, C, L] 以适配 Conv1d 的通道维
        x = self.dwconv(x)  # 沿 L 做局部混合，输出仍为 [B, C, L]
        x = x.transpose(1, 2)  # 转回 [B, L, C]
        x = self.proj(x)  # 线性投影，保持 [B, L, C]
        x = self.drop(x)  # 随机失活
        return shortcut + x  # 残差连接，输出 [B, L, C]

class BiomassModel(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        self.model_name = model_name  # 记录 backbone 名称
        
        # timm backbone：num_classes=0 使其不带分类头；global_pool='' 返回 token 序列特征（通常为 [B, L, C]）
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0, 
            global_pool=''
        )
        nf = self.backbone.num_features  # 特征维度 C（embedding dim）
        print(f"✓ Backbone: {model_name}, features={nf}")  # 打印主干特征维度
        
        # 融合模块：对拼接后的 token 序列做两层局部混合（近似在序列维建模局部依赖）
        self.fusion = nn.Sequential(
            LocalMambaBlock(nf, kernel_size=5, dropout=CFG.DROPOUT),
            LocalMambaBlock(nf, kernel_size=5, dropout=CFG.DROPOUT)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # 对 token 维 L 做平均池化到 1（将 [B, C, L] -> [B, C, 1]）
        
        # 三个基础分量回归头：Green/Dead/Clover，各输出非负（Softplus），形状 [B, 1]
        self.head_green = nn.Sequential(
            nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(CFG.DROPOUT),
            nn.Linear(nf // 2, 1), nn.Softplus()
        )
        self.head_dead = nn.Sequential(
            nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(CFG.DROPOUT),
            nn.Linear(nf // 2, 1), nn.Softplus()
        )
        self.head_clover = nn.Sequential(
            nn.Linear(nf, nf // 2), nn.GELU(), nn.Dropout(CFG.DROPOUT),
            nn.Linear(nf // 2, 1), nn.Softplus()
        )

    def forward(self, left, right):
        x_l = self.backbone(left)  # 左半幅 backbone 输出，通常 [B, L, C]
        x_r = self.backbone(right)  # 右半幅 backbone 输出，通常 [B, L, C]
        x_cat = torch.cat([x_l, x_r], dim=1)  # 在 token 维拼接：L 变为 2L，形状 [B, 2L, C]
        x_fused = self.fusion(x_cat)  # 融合后仍为 [B, 2L, C]
        x_pool = self.pool(x_fused.transpose(1, 2)).flatten(1)  # 先转为 [B, C, 2L] 再池化到 [B, C]（全局表示）
        
        green = self.head_green(x_pool)  # Dry_Green_g，[B,1]
        dead = self.head_dead(x_pool)  # Dry_Dead_g，[B,1]
        clover = self.head_clover(x_pool)  # Dry_Clover_g，[B,1]
        gdm = green + clover  # GDM_g 作为派生量：绿色干物质=绿草+苜蓿
        total = gdm + dead  # Dry_Total_g：总干物质=GDM+枯死物质
        
        return torch.cat([green, dead, clover, gdm, total], dim=1)  # 拼接为 [B, 5]，与 TARGET_COLS 顺序一致

print("✓ Model architecture defined")  # 模型结构定义完成

def biomass_loss(preds, labels):
    huber = nn.SmoothL1Loss(beta=5.0)  # Huber/平滑 L1，beta 控制线性区间（抗离群点）
    
    loss = huber(preds, labels)  # 直接对 5 维回归做平均损失
    return loss  # 返回标量 loss

def weighted_r2_score(y_true, y_pred):
    weights = np.array([0.1, 0.1, 0.1, 0.2, 0.5])  # 评分权重：Total 权重大，GDM 次之
    r2_scores = []  # 逐目标 R²
    
    for i in range(y_true.shape[1]):  # 遍历 5 个目标维度
        yt = y_true[:, i]  # 真实值 [N]
        yp = y_pred[:, i]  # 预测值 [N]
        ss_res = np.sum((yt - yp) ** 2)  # 残差平方和
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)  # 总平方和
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0  # 防止 ss_tot=0（常数标签）导致除零
        r2_scores.append(r2)  # 收集该维 R²
    
    r2_scores = np.array(r2_scores)  # [5]
    weighted = np.sum(r2_scores * weights) / np.sum(weights)  # 加权平均（此处除以 sum(weights) 便于泛化）
    return weighted, r2_scores  # 返回（加权 R²，逐目标 R²）

print("✓ Loss and metrics defined")  # 损失与指标定义完成

def build_optimizer(model):
    backbone_params = list(model.backbone.parameters())  # 主干参数集合
    backbone_ids = {id(p) for p in backbone_params}  # 记录主干参数 id，用于区分 head 参数
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]  # 非主干参数（融合层+回归头）
    
    # AdamW：对 backbone 与 head 使用不同学习率（微调常用策略）
    return optim.AdamW([
        {'params': backbone_params, 'lr': CFG.LR_BACKBONE},
        {'params': head_params, 'lr': CFG.LR_HEAD}
    ], weight_decay=CFG.WD)

def build_scheduler(optimizer, total_steps):
    def lr_lambda(step):
        warmup_steps = CFG.WARMUP_EPOCHS * (total_steps // CFG.EPOCHS)  # 每 epoch 的 step 数近似为 total_steps/EPOCHS
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))  # 线性 warmup：从 0 -> 1
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))  # 归一化进度 [0,1]
        return 0.5 * (1.0 + math.cos(math.pi * progress))  # 余弦退火：从 1 -> 0
    return LambdaLR(optimizer, lr_lambda)  # 按 step 更新的 LambdaLR

scaler = GradScaler()  # AMP 梯度缩放器（降低 FP16 下溢风险）

def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()  # 训练模式（启用 dropout/BN 等）
    total_loss = 0.0  # 累计 loss
    
    pbar = tqdm(loader, desc='Training')  # 训练进度条（按 batch）
    for i, (left, right, labels) in enumerate(pbar):
        left = left.to(device)  # 左图 Tensor -> GPU/CPU，[B,3,H,W]
        right = right.to(device)  # 右图 Tensor -> GPU/CPU，[B,3,H,W]
        labels = labels.to(device)  # 标签 -> device，[B,5]
        
        optimizer.zero_grad()  # 清空梯度
        
        with autocast():  # AMP 自动混合精度上下文
            preds = model(left, right)  # 前向输出 [B,5]
            loss = biomass_loss(preds, labels)  # 计算回归损失
        
        scaler.scale(loss).backward()  # 缩放后反向传播，减少 FP16 下溢
        scaler.unscale_(optimizer)  # 反缩放到真实梯度，便于裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.CLIP_GRAD_NORM)  # 梯度范数裁剪
        scaler.step(optimizer)  # 使用缩放器执行优化器 step（内部会跳过 inf/nan）
        scaler.update()  # 更新缩放因子
        scheduler.step()  # 按 step 更新学习率
        
        total_loss += loss.item()  # 累加标量 loss
        pbar.set_postfix({'loss': f'{total_loss/(i+1):.4f}'})  # 显示当前平均 loss
    
    return total_loss / len(loader)  # 返回 epoch 平均 loss

@torch.no_grad()
def validate(model, loader, device):
    model.eval()  # 推理模式（关闭 dropout 等）
    all_preds = []  # 收集预测（numpy）
    all_labels = []  # 收集标签（numpy）
    
    for left, right, labels in tqdm(loader, desc='Validating'):
        left = left.to(device)  # 左图到 device
        right = right.to(device)  # 右图到 device
        
        with autocast():  # 验证同样使用 AMP 以提速
            preds = model(left, right)  # 预测 [B,5]
        
        all_preds.append(preds.cpu().numpy())  # 转回 CPU numpy，便于后续指标计算
        all_labels.append(labels.numpy())  # labels 已在 CPU（来自 DataLoader），直接 numpy
    
    all_preds = np.concatenate(all_preds)  # 拼接为 [N,5]
    all_labels = np.concatenate(all_labels)  # 拼接为 [N,5]
    
    weighted_r2, per_target_r2 = weighted_r2_score(all_labels, all_preds)  # 计算加权 R² 与逐目标 R²
    return weighted_r2, per_target_r2  # 返回验证指标

print("✓ Training functions defined")  # 训练/验证函数定义完成

print(f"\n{'='*60}")
print("STEP 2: Training DINO HUGE Models")  # 模型训练阶段
print(f"{'='*60}")

def train_fold(fold, train_df):
    print(f"\n{'='*60}")
    print(f"TRAINING FOLD {fold}")  # 当前训练折
    print(f"{'='*60}")
    
    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)  # 当前折训练集（除去 fold）
    val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)  # 当前折验证集
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")  # 打印样本数
    
    train_dataset = BiomassDataset(train_data, CFG.TRAIN_IMAGE_DIR, get_train_transforms())  # 训练集（含增强）
    val_dataset = BiomassDataset(val_data, CFG.TRAIN_IMAGE_DIR, get_val_transforms())  # 验证集（无增强）
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CFG.BATCH_SIZE,  # batch 大小
        shuffle=True,  # 训练打乱
        num_workers=CFG.NUM_WORKERS,  # 读取进程数
        pin_memory=True,  # 固定内存，加速 CPU->GPU 拷贝
        drop_last=True  # 丢弃最后不足 batch 的样本，保持 batch 统计一致
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CFG.BATCH_SIZE,  # 验证 batch
        shuffle=False,  # 验证不打乱
        num_workers=CFG.NUM_WORKERS,  # 读取进程数
        pin_memory=True  # 加速拷贝
    )
    
    model = BiomassModel(CFG.MODEL_NAME, pretrained=True).to(CFG.DEVICE)  # 构建并加载预训练权重，移动到 device
    
    optimizer = build_optimizer(model)  # 构建分组学习率的 AdamW
    total_steps = len(train_loader) * CFG.EPOCHS  # 总 step 数（epoch*steps_per_epoch）
    scheduler = build_scheduler(optimizer, total_steps)  # warmup+余弦退火调度器
    
    best_r2 = -float('inf')  # 记录当前最佳加权 R²
    best_epoch = 0  # 记录最佳 epoch（从 1 开始展示）
    epochs_without_improvement = 0  # 连续未提升计数
    
    epoch_pbar = tqdm(range(CFG.EPOCHS), desc=f'Fold {fold} Epochs')  # epoch 级进度条
    for epoch in epoch_pbar:
        print(f"\nEpoch {epoch + 1}/{CFG.EPOCHS}")  # 当前 epoch
        
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, CFG.DEVICE)  # 训练一个 epoch
        val_r2, per_r2 = validate(model, val_loader, CFG.DEVICE)  # 验证并计算 R²
        
        print(f"Train Loss: {train_loss:.4f}")  # 打印训练损失
        print(f"Val R²: {val_r2:.4f}")  # 打印加权 R²
        print(f"Per-target: Green={per_r2[0]:.3f}, Dead={per_r2[1]:.3f}, Clover={per_r2[2]:.3f}, GDM={per_r2[3]:.3f}, Total={per_r2[4]:.3f}")  # 逐目标 R²
        
        epoch_pbar.set_postfix({'loss': f'{train_loss:.4f}', 'val_r2': f'{val_r2:.4f}', 'best_r2': f'{best_r2:.4f}'})  # 更新进度条信息
        
        if val_r2 > best_r2:  # 若验证指标提升则保存
            best_r2 = val_r2  # 更新最佳 R²
            best_epoch = epoch + 1  # 更新最佳 epoch（1-index）
            epochs_without_improvement = 0  # 重置早停计数
            save_path = f"{CFG.MODEL_DIR}/fold{fold}_best.pth"  # 保存路径
            torch.save(model.state_dict(), save_path)  # 保存权重（仅 state_dict）
            print(f"✓ Saved best model (R²={best_r2:.4f})")  # 保存提示
        else:
            epochs_without_improvement += 1  # 累计未提升次数
            print(f"No improvement for {epochs_without_improvement} epoch(s)")  # 打印未提升信息
            
            if epochs_without_improvement >= CFG.EARLY_STOPPING_PATIENCE:  # 达到耐心值则早停
                print(f"Early stopping triggered after {epoch + 1} epochs")  # 早停提示
                break
    
    print(f"\nFold {fold} Best: R²={best_r2:.4f} at epoch {best_epoch}")  # 当前折最佳结果
    
    del model, optimizer, scheduler, train_loader, val_loader  # 删除大对象以释放内存
    gc.collect()  # 触发 Python 垃圾回收
    torch.cuda.empty_cache()  # 清空 CUDA 缓存（减小峰值占用）
    
    return best_r2  # 返回该折最佳 R²

fold_scores = []  # 存放各折最佳分数
fold_pbar = tqdm(CFG.FOLDS_TO_TRAIN, desc='Training Folds')  # 折级进度条
for fold in fold_pbar:
    fold_pbar.set_description(f'Training Fold {fold}')  # 更新折描述
    score = train_fold(fold, train_df)  # 训练该折并得到最佳分数
    fold_scores.append(score)  # 记录分数
    fold_pbar.set_postfix({'current_r2': f'{score:.4f}', 'mean_r2': f'{np.mean(fold_scores):.4f}'})  # 显示当前/均值

print(f"\n{'='*60}")
print("DINO HUGE TRAINING COMPLETE!")  # 所有折训练完成
print(f"{'='*60}")
print(f"Fold scores: {fold_scores}")  # 打印各折分数
print(f"Mean CV R²: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")  # 打印均值±标准差

summary = {
    'model': CFG.MODEL_NAME,  # 模型名称
    'folds': CFG.N_FOLDS,  # 折数
    'epochs': CFG.EPOCHS,  # 最大 epoch
    'batch_size': CFG.BATCH_SIZE,  # batch 大小
    'fold_scores': fold_scores,  # 各折分数
    'mean_cv': np.mean(fold_scores),  # 交叉验证均值
    'std_cv': np.std(fold_scores)  # 交叉验证标准差
}

import json
with open(f'{CFG.OUTPUT_DIR}/training_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)  # 保存训练摘要为 JSON（缩进 2）

print(f"\n{'='*60}")
print("TRAINING COMPLETE!")  # 全流程结束
print(f"{'='*60}")
print(f"\nModels saved to: {CFG.MODEL_DIR}/")  # 模型保存目录提示
print(f"  - fold0_best.pth")  # fold0 最佳权重文件
print(f"  - fold1_best.pth")  # fold1 最佳权重文件
print(f"  - fold2_best.pth")  # fold2 最佳权重文件
print(f"  - fold3_best.pth")  # fold3 最佳权重文件
