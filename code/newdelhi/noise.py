#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_noisy_similarity.py

功能：
- 从原始 EEG 目录读取所有 .txt/.TXT 文件（递归）。
- 对每个文件按规则（1D视为样本，2D flatten为样本）加载为信号。
- 对每个噪声比例（10%～100%），添加高斯噪声（噪声标准差 = ratio * 原信号std），
  然后按行（样本）中心化 + L2归一化，计算 Pearson 相关矩阵（负相关置0，对角线置0）。
- 保存每个噪声比例对应的相似矩阵为 similarity10.npy, similarity20.npy, ... 
  到输出目录 /mnt/gs21/scratch/jiangj33/ldx/case1/similarity_pearson。

用法：直接运行，需确保原始数据目录存在。
"""

import os
import glob
import numpy as np

# ---------------------- 配置区 ----------------------
BASE_DIR = '/mnt/gs21/scratch/jiangj33/ldx/case17'
DATA_DIR = os.path.join(BASE_DIR, 'eeg_data')          # 原始干净数据
OUT_DIR = os.path.join(BASE_DIR, 'similarity_pearson') # 输出目录
NOISE_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]   # 10% ~ 100%
RANDOM_SEED = 42
# ---------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 1. 递归获取所有 .txt/.TXT 文件（保留顺序） ----------
pattern = os.path.join(DATA_DIR, '**', '*.txt')
file_paths = sorted(set(
    glob.glob(pattern, recursive=True) +
    glob.glob(pattern.replace('.txt', '.TXT'), recursive=True)
))
print(f"找到 {len(file_paths)} 个数据文件。")

if not file_paths:
    raise FileNotFoundError(f"未在 {DATA_DIR} 中找到任何 .txt/.TXT 文件。")

# ---------- 2. 加载所有原始信号（保持文件顺序） ----------
def load_signal(filepath):
    """加载文件并返回一个扁平的一维数组（保留原flatten行为）"""
    try:
        arr = np.loadtxt(filepath)
    except Exception as e:
        print(f"加载失败，跳过: {filepath} -> {e}")
        return None
    if arr.size == 0:
        print(f"空文件，跳过: {filepath}")
        return None
    # 2D矩阵 flatten，1D直接使用
    if arr.ndim == 1:
        return arr.reshape(-1)
    else:
        return arr.flatten()

raw_signals = []
skipped = 0
for p in file_paths:
    sig = load_signal(p)
    if sig is None:
        skipped += 1
        continue
    raw_signals.append(sig)

if not raw_signals:
    raise RuntimeError("没有加载到任何有效信号。请检查数据。")

# 检查所有信号长度是否一致（因为后续需要合并为矩阵，每行一个样本，长度必须相同）
lengths = set(sig.size for sig in raw_signals)
if len(lengths) > 1:
    raise ValueError(f"不同文件长度不一致: {sorted(lengths)}。请确保所有文件长度相同。")

n_samples = len(raw_signals)
n_features = raw_signals[0].size
print(f"成功加载 {n_samples} 个样本，每个样本特征数: {n_features} (跳过 {skipped} 个文件)")

# 将原始信号堆叠为矩阵 (n_samples, n_features)
X_clean = np.vstack(raw_signals)   # 每行一个样本

# ---------- 3. 定义相似矩阵计算函数 ----------
def compute_similarity(matrix, zero_std_eps=1e-12):
    """
    输入: matrix (n_samples, n_features)
    输出: 非负相关矩阵 (n_samples, n_samples)，对角线为0
    """
    # 中心化
    means = matrix.mean(axis=1, keepdims=True)
    Xc = matrix - means
    # L2范数
    norms = np.linalg.norm(Xc, axis=1)
    zero_std_mask = norms < zero_std_eps
    # 防止除零
    norms_safe = norms.copy()
    norms_safe[zero_std_mask] = 1.0
    Xn = Xc / norms_safe[:, None]   # 归一化行向量
    # 计算内积 -> Pearson相关
    corr = Xn @ Xn.T
    corr = np.clip(corr, -1.0, 1.0)
    # 零方差样本相关置0
    if zero_std_mask.any():
        corr[zero_std_mask, :] = 0.0
        corr[:, zero_std_mask] = 0.0
    # 负相关置0，对角线置0
    corr_nonneg = np.clip(corr, 0.0, 1.0)
    np.fill_diagonal(corr_nonneg, 0.0)
    return corr_nonneg

# ---------- 4. 对每个噪声比例生成带噪信号并计算相似矩阵 ----------
rng = np.random.default_rng(RANDOM_SEED)

for ratio in NOISE_RATIOS:
    print(f"\n处理噪声比例: {ratio*100:.0f}%")
    # 生成带噪信号 (与原始信号相同形状)
    noisy_signals = []
    for idx, sig in enumerate(raw_signals):
        # 计算原始信号标准差
        std_orig = np.std(sig)
        if std_orig == 0.0:
            # 常数信号，使用1.0避免零噪声尺度
            scale = 1.0
        else:
            scale = std_orig
        # 生成高斯噪声
        noise = rng.normal(loc=0.0, scale=ratio * scale, size=sig.shape)
        noisy_sig = sig + noise
        noisy_signals.append(noisy_sig)
    
    # 堆叠为矩阵
    X_noisy = np.vstack(noisy_signals)   # 形状 (n_samples, n_features)
    # 计算相似矩阵
    sim = compute_similarity(X_noisy)
    # 保存为 .npy
    out_file = os.path.join(OUT_DIR, f"similarity{int(ratio*100)}.npy")
    np.save(out_file, sim)
    print(f"已保存: {out_file}")

print("\n所有噪声比例的相似矩阵已生成完毕。")
print(f"输出目录: {OUT_DIR}")
