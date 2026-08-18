#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_sample_similarity.py (glob recursive, case-insensitive file matching + flatten 2D files)

功能：
- 递归读取 data_dir 下所有 .txt/.TXT 文件（子目录也会被收集）。
- 若文件为 1D 向量：视为 1 个样本；若为 2D 矩阵：按文件整体 flatten 为 1 个样本（与用户 snippet 一致）。
- 合并为 (n_samples, n_features)，检查长度一致性（不同长度会报错）。
- 按行（每行是样本）中心化 + L2 归一化，计算内积 -> Pearson 等价相关矩阵（-1..1）。
- 对零方差样本特殊处理；将负相关置 0，并将对角线置 0。
"""

import os
import glob
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ---------------------- 配置区（按需修改） ----------------------
data_dir = '/mnt/gs21/scratch/jiangj33/ldx/case1/eeg_data'   # 存放 .txt 的根目录（会递归）
save_dir = '/mnt/gs21/scratch/jiangj33/ldx/case1/similarity_pearson'  # 输出目录
save_heatmap = True         # 是否保存热图 PNG
heatmap_prefix = 'sample_corr_pearson'  # heatmap 文件名前缀
# ---------------------------------------------------------------

os.makedirs(save_dir, exist_ok=True)

# ---------- 1. 递归匹配所有 .txt/.TXT 文件（不区分大小写） ----------
pattern = os.path.join(data_dir, '**', '*.txt')
file_paths = sorted(set(
    glob.glob(pattern, recursive=True) +
    glob.glob(pattern.replace('.txt', '.TXT'), recursive=True)
))
print("匹配到的 txt 文件数:", len(file_paths))

# ---------- 2. 加载文件（按你的 snippet 行为：2D->flatten） ----------
loaded = []
skipped = 0
for p in file_paths:
    if not os.path.isfile(p):
        print("跳过（非普通文件）:", p)
        skipped += 1
        continue
    try:
        arr = np.loadtxt(p)
    except Exception as e:
        print(f"加载失败，跳过: {p}  -> {e}")
        skipped += 1
        continue

    if arr.size == 0:
        print("跳过空文件:", p)
        skipped += 1
        continue

    if arr.ndim == 1:
        # 单行/向量，视为 1 个样本
        loaded.append(arr.reshape(1, -1))
    else:
        # 多行矩阵：按文件整体 flatten 为 1 个样本
        arr_flat = arr.flatten()
        loaded.append(arr_flat.reshape(1, -1))

if len(loaded) == 0:
    raise RuntimeError("没有加载任何文件，请检查路径与权限。")

# 检查不同文件长度集合
lengths = set(x.size for x in loaded)
print("不同文件长度集合:", lengths)
if len(lengths) > 1:
    msg = (
        "检测到不同文件具有不同长度（features）。当前加载文件的长度集合为:\n"
        f"{sorted(lengths)}\n\n"
        "请确保所有文件产生的 feature 长度一致。"
    )
    raise ValueError(msg)

# 合并为 (n_samples, n_features)
eeg_data = np.vstack([x.reshape(1, -1) for x in loaded])
n_samples, n_features = eeg_data.shape
print("合并后形状：", eeg_data.shape)
print(f"实际加载样本数: {n_samples}, 每样本特征数: {n_features} (skipped files: {skipped})")

# ---------- 3. 计算样本间 Pearson 相关（按行样本，稳健处理零方差） ----------
means = eeg_data.mean(axis=1, keepdims=True)
Xc = eeg_data - means
norms = np.linalg.norm(Xc, axis=1)
zero_std_mask = norms < 1e-12

# 防止除零
norms_safe = norms.copy()
norms_safe[zero_std_mask] = 1.0
Xn = Xc / norms_safe[:, None]  # 归一化后的行向量（L2=1 或 0）

# 计算相关（-1..1）
corr = Xn @ Xn.T
corr = np.clip(corr, -1.0, 1.0)  # 数值稳定性

# 零方差样本：与其它样本相关度设为 0
if zero_std_mask.any():
    num_zero = zero_std_mask.sum()
    print(f"发现 {num_zero} 个零方差（常数）样本；将其与其它样本的相关置为 0。")
    corr[zero_std_mask, :] = 0.0
    corr[:, zero_std_mask] = 0.0

# 对角线置 0（不保留自连接）
np.fill_diagonal(corr, 0.0)

# 负相关置 0，保留正相关
corr_nonneg = np.clip(corr, 0.0, 1.0)

print("相关矩阵计算完成。范围检查：", corr.min(), corr.max(), "-> 非负后:", corr_nonneg.min(), corr_nonneg.max())

# ---------- 4. 保存结果 ----------
out_npy = os.path.join(save_dir, "similarity.npy")
np.save(out_npy, corr_nonneg)

out_csv = os.path.join(save_dir, "similarity.csv")
np.savetxt(out_csv, corr_nonneg, delimiter=',')

print("已保存：", out_npy)
print("已保存：", out_csv)

print("全部完成。")