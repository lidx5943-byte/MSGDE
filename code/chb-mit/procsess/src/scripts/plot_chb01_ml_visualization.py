# Author: 王梓涵 <wangzh011031@163.com>
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHB01患者LR模型S+D特征可视化脚本

功能：
1. 加载chb01患者S和D特征数据
2. 组合S+D特征
3. 使用LR模型训练并预测（只绘制测试集）
4. 绘制三种降维方式（PCA, t-SNE, UMAP）的分类边界图（class 0=interictal, class 1=preictal）
5. 单独绘制ROC曲线（无标题）
6. 按SCI顶刊风格输出高清晰度图像

输出目录: /srv/New_eeg_code/3M/supplyment_plot
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegression

# 添加项目根目录到pythonpath
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.visualization.nature_style import set_nature_style

# 尝试导入UMAP
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn not installed, UMAP visualization will be skipped")


def load_chb01_all_features(results_dir: str = "/mnt/3M/chbmit-allchannels/per_patient_results") -> Tuple[np.ndarray, np.ndarray, Dict]:
    """加载chb01患者所有特征数据并加载最优模型参数"""
    patient_dir = Path(results_dir) / "chb01"
    models_dir = patient_dir / "models"

    # 加载所有特征 (ALL = S+T+D)
    features = np.load(patient_dir / "features.npy")
    labels = np.load(patient_dir / "labels.npy")

    # 加载最优模型参数 (ALL_LR)
    params_file = models_dir / "ALL_LR_params.json"
    if params_file.exists():
        with open(params_file, 'r') as f:
            best_params = json.load(f)
        print(f"Loaded optimal LR parameters from ALL_LR: {best_params}")
    else:
        best_params = {'C': 0.001, 'solver': 'lbfgs'}
        print(f"Using default LR parameters: {best_params}")

    print(f"Loaded chb01 ALL features:")
    print(f"  features shape: {features.shape}")
    print(f"  labels shape: {labels.shape}, unique: {np.unique(labels)}")

    return features, labels, best_params


def prepare_2d_data(features: np.ndarray, labels: np.ndarray, method: str = 'pca') -> Tuple[np.ndarray, np.ndarray, object]:
    """
    将高维特征降维到2D
    """
    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    if method == 'pca':
        transformer = PCA(n_components=2, random_state=42)
        features_2d = transformer.fit_transform(features_scaled)
        explained_var = transformer.explained_variance_ratio_
        print(f"  PCA explained variance: {explained_var[0]:.3f}, {explained_var[1]:.3f}")
    elif method == 'tsne':
        transformer = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        features_2d = transformer.fit_transform(features_scaled)
        print(f"  t-SNE completed")
    elif method == 'umap':
        if not UMAP_AVAILABLE:
            raise ImportError("umap-learn is not installed")
        transformer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        features_2d = transformer.fit_transform(features_scaled)
        print(f"  UMAP completed")
    else:
        raise ValueError(f"Unknown method: {method}")

    return features_2d, labels, transformer


def plot_decision_boundary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model,
    method: str,
    output_dir: Path
):
    """
    绘制分类边界图（只显示测试集，class 0=interictal, class 1=preictal，无标题）
    """
    set_nature_style()

    # 深蓝深红配色
    color_class_0 = '#1B4F72'  # 深蓝 - interictal
    color_class_1 = '#922B21'  # 深红 - preictal

    fig, ax = plt.subplots(figsize=(6.0, 5.5))

    # 训练模型
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_clone = model.__class__(**model.get_params())
    model_clone.fit(X_train_scaled, y_train)

    # 创建网格用于绘制决策边界
    h = 0.02
    x_min, x_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
    y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 预测网格点
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model_clone.predict(scaler.transform(grid_points))
    Z = Z.reshape(xx.shape)

    # 绘制决策边界背景（淡色）
    ax.contourf(xx, yy, Z, alpha=0.25, levels=1, colors=[color_class_0, color_class_1])
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=1.2, linestyles='--')

    # 只绘制测试集数据点
    mask_0 = y_test == 0
    mask_1 = y_test == 1

    ax.scatter(X_test[mask_0, 0], X_test[mask_0, 1],
              c=color_class_0, edgecolors='black', linewidths=0.8,
              s=70, alpha=0.9, marker='o', label='interictal')
    ax.scatter(X_test[mask_1, 0], X_test[mask_1, 1],
              c=color_class_1, edgecolors='black', linewidths=0.8,
              s=70, alpha=0.9, marker='s', label='preictal')

    # 设置标签（无标题）
    if method == 'pca':
        xlabel = 'PC1'
        ylabel = 'PC2'
    elif method == 'tsne':
        xlabel = 't-SNE 1'
        ylabel = 't-SNE 2'
    else:
        xlabel = 'UMAP 1'
        ylabel = 'UMAP 2'

    ax.set_xlabel(xlabel, fontweight='bold', fontsize=10)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=10)
    # 无标题
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    plt.tight_layout()

    # 保存
    filename = f"chb01_LR_ALL_optimal_{method}_boundary"
    formats = ['png', 'pdf', 'svg', 'tif']
    for fmt in formats:
        filepath = output_dir / f"{filename}.{fmt}"
        if fmt == 'tif':
            fig.savefig(filepath, format='tiff', dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none', pil_kwargs={"compression": "tiff_lzw"})
        else:
            fig.savefig(filepath, format=fmt, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        print(f"  Saved: {filepath}")

    plt.close(fig)

    return model_clone, scaler


def plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    method: str,
    output_dir: Path
):
    """绘制混淆矩阵（无标题）"""
    set_nature_style()

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    # 颜色映射 - 使用蓝色渐变
    cmap = plt.cm.Blues

    # 绘制热力图
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=9)

    # 设置坐标轴
    classes = ['interictal', 'preictal']
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)

    # 设置轴标签
    ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=10)
    ax.set_ylabel('True Label', fontweight='bold', fontsize=10)

    # 在每个单元格中添加数值
    thresh = cm.max() / 2.
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14, fontweight='bold')

    # 移除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # 保存
    filename = f"chb01_LR_ALL_optimal_{method}_confusion_matrix"
    formats = ['png', 'pdf', 'svg', 'tif']
    for fmt in formats:
        filepath = output_dir / f"{filename}.{fmt}"
        if fmt == 'tif':
            fig.savefig(filepath, format='tiff', dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none', pil_kwargs={"compression": "tiff_lzw"})
        else:
            fig.savefig(filepath, format=fmt, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        print(f"  Saved: {filepath}")

    plt.close(fig)

    return cm


def plot_roc_curve(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    method: str,
    output_dir: Path
):
    """单独绘制ROC曲线（无标题）"""
    set_nature_style()

    color_roc = '#1B4F72'  # 深蓝

    fig, ax = plt.subplots(figsize=(6.0, 5.5))

    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    ax.plot(fpr, tpr, color=color_roc, linewidth=2.5,
           label=f'LR (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.0, alpha=0.5, label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=10)
    # 无标题
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    plt.tight_layout()

    # 保存
    filename = f"chb01_LR_ALL_optimal_{method}_roc"
    formats = ['png', 'pdf', 'svg', 'tif']
    for fmt in formats:
        filepath = output_dir / f"{filename}.{fmt}"
        if fmt == 'tif':
            fig.savefig(filepath, format='tiff', dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none', pil_kwargs={"compression": "tiff_lzw"})
        else:
            fig.savefig(filepath, format=fmt, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        print(f"  Saved: {filepath}")

    plt.close(fig)

    return roc_auc


def create_lr_visualization(
    features: np.ndarray,
    labels: np.ndarray,
    best_params: Dict,
    method: str,
    output_dir: Path
):
    """
    使用最优参数为LR创建可视化（分类边界、ROC和混淆矩阵）
    """
    print(f"\nProcessing with {method.upper()}...")

    # 降维到2D
    X_2d, y, _ = prepare_2d_data(features, labels, method=method)

    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_2d, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"  Train set: {len(y_train)} samples, Test set: {len(y_test)} samples")

    # 创建LR模型（使用最优参数）
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=best_params.get('C', 0.001),
        solver=best_params.get('solver', 'lbfgs')
    )
    print(f"  Using optimal parameters: C={lr_model.C}, solver={lr_model.solver}")

    # 1. 绘制分类边界图（只显示测试集）
    model_fitted, scaler = plot_decision_boundary(
        X_train, y_train, X_test, y_test, lr_model, method, output_dir
    )

    # 2. 预测概率并绘制ROC曲线
    X_test_scaled = scaler.transform(X_test)
    y_prob = model_fitted.predict_proba(X_test_scaled)[:, 1]

    auc_score = plot_roc_curve(y_test, y_prob, method, output_dir)

    # 3. 预测标签并绘制混淆矩阵
    y_pred = model_fitted.predict(X_test_scaled)
    cm = plot_confusion_matrix(y_test, y_pred, method, output_dir)
    print(f"  Confusion Matrix:\n{cm}")

    # 计算准确率
    accuracy = np.mean(y_pred == y_test)
    print(f"  Test Accuracy: {accuracy:.4f}")

    return auc_score, accuracy


def main():
    """主函数"""
    print("=" * 60)
    print("CHB01 Optimal LR (ALL features) Visualization")
    print("=" * 60)
    print("Using optimal parameters from ALL_LR model:")
    print("  - Feature set: ALL (S+T+D)")
    print("  - Model: Logistic Regression")
    print("  - Optimal params: C=0.001, solver=lbfgs")
    print("=" * 60)

    # 设置输出目录
    output_dir = Path("/srv/New_eeg_code/3M/supplyment_plot")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载所有特征数据和最优模型参数
    features, labels, best_params = load_chb01_all_features()

    # 2. 使用三种降维方式创建可视化
    methods = ['pca', 'tsne']
    if UMAP_AVAILABLE:
        methods.append('umap')

    results_summary = {}

    for method in methods:
        try:
            auc_score, accuracy = create_lr_visualization(features, labels, best_params, method, output_dir)
            results_summary[method] = {'auc': auc_score, 'accuracy': accuracy}
        except Exception as e:
            print(f"  Error with {method}: {e}")
            results_summary[method] = {'auc': 0.0, 'accuracy': 0.0, 'error': str(e)}

    # 3. 保存结果摘要
    results_path = output_dir / "chb01_lr_optimal_visualization_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n  Saved results summary: {results_path}")

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
