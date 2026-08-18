# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""特征分布、t-SNE、PCA、重要性及组合面板。"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from typing import Optional, List
from .nature_style import (
    set_nature_style, add_panel_label,
    NATURE_COLORS, PALETTE_2CLASS
)


def plot_feature_distribution(
    features: np.ndarray, 
    output_path: str,
    feature_names: Optional[List[str]] = None,
    labels: Optional[np.ndarray] = None,
    n_plot: int = 9
):
    """特征分布直方图（可选按标签分组）。"""
    set_nature_style()
    
    n_features = features.shape[1]
    n_plot = min(n_features, n_plot)
    
    n_cols = 3
    n_rows = (n_plot + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3 * n_rows))
    axes = axes.flatten() if n_plot > 1 else [axes]
    
    for i in range(n_plot):
        ax = axes[i]
        
        if labels is not None:
            for label_val in np.unique(labels):
                mask = labels == label_val
                color = PALETTE_2CLASS[int(label_val) % len(PALETTE_2CLASS)]
                label_name = f'Class {int(label_val)}'
                sns.histplot(features[mask, i], kde=True, ax=ax, 
                            color=color, alpha=0.5, label=label_name)
            ax.legend(fontsize=7, loc='upper right')
        else:
            sns.histplot(features[:, i], kde=True, ax=ax, 
                        color=NATURE_COLORS['blue'], alpha=0.7)
        
        title = feature_names[i] if feature_names and i < len(feature_names) else f"Feature {i}"
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Count' if i % n_cols == 0 else '')
    
    for i in range(n_plot, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_tsne(
    features: np.ndarray, 
    labels: np.ndarray, 
    output_path: str,
    perplexity: int = 30,
    title: str = "t-SNE Visualization"
):
    """t-SNE 二维嵌入。"""
    set_nature_style()
    
    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        n_samples = features.shape[0]
        perplexity = min(perplexity, n_samples // 3, 50)
        perplexity = max(perplexity, 5)
        
        tsne = TSNE(n_components=2, perplexity=perplexity, 
                   random_state=42, init='pca', learning_rate='auto')
        X_emb = tsne.fit_transform(features_scaled)
        
        fig, ax = plt.subplots(figsize=(7, 6))
        
        unique_labels = np.unique(labels)
        for i, label_val in enumerate(unique_labels):
            mask = labels == label_val
            color = PALETTE_2CLASS[i % len(PALETTE_2CLASS)]
            
            ax.scatter(X_emb[mask, 0], X_emb[mask, 1], 
                      c=color, s=50, alpha=0.7, 
                      edgecolors='white', linewidths=0.5,
                      label=f'Class {int(label_val)}')
        
        ax.set_xlabel('t-SNE 1', fontsize=10)
        ax.set_ylabel('t-SNE 2', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right', frameon=True,
                 fancybox=False, edgecolor=NATURE_COLORS['border'])
        
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        
        add_panel_label(ax, 'a')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        pdf_path = output_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
        plt.close()
        
    except ImportError:
        pass


def plot_pca(
    features: np.ndarray, 
    labels: np.ndarray, 
    output_path: str,
    title: str = "PCA Visualization"
):
    """PCA 二维投影。"""
    set_nature_style()
    
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        pca = PCA(n_components=2)
        X_emb = pca.fit_transform(features_scaled)
        
        fig, ax = plt.subplots(figsize=(7, 6))
        
        unique_labels = np.unique(labels)
        for i, label_val in enumerate(unique_labels):
            mask = labels == label_val
            color = PALETTE_2CLASS[i % len(PALETTE_2CLASS)]
            ax.scatter(X_emb[mask, 0], X_emb[mask, 1], 
                      c=color, s=50, alpha=0.7, 
                      edgecolors='white', linewidths=0.5,
                      label=f'Class {int(label_val)}')
        
        var_explained = pca.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)', fontsize=10)
        ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right', frameon=True,
                 fancybox=False, edgecolor=NATURE_COLORS['border'])
        
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        
        add_panel_label(ax, 'a')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(output_path.replace('.png', '.pdf'), format='pdf',
                   bbox_inches='tight', facecolor='white')
        plt.close()
        
    except ImportError:
        pass


def plot_feature_importance(
    importance_scores: np.ndarray,
    feature_names: List[str],
    output_path: str,
    top_k: int = 20,
    title: str = "Feature Importance"
):
    """特征重要性条形图。"""
    set_nature_style()
    
    indices = np.argsort(importance_scores)[::-1][:top_k]
    scores = importance_scores[indices]
    names = [feature_names[i] for i in indices]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(scores)))
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, scores, color=colors, edgecolor='white', height=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    max_score = scores.max()
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax.text(width + max_score * 0.01, bar.get_y() + bar.get_height()/2,
               f'{score:.3f}', va='center', fontsize=7)
    
    add_panel_label(ax, 'a')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf',
               bbox_inches='tight', facecolor='white')
    plt.close()


def plot_feature_comparison(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    feature_names: Optional[List[str]] = None,
    method: str = 'all'
):
    """PCA、t-SNE 与类间分离度组合面板。"""
    set_nature_style()
    
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)
    
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    ax_a = fig.add_subplot(gs[0])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(features_scaled)
    
    for i, label_val in enumerate(np.unique(labels)):
        mask = labels == label_val
        ax_a.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                    c=PALETTE_2CLASS[i % 2], s=40, alpha=0.7,
                    edgecolors='white', linewidths=0.3,
                    label=f'Class {int(label_val)}')
    
    var = pca.explained_variance_ratio_
    ax_a.set_xlabel(f'PC1 ({var[0]*100:.1f}%)')
    ax_a.set_ylabel(f'PC2 ({var[1]*100:.1f}%)')
    ax_a.set_title('PCA', fontsize=10, fontweight='bold')
    ax_a.legend(fontsize=7)
    ax_a.grid(True, linestyle='--', alpha=0.3)
    add_panel_label(ax_a, 'a')
    
    ax_b = fig.add_subplot(gs[1])
    perplexity = min(30, len(labels) // 3)
    tsne = TSNE(n_components=2, perplexity=max(5, perplexity), 
               random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(features_scaled)
    
    for i, label_val in enumerate(np.unique(labels)):
        mask = labels == label_val
        ax_b.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                    c=PALETTE_2CLASS[i % 2], s=40, alpha=0.7,
                    edgecolors='white', linewidths=0.3,
                    label=f'Class {int(label_val)}')
    
    ax_b.set_xlabel('t-SNE 1')
    ax_b.set_ylabel('t-SNE 2')
    ax_b.set_title('t-SNE', fontsize=10, fontweight='bold')
    ax_b.legend(fontsize=7)
    ax_b.grid(True, linestyle='--', alpha=0.3)
    add_panel_label(ax_b, 'b')
    
    ax_c = fig.add_subplot(gs[2])
    
    class_means = []
    class_stds = []
    for label_val in np.unique(labels):
        mask = labels == label_val
        class_means.append(X_pca[mask].mean(axis=0))
        class_stds.append(X_pca[mask].std(axis=0).mean())
    
    inter_dist = np.linalg.norm(class_means[0] - class_means[1]) if len(class_means) > 1 else 0
    intra_dist = np.mean(class_stds)
    sep_ratio = inter_dist / (intra_dist + 1e-8)
    
    metrics = ['Inter-class\nDistance', 'Intra-class\nDistance', 'Separation\nRatio']
    values = [inter_dist, intra_dist, sep_ratio]
    colors = [NATURE_COLORS['green'], NATURE_COLORS['orange'], NATURE_COLORS['red']]
    
    ax_c.bar(range(len(metrics)), values, color=colors, edgecolor='white', width=0.6)
    ax_c.set_xticks(range(len(metrics)))
    ax_c.set_xticklabels(metrics, fontsize=8)
    ax_c.set_ylabel('Value')
    ax_c.set_title('Class Separation Metrics', fontsize=10, fontweight='bold')
    
    for i, v in enumerate(values):
        ax_c.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=8, fontweight='bold')
    
    add_panel_label(ax_c, 'c')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf',
               bbox_inches='tight', facecolor='white')
    plt.close()
