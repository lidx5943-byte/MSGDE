# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""
微扰可视化模块
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
from typing import List, Optional
from .nature_style import set_nature_style, add_panel_label, NATURE_COLORS, styled_colorbar


def plot_perturbation_heatmap(
    perturbation_matrix: np.ndarray,
    output_path: str,
    node_labels: Optional[List[str]] = None,
    metric_names: Optional[List[str]] = None,
    title: str = "Perturbation Impact Analysis"
):
    """
    绘制微扰影响热力图 (Nature Style)
    
    Parameters
    ----------
    perturbation_matrix : ndarray, shape (N_nodes, N_metrics)
        表示每个节点移除后指标的变化量
    output_path : str
        输出路径
    node_labels : list, optional
        节点标签
    metric_names : list, optional
        指标名称
    """
    set_nature_style()
    
    n_nodes, n_metrics = perturbation_matrix.shape
    fig_width = max(8, n_metrics * 0.8)
    fig_height = max(6, n_nodes * 0.4)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # 使用发散色图 (正负变化)
    abs_max = np.abs(perturbation_matrix).max()
    im = ax.imshow(perturbation_matrix, cmap='RdBu_r', aspect='auto',
                   vmin=-abs_max, vmax=abs_max)
    
    # 设置标签
    if metric_names is not None:
        ax.set_xticks(range(n_metrics))
        ax.set_xticklabels(metric_names, rotation=45, ha='right', fontsize=8)
    
    if node_labels is not None:
        ax.set_yticks(range(n_nodes))
        ax.set_yticklabels(node_labels, fontsize=8)
    elif n_nodes <= 30:
        ax.set_yticks(range(n_nodes))
    
    ax.set_xlabel('Metric', fontsize=10)
    ax.set_ylabel('Node Index', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    # Colorbar
    cbar = styled_colorbar(im, ax, label='Change', shrink=0.7)
    
    add_panel_label(ax, 'a')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf',
               bbox_inches='tight', facecolor='white')
    plt.close()


def plot_top_critical_nodes(
    impact_scores: np.ndarray,
    output_path: str,
    top_k: int = 20,
    node_labels: Optional[List[str]] = None,
    title: str = "Critical Nodes via Perturbation"
):
    """
    绘制 Top-K 关键节点 (Nature Style)
    
    Parameters
    ----------
    impact_scores : ndarray, shape (N_nodes,)
        综合影响分数
    output_path : str
        输出路径
    top_k : int
        显示前K个节点
    node_labels : list, optional
        节点标签
    """
    set_nature_style()
    
    # 排序获取 Top-K
    indices = np.argsort(impact_scores)[::-1][:top_k]
    scores = impact_scores[indices]
    
    # 获取标签
    if node_labels is not None:
        labels = [node_labels[i] for i in indices]
    else:
        labels = [f'Node {i}' for i in indices]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 渐变色
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(scores)))
    
    bars = ax.bar(range(top_k), scores, color=colors, 
                  edgecolor='white', linewidth=0.5, width=0.7)
    
    ax.set_xticks(range(top_k))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Node', fontsize=10)
    ax.set_ylabel('Impact Score', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    # 添加数值标注 (仅前5)
    for i, (bar, score) in enumerate(zip(bars[:5], scores[:5])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{score:.3f}', ha='center', fontsize=7, fontweight='bold')
    
    # 平均线
    mean_score = np.mean(impact_scores)
    ax.axhline(y=mean_score, color=NATURE_COLORS['purple'], linestyle='--',
              linewidth=1, label=f'Mean: {mean_score:.3f}')
    ax.legend(fontsize=8, loc='upper right')
    
    add_panel_label(ax, 'a')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf',
               bbox_inches='tight', facecolor='white')
    plt.close()


def plot_perturbation_summary(
    perturbation_matrix: np.ndarray,
    output_path: str,
    metric_names: Optional[List[str]] = None,
    node_labels: Optional[List[str]] = None
):
    """
    绘制微扰分析综合图 (Nature Style)
    
    包含:
    - a. 热图
    - b. Top关键节点
    - c. 指标敏感度
    """
    set_nature_style()
    
    n_nodes, n_metrics = perturbation_matrix.shape
    
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35, width_ratios=[1.2, 1, 0.8])
    
    # === Panel a: 热图 ===
    ax_a = fig.add_subplot(gs[0])
    abs_max = np.abs(perturbation_matrix).max()
    im = ax_a.imshow(perturbation_matrix, cmap='RdBu_r', aspect='auto',
                     vmin=-abs_max, vmax=abs_max)
    
    if metric_names:
        ax_a.set_xticks(range(n_metrics))
        ax_a.set_xticklabels(metric_names, rotation=45, ha='right', fontsize=7)
    ax_a.set_ylabel('Node Index', fontsize=9)
    ax_a.set_title('Perturbation Matrix', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax_a, shrink=0.6, label='ΔMetric')
    add_panel_label(ax_a, 'a')
    
    # === Panel b: Top 关键节点 ===
    ax_b = fig.add_subplot(gs[1])
    impact_scores = np.sum(np.abs(perturbation_matrix), axis=1)
    top_k = min(15, n_nodes)
    indices = np.argsort(impact_scores)[::-1][:top_k]
    scores = impact_scores[indices]
    
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, top_k))
    ax_b.barh(range(top_k), scores, color=colors, edgecolor='white', height=0.7)
    ax_b.set_yticks(range(top_k))
    ax_b.set_yticklabels([f'Node {i}' for i in indices], fontsize=8)
    ax_b.invert_yaxis()
    ax_b.set_xlabel('Total Impact', fontsize=9)
    ax_b.set_title('Critical Nodes', fontsize=10, fontweight='bold')
    add_panel_label(ax_b, 'b')
    
    # === Panel c: 指标敏感度 ===
    ax_c = fig.add_subplot(gs[2])
    metric_sensitivity = np.std(perturbation_matrix, axis=0)
    
    if metric_names and len(metric_names) == n_metrics:
        labels = metric_names
    else:
        labels = [f'M{i}' for i in range(n_metrics)]
    
    ax_c.bar(range(n_metrics), metric_sensitivity, 
            color=NATURE_COLORS['green'], edgecolor='white', width=0.6)
    ax_c.set_xticks(range(n_metrics))
    ax_c.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax_c.set_ylabel('Sensitivity (Std)', fontsize=9)
    ax_c.set_title('Metric Sensitivity', fontsize=10, fontweight='bold')
    add_panel_label(ax_c, 'c')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf',
               bbox_inches='tight', facecolor='white')
    plt.close()
