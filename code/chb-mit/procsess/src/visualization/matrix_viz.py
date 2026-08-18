# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""
矩阵可视化模块 (Nature Journal Style)
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List
from .nature_style import set_nature_style, add_panel_label, NATURE_COLORS, styled_colorbar


def plot_adjacency_matrix(
    matrix: np.ndarray, 
    output_path: str, 
    title: str = "Adjacency Matrix",
    cmap: str = 'viridis',
    labels: Optional[List[str]] = None,
    show_values: bool = False
):
    """绘制邻接矩阵热力图 (Nature Style)"""
    set_nature_style()
    
    n = matrix.shape[0]
    fig_size = max(6, min(12, n * 0.4))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    
    vmin, vmax = 0, matrix.max()
    im = ax.imshow(matrix, cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax)
    
    if show_values and n <= 15:
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                text_color = 'white' if val > (vmax + vmin) / 2 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=7)
    
    ax.set_xlabel('Node Index', fontsize=10)
    ax.set_ylabel('Node Index', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    styled_colorbar(im, ax, label='Weight', shrink=0.8)
    add_panel_label(ax, 'a')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_connectivity_graph(
    matrix: np.ndarray, 
    output_path: str, 
    title: str = "Network Connectivity",
    layout: str = 'spring',
    threshold: float = 0.1
):
    """绘制连接图 (Nature Style)"""
    set_nature_style()
    
    try:
        import networkx as nx
        n = matrix.shape[0]
        G = nx.Graph()
        for i in range(n):
            G.add_node(i)
        for i in range(n):
            for j in range(i + 1, n):
                if matrix[i, j] > threshold:
                    G.add_edge(i, j, weight=matrix[i, j])
        
        fig, ax = plt.subplots(figsize=(8, 8))
        pos = nx.spring_layout(G, seed=42, k=2/np.sqrt(n)) if layout == 'spring' else nx.circular_layout(G)
        
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.4)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=NATURE_COLORS['blue'],
                               node_size=300, edgecolors='white', linewidths=1.5)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color='white', font_weight='bold')
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
        add_panel_label(ax, 'a', x=0.02, y=0.98)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    except ImportError:
        print("Warning: networkx not installed")
