# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""耦合强度与尺度的联合消融：热图、最优分析、综合面板。"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
from .nature_style import (
    set_nature_style, add_panel_label,
    NATURE_COLORS, styled_colorbar
)
from ..utils.console import console


def plot_coupling_scale_heatmap(
    df: pd.DataFrame, 
    output_dir: Path, 
    config: Dict[str, Any]
):
    """耦合×尺度准确率热图及衍生图。"""
    set_nature_style()
    
    console.print("[cyan]生成耦合–尺度热图[/cyan]")
    
    vis_cfg = config.get("visualization", {})
    dpi = vis_cfg.get("dpi", 300)
    
    pivot_table = df.pivot_table(
        index='Scale',
        columns='CouplingStrength',
        values='Accuracy',
        aggfunc='mean'
    )
    
    n_scales = len(pivot_table.index)
    n_couplings = len(pivot_table.columns)
    
    fig_width = max(16, n_couplings * 0.4)
    fig_height = max(6, n_scales * 0.6)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    annot_matrix = pivot_table.map(lambda x: f'{x:.2f}' if x >= 0.9 else "")
    
    im = sns.heatmap(
        pivot_table,
        cmap='RdYlGn',
        annot=annot_matrix,
        fmt="",
        annot_kws={'size': 7, 'weight': 'bold'},
        ax=ax,
        cbar_kws={'label': 'Accuracy', 'shrink': 0.6},
        xticklabels=max(1, n_couplings // 20),
        yticklabels=True,
        linewidths=0.1,
        linecolor='white',
        vmin=0.4,
        vmax=1.0
    )
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.5)
    
    ax.set_title('Accuracy Heatmap: Coupling Strength × Scale', 
                fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Coupling Strength', fontsize=10)
    ax.set_ylabel('Scale Index', fontsize=10)
    
    xtick_labels = ax.get_xticklabels()
    ax.set_xticklabels([f'{float(t.get_text()):.2f}' for t in xtick_labels], 
                       rotation=45, ha='right', fontsize=7)
    ax.tick_params(axis='y', labelsize=8)
    
    max_acc = pivot_table.max().max()
    best_positions = np.where(pivot_table.values >= max_acc - 0.01)
    for i, j in zip(best_positions[0], best_positions[1]):
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                   edgecolor=NATURE_COLORS['red'], linewidth=2))
    
    add_panel_label(ax, 'a')
    
    plt.tight_layout()
    
    output_path = output_dir / "Coupling_Scale_Accuracy_Heatmap.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / "Coupling_Scale_Accuracy_Heatmap.pdf", 
               format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    
    console.print(f"[green]热图已保存：{output_path}[/green]")
    
    console.print("[cyan]生成最优耦合强度分析图[/cyan]")
    
    best_coupling_per_scale = []
    for scale_idx in sorted(pivot_table.index):
        row = pivot_table.loc[scale_idx]
        best_acc = row.max()
        best_c = row.idxmax()
        best_coupling_per_scale.append({
            'Scale': scale_idx,
            'BestCoupling': best_c,
            'BestAccuracy': best_acc
        })
    
    df_best = pd.DataFrame(best_coupling_per_scale)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1 = axes[0]
    ax1.bar(df_best['Scale'], df_best['BestCoupling'], 
                    color=NATURE_COLORS['blue'], edgecolor='white', 
                    linewidth=0.5, alpha=0.85)
    
    ax1.set_xlabel('Scale Index', fontsize=10)
    ax1.set_ylabel('Optimal Coupling Strength', fontsize=10)
    ax1.set_title('Optimal Coupling Strength per Scale', fontsize=11, fontweight='bold')
    ax1.set_xticks(df_best['Scale'])
    ax1.set_ylim([0, df_best['BestCoupling'].max() * 1.2])
    
    z = np.polyfit(df_best['Scale'], df_best['BestCoupling'], 1)
    p = np.poly1d(z)
    ax1.plot(df_best['Scale'], p(df_best['Scale']), '--', 
            color=NATURE_COLORS['red'], linewidth=1.5, alpha=0.7,
            label=f'Trend (slope={z[0]:.3f})')
    ax1.legend(fontsize=8, loc='upper right')
    
    for i, row in df_best.iterrows():
        ax1.text(row['Scale'], row['BestCoupling'] + 0.02, 
                f"{row['BestCoupling']:.2f}", ha='center', fontsize=7)
    
    add_panel_label(ax1, 'b')
    
    ax2 = axes[1]
    
    colors2 = [NATURE_COLORS['green'] if acc >= 0.9 else 
               NATURE_COLORS['orange'] if acc >= 0.8 else NATURE_COLORS['red'] 
               for acc in df_best['BestAccuracy']]
    
    ax2.bar(df_best['Scale'], df_best['BestAccuracy'], 
                    color=colors2, edgecolor='white', linewidth=0.5)
    
    ax2.set_xlabel('Scale Index', fontsize=10)
    ax2.set_ylabel('Best Accuracy', fontsize=10)
    ax2.set_title('Best Accuracy per Scale', fontsize=11, fontweight='bold')
    ax2.set_xticks(df_best['Scale'])
    
    y_min = max(0.5, df_best['BestAccuracy'].min() - 0.05)
    y_max = min(1.02, df_best['BestAccuracy'].max() + 0.05)
    ax2.set_ylim([y_min, y_max])
    
    mean_acc = df_best['BestAccuracy'].mean()
    ax2.axhline(y=mean_acc, color=NATURE_COLORS['purple'], linestyle='--', 
               linewidth=1.2, label=f'Mean: {mean_acc:.3f}')
    ax2.legend(fontsize=8, loc='lower right')
    
    for i, row in df_best.iterrows():
        ax2.text(row['Scale'], row['BestAccuracy'] + 0.005, 
                f"{row['BestAccuracy']:.3f}", ha='center', fontsize=7, fontweight='bold')
    
    add_panel_label(ax2, 'c')
    
    plt.tight_layout()
    
    output_path_best = output_dir / "Coupling_Scale_Best_Analysis.png"
    plt.savefig(output_path_best, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / "Coupling_Scale_Best_Analysis.pdf", 
               format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    
    console.print(f"[green]最优耦合分析图已保存：{output_path_best}[/green]")
    
    df_best.to_csv(output_dir / "best_coupling_per_scale.csv", index=False)
    
    _plot_coupling_scale_dashboard(pivot_table, df_best, output_dir, dpi)


def _plot_coupling_scale_dashboard(
    pivot_table: pd.DataFrame,
    df_best: pd.DataFrame,
    output_dir: Path,
    dpi: int = 300
):
    """耦合–尺度综合面板。"""
    set_nature_style()
    
    console.print("[cyan]生成耦合–尺度综合面板[/cyan]")
    
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3,
                          width_ratios=[1.5, 1, 1])
    
    ax_a = fig.add_subplot(gs[:, 0])
    
    step = max(1, len(pivot_table.columns) // 30)
    pivot_sampled = pivot_table.iloc[:, ::step]
    
    im = ax_a.imshow(pivot_sampled.values, cmap='RdYlGn', aspect='auto',
                    vmin=0.4, vmax=1.0, interpolation='nearest')
    
    ax_a.set_xticks(np.arange(0, len(pivot_sampled.columns), 5))
    ax_a.set_xticklabels([f'{pivot_sampled.columns[i]:.2f}' 
                          for i in range(0, len(pivot_sampled.columns), 5)],
                        rotation=45, ha='right', fontsize=7)
    ax_a.set_yticks(range(len(pivot_sampled.index)))
    ax_a.set_yticklabels(pivot_sampled.index, fontsize=8)
    ax_a.set_xlabel('Coupling Strength', fontsize=9)
    ax_a.set_ylabel('Scale Index', fontsize=9)
    ax_a.set_title('Accuracy Landscape', fontsize=10, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax_a, shrink=0.6, pad=0.02)
    cbar.set_label('Accuracy', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    
    add_panel_label(ax_a, 'a', x=-0.15)
    
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.bar(df_best['Scale'], df_best['BestCoupling'], 
            color=NATURE_COLORS['blue'], edgecolor='white', width=0.7)
    ax_b.set_xlabel('Scale', fontsize=9)
    ax_b.set_ylabel('Optimal ε', fontsize=9)
    ax_b.set_title('Optimal Coupling', fontsize=10, fontweight='bold')
    add_panel_label(ax_b, 'b')
    
    ax_c = fig.add_subplot(gs[0, 2])
    colors = [NATURE_COLORS['green'] if acc >= 0.9 else 
              NATURE_COLORS['orange'] if acc >= 0.8 else NATURE_COLORS['gray'] 
              for acc in df_best['BestAccuracy']]
    ax_c.bar(df_best['Scale'], df_best['BestAccuracy'], 
            color=colors, edgecolor='white', width=0.7)
    ax_c.set_xlabel('Scale', fontsize=9)
    ax_c.set_ylabel('Best Acc.', fontsize=9)
    ax_c.set_ylim([0.5, 1.02])
    ax_c.set_title('Best Performance', fontsize=10, fontweight='bold')
    add_panel_label(ax_c, 'c')
    
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.scatter(df_best['BestCoupling'], df_best['BestAccuracy'], 
                c=df_best['Scale'], cmap='viridis', s=80, edgecolors='white',
                linewidths=0.5, alpha=0.85)
    ax_d.set_xlabel('Optimal Coupling', fontsize=9)
    ax_d.set_ylabel('Best Accuracy', fontsize=9)
    ax_d.set_title('Coupling vs. Accuracy', fontsize=10, fontweight='bold')
    
    corr = np.corrcoef(df_best['BestCoupling'], df_best['BestAccuracy'])[0, 1]
    ax_d.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax_d.transAxes,
             fontsize=9, va='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    add_panel_label(ax_d, 'd')
    
    ax_e = fig.add_subplot(gs[1, 2])
    ax_e.axis('off')
    
    summary_text = f"""
    摘要统计
    ────────
    尺度数 {len(df_best)}
    准确率 均值 {df_best['BestAccuracy'].mean():.4f}  标准差 {df_best['BestAccuracy'].std():.4f}
    准确率 最大 {df_best['BestAccuracy'].max():.4f}  最小 {df_best['BestAccuracy'].min():.4f}
    最优耦合 均值 {df_best['BestCoupling'].mean():.3f}  范围 [{df_best['BestCoupling'].min():.2f}, {df_best['BestCoupling'].max():.2f}]
    最佳尺度 {df_best.loc[df_best['BestAccuracy'].idxmax(), 'Scale']}
    """
    
    ax_e.text(0.1, 0.9, summary_text, transform=ax_e.transAxes,
             fontsize=9, va='top', ha='left', family='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', 
                      edgecolor=NATURE_COLORS['border']))
    add_panel_label(ax_e, 'e')
    
    plt.savefig(output_dir / "Coupling_Scale_Dashboard.png", dpi=dpi,
               bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / "Coupling_Scale_Dashboard.pdf", 
               format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    
    console.print("[green]综合面板已保存[/green]")
