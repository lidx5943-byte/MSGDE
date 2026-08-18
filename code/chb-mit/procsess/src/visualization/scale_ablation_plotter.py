# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""尺度消融：热图、趋势折线、综合四宫格。"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
from .nature_style import (
    set_nature_style, add_panel_label,
    NATURE_COLORS, PALETTE_MULTI, styled_colorbar
)
from ..utils.console import console


def plot_scale_ablation_results(
    df: pd.DataFrame, 
    output_dir: Path, 
    config: Dict[str, Any]
):
    """按模型绘制尺度×指标热图，并输出尺度趋势折线。"""
    set_nature_style()
    
    console.print("[cyan]生成尺度消融热图[/cyan]")
    
    metrics = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        console.print("[yellow]未找到指标列，已跳过[/yellow]")
        return
    
    models = df['Model'].unique()
    cmap = config.get("visualization", {}).get("colormap", "YlGnBu")
    
    for model_name in models:
        df_model = df[df['Model'] == model_name].copy()
        df_model = df_model.sort_values('Scale')
        
        plot_data = df_model.set_index('Scale')[available_metrics].T
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        im = ax.imshow(plot_data.values, cmap=cmap, aspect='auto', vmin=0.4, vmax=1.0)
        
        for i in range(len(available_metrics)):
            for j in range(len(plot_data.columns)):
                val = plot_data.iloc[i, j]
                text_color = 'white' if val > 0.85 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                       color=text_color, fontsize=8)
        
        ax.set_xticks(range(len(plot_data.columns)))
        ax.set_xticklabels(plot_data.columns, fontsize=9)
        ax.set_yticks(range(len(available_metrics)))
        ax.set_yticklabels(available_metrics, fontsize=9)
        
        ax.set_xlabel('Scale Index', fontsize=10)
        ax.set_ylabel('Metric', fontsize=10)
        ax.set_title(f'Performance Metrics across Scales - {model_name}', 
                    fontsize=11, fontweight='bold')
        
        cbar = styled_colorbar(im, ax, label='Score', shrink=0.8)
        add_panel_label(ax, 'a')
        
        plt.tight_layout()
        filename = f"Scale_Ablation_Heatmap_{model_name}"
        plt.savefig(output_dir / f"{filename}.png", dpi=300, 
                   bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / f"{filename}.pdf", format='pdf',
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    console.print(f"[green]已为 {len(models)} 个模型生成尺度热图[/green]")
    
    _plot_scale_trend(df, available_metrics[0], output_dir)


def _plot_scale_trend(df: pd.DataFrame, metric: str, output_dir: Path):
    """尺度–指标折线（多模型）。"""
    set_nature_style()
    
    console.print("[cyan]生成尺度趋势折线图[/cyan]")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    models = df['Model'].unique()
    colors = PALETTE_MULTI[:len(models)]
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    for i, model in enumerate(models):
        df_model = df[df['Model'] == model].sort_values('Scale')
        ax.plot(df_model['Scale'], df_model[metric], 
               marker=markers[i % len(markers)], 
               color=colors[i % len(colors)],
               linewidth=1.5, markersize=5, alpha=0.85,
               label=model)
    
    ax.set_xlabel('Scale Index', fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(f'{metric} across Scales for Different Classifiers', 
                fontsize=11, fontweight='bold')
    ax.set_xticks(df['Scale'].unique())
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    
    ax.legend(title='Classifier', fontsize=8, loc='best',
             frameon=True, fancybox=False, edgecolor=NATURE_COLORS['border'])
    
    add_panel_label(ax, 'b')
    
    plt.tight_layout()
    plt.savefig(output_dir / "Scale_Ablation_Accuracy_Trend.png", dpi=300,
               bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / "Scale_Ablation_Accuracy_Trend.pdf", format='pdf',
               bbox_inches='tight', facecolor='white')
    plt.close()
    
    console.print("[green]尺度趋势图已保存[/green]")


def plot_scale_summary(
    df: pd.DataFrame,
    output_dir: Path,
    config: Dict[str, Any]
):
    """尺度消融四宫格综合图。"""
    set_nature_style()
    
    console.print("[cyan]生成尺度消融综合图[/cyan]")
    
    metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        return
    
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    ax_a = fig.add_subplot(gs[0, 0])
    
    best_model = df.groupby('Scale')['Accuracy'].max().reset_index()
    best_model.columns = ['Scale', 'Best_Accuracy']
    
    colors = [NATURE_COLORS['green'] if acc >= 0.9 else 
              NATURE_COLORS['orange'] if acc >= 0.8 else NATURE_COLORS['gray']
              for acc in best_model['Best_Accuracy']]
    
    ax_a.bar(best_model['Scale'], best_model['Best_Accuracy'], 
            color=colors, edgecolor='white', width=0.7)
    ax_a.set_xlabel('Scale', fontsize=9)
    ax_a.set_ylabel('Best Accuracy', fontsize=9)
    ax_a.set_title('Best Performance per Scale', fontsize=10, fontweight='bold')
    ax_a.set_ylim([0.5, 1.02])
    
    for i, row in best_model.iterrows():
        ax_a.text(row['Scale'], row['Best_Accuracy'] + 0.01, 
                 f'{row["Best_Accuracy"]:.3f}', ha='center', fontsize=7)
    
    add_panel_label(ax_a, 'a')
    
    ax_b = fig.add_subplot(gs[0, 1])
    models = df['Model'].unique()[:4]
    
    for i, model in enumerate(models):
        df_m = df[df['Model'] == model].sort_values('Scale')
        ax_b.plot(df_m['Scale'], df_m['Accuracy'], 'o-',
                 color=PALETTE_MULTI[i], linewidth=1.5, markersize=4, label=model)
    
    ax_b.set_xlabel('Scale', fontsize=9)
    ax_b.set_ylabel('Accuracy', fontsize=9)
    ax_b.set_title('Classifier Comparison', fontsize=10, fontweight='bold')
    ax_b.legend(fontsize=7, loc='lower right')
    ax_b.grid(True, linestyle='--', alpha=0.3)
    add_panel_label(ax_b, 'b')
    
    ax_c = fig.add_subplot(gs[1, 0])
    
    best_scales = df.loc[df.groupby('Model')['Accuracy'].idxmax()][['Model', 'Scale', 'Accuracy']]
    
    ax_c.barh(range(len(best_scales)), best_scales['Scale'].values,
             color=NATURE_COLORS['blue'], edgecolor='white', height=0.6)
    ax_c.set_yticks(range(len(best_scales)))
    ax_c.set_yticklabels(best_scales['Model'].values, fontsize=8)
    ax_c.set_xlabel('Optimal Scale', fontsize=9)
    ax_c.set_title('Optimal Scale per Classifier', fontsize=10, fontweight='bold')
    add_panel_label(ax_c, 'c')
    
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis('off')
    
    summary = f"""
    尺度消融摘要
    ────────────
    尺度数: {len(df['Scale'].unique())}
    模型数: {len(df['Model'].unique())}
    准确率 均值 {df['Accuracy'].mean():.4f}  标准差 {df['Accuracy'].std():.4f}
    准确率 最大 {df['Accuracy'].max():.4f}  最小 {df['Accuracy'].min():.4f}
    最优尺度 {df.loc[df['Accuracy'].idxmax(), 'Scale']}  模型 {df.loc[df['Accuracy'].idxmax(), 'Model']}  Acc {df['Accuracy'].max():.4f}
    """
    
    ax_d.text(0.1, 0.9, summary, transform=ax_d.transAxes,
             fontsize=9, va='top', family='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa',
                      edgecolor=NATURE_COLORS['border']))
    add_panel_label(ax_d, 'd')
    
    plt.savefig(output_dir / "Scale_Ablation_Summary.png", dpi=300,
               bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / "Scale_Ablation_Summary.pdf", format='pdf',
               bbox_inches='tight', facecolor='white')
    plt.close()
    
    console.print("[green]尺度综合图已保存[/green]")
