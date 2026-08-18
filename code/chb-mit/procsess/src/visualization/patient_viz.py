# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""患者级准确率、分类器热图、雷达、样本分布与报告。"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from .nature_style import (
    set_nature_style, create_figure, save_figure, add_panel_label,
    NATURE_COLORS, PALETTE_MULTI, CMAPS,
    nature_barplot, nature_boxplot, nature_heatmap, styled_colorbar,
    add_significance_annotation
)
from ..utils.console import console

matplotlib.use('Agg')


def plot_patient_accuracy_bar(
    results_df: pd.DataFrame, 
    output_path: str,
    show_error: bool = True
):
    """各患者分类准确率柱状图。"""
    set_nature_style()
    
    if 'accuracy' not in results_df.columns:
        console.print("[yellow]缺少 accuracy 列，已跳过[/yellow]")
        return
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    patients = results_df['patient_id'].values
    values = results_df['accuracy'].values
    n_patients = len(patients)
    
    colors = []
    for v in values:
        if v >= 0.8:
            colors.append(NATURE_COLORS['green'])
        elif v >= 0.6:
            colors.append(NATURE_COLORS['orange'])
        else:
            colors.append(NATURE_COLORS['red'])
    
    x_pos = np.arange(n_patients)
    bars = ax.bar(x_pos, values, color=colors, edgecolor='white', linewidth=0.5, width=0.7)
    
    for i, (bar, val) in enumerate(zip(bars, values)):
        if val >= 0.9 or val <= 0.5:
            ax.annotate(f'{val:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=6, fontweight='bold')
    
    mean_acc = np.mean(values)
    std_acc = np.std(values)
    ax.axhline(y=mean_acc, color=NATURE_COLORS['purple'], linestyle='-', linewidth=1.5, 
               label=f'Mean: {mean_acc:.3f} ± {std_acc:.3f}')
    
    if show_error:
        ax.axhspan(mean_acc - std_acc, mean_acc + std_acc, 
                   alpha=0.15, color=NATURE_COLORS['purple'])

    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7, label='Chance level')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(patients, rotation=45, ha='right', fontsize=7)
    ax.set_xlabel('Patient ID', fontsize=9)
    ax.set_ylabel('Accuracy', fontsize=9)
    ax.set_title('Per-Patient Classification Accuracy', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.08)
    ax.legend(loc='upper right', fontsize=7, frameon=True, fancybox=False, 
              edgecolor=NATURE_COLORS['border'])
    
    add_panel_label(ax, 'a')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    console.print(f"[green]已保存：{output_path}[/green]")


def plot_classifier_comparison_heatmap(
    results_df: pd.DataFrame, 
    output_path: str
):
    """各分类器在各患者上的准确率热图。"""
    set_nature_style()
    
    patients = results_df['patient_id'].values
    classifiers = ['SVM', 'RandomForest', 'XGBoost', 'LightGBM', 'KNN', 'LR', 'GBDT']
    
    heatmap_data = []
    valid_classifiers = []
    
    for clf in classifiers:
        col_name = None
        for c in [f'{clf}_accuracy', f'{clf}_Acc', f'ALL_{clf}_accuracy', f'ALL_{clf}_Acc']:
            if c in results_df.columns:
                col_name = c
                break
        
        if col_name:
            vals = results_df[col_name].values
            heatmap_data.append(vals)
            valid_classifiers.append(clf)
    
    if not heatmap_data:
        console.print("[yellow]未找到分类器准确率列，已跳过[/yellow]")
        return
    
    heatmap_data = np.array(heatmap_data)
    
    fig_width = max(10, len(patients) * 0.5)
    fig_height = max(4, len(valid_classifiers) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    im = ax.imshow(heatmap_data, cmap=CMAPS['accuracy'], aspect='auto', vmin=0.4, vmax=1.0)
    
    cbar = styled_colorbar(im, ax, label='Accuracy', shrink=0.7)
    
    for i in range(len(valid_classifiers)):
        for j in range(len(patients)):
            val = heatmap_data[i, j]
            if val >= 0.9 or val <= 0.5:
                text_color = 'white' if val < 0.6 or val > 0.85 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       color=text_color, fontsize=6, fontweight='bold')
    
    ax.set_xticks(range(len(patients)))
    ax.set_xticklabels(patients, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(valid_classifiers)))
    ax.set_yticklabels(valid_classifiers, fontsize=8)
    ax.set_xlabel('Patient ID', fontsize=9)
    ax.set_ylabel('Classifier', fontsize=9)
    ax.set_title('Classifier Performance Across Patients', fontsize=10, fontweight='bold')
    
    add_panel_label(ax, 'b')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    console.print(f"[green]已保存：{output_path}[/green]")


def plot_metrics_radar(
    results_df: pd.DataFrame, 
    output_path: str
):
    """各指标雷达图（跨患者平均）。"""
    set_nature_style()
    
    metrics = ['accuracy', 'sensitivity', 'specificity', 'f1_score', 'auc']
    classifiers = ['SVM', 'RandomForest', 'XGBoost', 'LightGBM']
    
    data = {}
    for clf in classifiers:
        clf_data = []
        has_data = False
        for metric in metrics:
            col_found = False
            for c in [f'{clf}_{metric}', f'ALL_{clf}_{metric}']:
                if c in results_df.columns:
                    clf_data.append(results_df[c].mean())
                    col_found = True
                    has_data = True
                    break
            if not col_found:
                clf_data.append(0)
        
        if has_data and np.sum(clf_data) > 0:
            data[clf] = clf_data
    
    if not data:
        console.print("[yellow]雷达图所需列缺失，已跳过[/yellow]")
        return
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    colors = [NATURE_COLORS['red'], NATURE_COLORS['blue'], 
              NATURE_COLORS['green'], NATURE_COLORS['orange']]
    
    for i, (clf, values) in enumerate(data.items()):
        values = values + values[:1]
        color = colors[i % len(colors)]
        ax.plot(angles, values, 'o-', linewidth=1.5, label=clf, color=color, markersize=4)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', '\n').title() for m in metrics], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title('Average Performance Across Patients', fontsize=10, fontweight='bold', pad=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.05), fontsize=7, frameon=False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    console.print(f"[green]已保存：{output_path}[/green]")


def plot_sample_distribution(
    results_df: pd.DataFrame, 
    output_path: str
):
    """各患者两类样本数分布。"""
    set_nature_style()
    
    if 'n_class0' not in results_df.columns or 'n_class1' not in results_df.columns:
        console.print("[yellow]缺少样本计数列，已跳过[/yellow]")
        return
    
    fig, ax = plt.subplots(figsize=(12, 4.5))
    
    patients = results_df['patient_id'].values
    class0 = results_df['n_class0'].values
    class1 = results_df['n_class1'].values
    
    x = np.arange(len(patients))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, class0, width, label='Interictal', 
                   color=NATURE_COLORS['blue'], edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, class1, width, label='Preictal', 
                   color=NATURE_COLORS['red'], edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Patient ID', fontsize=9)
    ax.set_ylabel('Number of Samples', fontsize=9)
    ax.set_title('Sample Distribution per Patient', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(patients, rotation=45, ha='right', fontsize=7)
    ax.legend(fontsize=8, frameon=False)
    
    for i, (c0, c1) in enumerate(zip(class0, class1)):
        ratio = c1 / (c0 + c1) if (c0 + c1) > 0 else 0
        if ratio < 0.3 or ratio > 0.7:
            max_height = max(c0, c1)
            ax.text(i, max_height + 5, f'{ratio:.0%}', ha='center', fontsize=5, 
                   color=NATURE_COLORS['red'] if ratio < 0.3 else NATURE_COLORS['green'])
    
    add_panel_label(ax, 'c')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    console.print(f"[green]已保存：{output_path}[/green]")


def plot_boxplot_metrics(
    results_df: pd.DataFrame, 
    output_path: str
):
    """
    绘制各指标的箱线图 (Nature Style)
    """
    set_nature_style()
    
    metrics = ['accuracy', 'sensitivity', 'specificity', 'f1_score', 'auc']
    
    data_to_plot = []
    labels = []
    
    for metric in metrics:
        if metric in results_df.columns:
            values = results_df[metric].dropna().values
            if len(values) > 0:
                data_to_plot.append(values)
                labels.append(metric.replace('_', '\n').title())
    
    if not data_to_plot:
        console.print("[yellow]Warning: No metric columns found for boxplot[/yellow]")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = [NATURE_COLORS['blue'], NATURE_COLORS['green'], NATURE_COLORS['purple'],
              NATURE_COLORS['orange'], NATURE_COLORS['red']]
    
    # 使用 Nature boxplot
    ax, bp = nature_boxplot(data_to_plot, ax=ax, labels=labels, 
                            colors=colors[:len(data_to_plot)], show_points=True)
    
    ax.set_ylabel('Score', fontsize=9)
    ax.set_title('Distribution of Metrics Across Patients', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    
    add_panel_label(ax, 'd')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    console.print(f"[green]✓ Saved: {output_path}[/green]")


def plot_performance_summary(
    results_df: pd.DataFrame,
    output_path: str
):
    """
    绘制综合性能总结图 (2x2 组合图)
    
    包含:
    - a. 患者准确率柱状图
    - b. 分类器热图
    - c. 样本分布
    - d. 指标箱线图
    """
    set_nature_style()
    
    fig = plt.figure(figsize=(14, 10))
    
    # 创建 2x2 网格
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # === Panel a: 准确率柱状图 ===
    ax_a = fig.add_subplot(gs[0, 0])
    if 'accuracy' in results_df.columns:
        patients = results_df['patient_id'].values
        values = results_df['accuracy'].values
        colors = [NATURE_COLORS['green'] if v >= 0.8 else 
                  NATURE_COLORS['orange'] if v >= 0.6 else NATURE_COLORS['red'] 
                  for v in values]
        
        ax_a.bar(range(len(patients)), values, color=colors, edgecolor='white', width=0.7)
        ax_a.axhline(y=np.mean(values), color=NATURE_COLORS['purple'], linestyle='-', 
                    linewidth=1.2, label=f'Mean: {np.mean(values):.3f}')
        ax_a.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_a.set_xticks(range(len(patients)))
        ax_a.set_xticklabels(patients, rotation=45, ha='right', fontsize=6)
        ax_a.set_ylabel('Accuracy')
        ax_a.set_ylim(0, 1.08)
        ax_a.legend(fontsize=7, loc='upper right')
    ax_a.set_title('Per-Patient Accuracy', fontsize=9, fontweight='bold')
    add_panel_label(ax_a, 'a')
    
    # === Panel b: 分类器性能雷达 ===
    ax_b = fig.add_subplot(gs[0, 1], polar=True)
    metrics = ['accuracy', 'sensitivity', 'specificity', 'f1_score', 'auc']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    # 尝试获取最佳分类器数据
    best_clf = None
    best_vals = []
    for clf in ['RandomForest', 'XGBoost', 'SVM']:
        vals = []
        for m in metrics:
            for col in [f'{clf}_{m}', f'ALL_{clf}_{m}']:
                if col in results_df.columns:
                    vals.append(results_df[col].mean())
                    break
            else:
                vals.append(np.nan)
        if not any(np.isnan(vals)):
            best_clf = clf
            best_vals = vals
            break
    
    if best_vals:
        best_vals += best_vals[:1]
        ax_b.plot(angles, best_vals, 'o-', linewidth=1.5, color=NATURE_COLORS['red'], 
                 markersize=4, label=best_clf)
        ax_b.fill(angles, best_vals, alpha=0.15, color=NATURE_COLORS['red'])
        ax_b.set_xticks(angles[:-1])
        ax_b.set_xticklabels([m.replace('_', '\n').title() for m in metrics], fontsize=7)
        ax_b.set_ylim(0, 1.0)
        ax_b.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax_b.set_title('Best Classifier Profile', fontsize=9, fontweight='bold', pad=10)
    fig.text(0.52, 0.95, 'b', fontsize=12, fontweight='bold', color=NATURE_COLORS['text'])
    
    # === Panel c: 样本分布 ===
    ax_c = fig.add_subplot(gs[1, 0])
    if 'n_class0' in results_df.columns and 'n_class1' in results_df.columns:
        patients = results_df['patient_id'].values
        class0 = results_df['n_class0'].values
        class1 = results_df['n_class1'].values
        x = np.arange(len(patients))
        width = 0.35
        ax_c.bar(x - width/2, class0, width, label='Interictal', color=NATURE_COLORS['blue'])
        ax_c.bar(x + width/2, class1, width, label='Preictal', color=NATURE_COLORS['red'])
        ax_c.set_xticks(x)
        ax_c.set_xticklabels(patients, rotation=45, ha='right', fontsize=6)
        ax_c.set_ylabel('Samples')
        ax_c.legend(fontsize=7, loc='upper right')
    ax_c.set_title('Sample Distribution', fontsize=9, fontweight='bold')
    add_panel_label(ax_c, 'c')
    
    # === Panel d: 指标箱线图 ===
    ax_d = fig.add_subplot(gs[1, 1])
    data_to_plot = []
    labels = []
    colors_box = []
    color_list = [NATURE_COLORS['blue'], NATURE_COLORS['green'], NATURE_COLORS['purple'],
                  NATURE_COLORS['orange'], NATURE_COLORS['red']]
    
    for i, metric in enumerate(metrics):
        if metric in results_df.columns:
            vals = results_df[metric].dropna().values
            if len(vals) > 0:
                data_to_plot.append(vals)
                labels.append(metric.replace('_', '\n').title())
                colors_box.append(color_list[i % len(color_list)])
    
    if data_to_plot:
        bp = ax_d.boxplot(data_to_plot, patch_artist=True, labels=labels)
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax_d.set_ylabel('Score')
        ax_d.set_ylim(0, 1.1)
    ax_d.set_title('Metric Distributions', fontsize=9, fontweight='bold')
    add_panel_label(ax_d, 'd')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    console.print(f"[green]✓ Saved: {output_path}[/green]")


def generate_markdown_report(
    results_df: pd.DataFrame, 
    output_dir: str, 
    config: Dict[str, Any]
):
    """生成 Markdown 报告"""
    report_path = os.path.join(output_dir, "analysis_report.md")
    
    n_patients = len(results_df)
    
    if 'error' not in results_df.columns:
        results_df['error'] = ''
    
    success_df = results_df[results_df['error'] == '']
    n_success = len(success_df)
    n_failed = n_patients - n_success
    
    if n_success > 0 and 'accuracy' in success_df.columns:
        mean_acc = success_df['accuracy'].mean()
        std_acc = success_df['accuracy'].std()
    else:
        mean_acc = std_acc = 0
    
    md = f"""# Per-Patient Analysis Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Patients | {n_patients} |
| Successful | {n_success} |
| Failed | {n_failed} |
| Mean Accuracy | {mean_acc:.4f} ± {std_acc:.4f} |

## Visualizations

### Performance Overview
![Summary](Fig0_performance_summary.png)

### Individual Figures
| Figure | Description |
|--------|-------------|
| ![Accuracy](Fig1_patient_accuracy.png) | Per-patient classification accuracy |
| ![Samples](Fig2_sample_distribution.png) | Sample distribution per patient |
| ![Heatmap](Fig3_classifier_heatmap.png) | Classifier comparison heatmap |
| ![Boxplot](Fig4_metrics_boxplot.png) | Metric distribution boxplots |
| ![Radar](Fig5_metrics_radar.png) | Multi-metric radar chart |

## Configuration

```yaml
{config if isinstance(config, str) else 'See config file'}
```

---
*Generated by 3M EEG Analysis Framework*
"""
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    
    console.print(f"[green]✓ Report saved: {report_path}[/green]")


def generate_patient_visualizations(
    results_df: pd.DataFrame, 
    output_dir: str, 
    config: Dict[str, Any]
):
    """
    生成所有患者分析可视化 (Nature Journal Style)
    """
    console.print("\n[bold cyan]═══ Generating Nature-Style Visualizations ═══[/bold cyan]")
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成各图表
    plot_performance_summary(results_df, os.path.join(output_dir, "Fig0_performance_summary.png"))
    plot_patient_accuracy_bar(results_df, os.path.join(output_dir, "Fig1_patient_accuracy.png"))
    plot_sample_distribution(results_df, os.path.join(output_dir, "Fig2_sample_distribution.png"))
    plot_classifier_comparison_heatmap(results_df, os.path.join(output_dir, "Fig3_classifier_heatmap.png"))
    plot_boxplot_metrics(results_df, os.path.join(output_dir, "Fig4_metrics_boxplot.png"))
    plot_metrics_radar(results_df, os.path.join(output_dir, "Fig5_metrics_radar.png"))
    
    # 生成报告
    generate_markdown_report(results_df, output_dir, config)
    
    console.print("[bold green]✓ All visualizations generated successfully![/bold green]")
