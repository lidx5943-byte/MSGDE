# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""消融实验图表：柱状图、分组对比、热图、雷达图、ROC。"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

from .nature_style import (
    set_nature_style, create_figure, save_figure, add_panel_label,
    NATURE_COLORS, PALETTE_MULTI, CMAPS,
    nature_barplot, nature_heatmap, styled_colorbar
)
from ..utils.console import console


def load_results(base_dir: Path):
    """读取 ablation_results.csv 与 roc 数据。"""
    df_path = base_dir / "ablation_results.csv"
    roc_path = base_dir / "roc_curves_data.pkl"
    
    if df_path.exists():
        df = pd.read_csv(df_path)
        roc_data = joblib.load(roc_path) if roc_path.exists() else {}
        return df, roc_data

    summary_path = base_dir / "summary.csv"
    if summary_path.exists():
        return load_and_convert_summary(summary_path), {}
        
    return None, None


def load_and_convert_summary(summary_path: Path) -> pd.DataFrame:
    """summary.csv 转为消融绘图用 DataFrame。"""
    df_raw = pd.read_csv(summary_path)
    avg_row = df_raw[df_raw['patient_id'] == 'AVERAGE']
    
    if avg_row.empty:
        numeric_df = df_raw.select_dtypes(include=[np.number])
        avg_data = numeric_df.mean()
    else:
        avg_data = avg_row.iloc[0]
        
    models = ['SVM', 'RF', 'KNN', 'LR', 'GBDT']
    feature_sets = ['S', 'T', 'D', 'S+T', 'S+D', 'T+D', 'ALL']
    metrics = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC']
    
    records = []
    for model in models:
        for fs in feature_sets:
            record = {'Model': model, 'Feature Set': fs}
            valid = False
            for m in metrics:
                col = f"{fs}_{model}_{m}"
                if col in avg_data:
                    record[m] = avg_data[col]
                    valid = True
            if valid:
                records.append(record)
                
    return pd.DataFrame(records)


def plot_ablation_bar_comparison(
    df: pd.DataFrame, 
    output_dir: Path,
    model_name: str = 'RF'
):
    """按特征集绘制消融柱状图。"""
    set_nature_style()
    
    console.print(f"[cyan]生成消融性能对比图（模型：{model_name}）[/cyan]")
    
    df_model = df[df['Model'] == model_name].copy()
    if df_model.empty:
        model_name = df['Model'].iloc[0]
        df_model = df[df['Model'] == model_name].copy()
    
    name_map = {
        'S': 'Geometric',
        'T': 'Topological',
        'D': 'Dynamics',
        'S+T': 'Geometric\n+ Topological',
        'S+D': 'Geometric\n+ Dynamics',
        'T+D': 'Topological\n+ Dynamics',
        'ALL': 'Geometric\n+ Topological\n+ Dynamics',
        'Baseline (Raw Stat)': 'Baseline'
    }
    
    df_model['Feature Set'] = df_model['Feature Set'].map(lambda x: name_map.get(x, x))

    desired_sets = [
         'Baseline',
         'Geometric',
         'Topological',
         'Dynamics',
         'Geometric\n+ Topological',
         'Geometric\n+ Dynamics',
         'Topological\n+ Dynamics',
         'Geometric\n+ Topological\n+ Dynamics'
    ]
    
    df_plot = df_model[df_model['Feature Set'].isin(desired_sets)].copy()
    df_plot = df_plot.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
    
    if df_plot.empty:
        console.print("[yellow]未找到有效特征集，已跳过[/yellow]")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    try:
        from .nature_style import PALETTE_MULTI, NATURE_COLORS
    except ImportError:
         PALETTE_MULTI = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000']
    
    n_sets = len(df_plot)
    colors = PALETTE_MULTI[:n_sets]
    
    if len(colors) < n_sets:
        colors = colors * (n_sets // len(colors) + 1)
        colors = colors[:n_sets]

    x_pos = np.arange(n_sets)
    accuracies = df_plot['Accuracy'].values
    sets_labels = df_plot['Feature Set'].values
    
    bars = ax.bar(x_pos, accuracies, color=colors, edgecolor='white', 
                  linewidth=0.8, width=0.65)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sets_labels, rotation=0, ha='center', fontsize=9)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title(f'Feature Ablation Study ({model_name})', 
                fontsize=12, fontweight='bold', pad=15)
    
    y_min = max(0, min(accuracies) - 0.05)
    ax.set_ylim([y_min, 1.05])
    
    add_panel_label(ax, 'a')
    
    plt.tight_layout()
    plt.savefig(output_dir / "Fig12_Ablation_Bar.png", dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / "Fig12_Ablation_Bar.pdf", format='pdf',
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    console.print("[green]已保存：Fig12_Ablation_Bar[/green]")


def plot_grouped_ablation(
    df: pd.DataFrame, 
    output_dir: Path
):
    """多分类器 × 特征集分组柱状图。"""
    set_nature_style()
    
    console.print("[cyan]生成分组消融对比图...[/cyan]")
    
    name_map = {
        'S': 'Geometric',
        'T': 'Topological',
        'D': 'Dynamics',
        'S+T': 'Geometric\n+ Topological',
        'S+D': 'Geometric\n+ Dynamics',
        'T+D': 'Topological\n+ Dynamics',
        'ALL': 'Geometric\n+ Topological\n+ Dynamics',
        'Baseline (Raw Stat)': 'Baseline'
    }
    
    df_plot = df.copy()
    df_plot['Feature Set'] = df_plot['Feature Set'].map(lambda x: name_map.get(x, x))
    
    target_sets = ['Baseline', 'Dynamics', 'Geometric\n+ Dynamics', 'Geometric\n+ Topological\n+ Dynamics']
    
    actual_targets = [t for t in target_sets if t in df_plot['Feature Set'].values]
    df_plot = df_plot[df_plot['Feature Set'].isin(actual_targets)].copy()
    
    if df_plot.empty:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_plot['Feature Set'] = pd.Categorical(df_plot['Feature Set'], 
                                            categories=actual_targets, ordered=True)
    
    grouped_palette = {}
    if 'Baseline' in actual_targets: grouped_palette['Baseline'] = NATURE_COLORS['gray']
    if 'Dynamics' in actual_targets: grouped_palette['Dynamics'] = '#117864' 
    try:
        from .nature_style import PALETTE_MULTI
        if 'Geometric\n+ Dynamics' in actual_targets: grouped_palette['Geometric\n+ Dynamics'] = PALETTE_MULTI[1]
        if 'Geometric\n+ Topological\n+ Dynamics' in actual_targets: grouped_palette['Geometric\n+ Topological\n+ Dynamics'] = PALETTE_MULTI[0]
    except ImportError:
        pass
        
    sns.barplot(data=df_plot, x='Model', y='Accuracy', hue='Feature Set',
                palette=grouped_palette, edgecolor='white', linewidth=0.5, ax=ax)
    
    ax.set_ylim([0.96, 1.01])
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    
    ax.set_xlabel('Classifier', fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.set_title('Classifiers Robustness Ablation Analysis', 
                fontsize=11, fontweight='bold')
    
    ax.legend(title='', fontsize=8, loc='lower right', frameon=True, edgecolor='#cccccc', ncol=2)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    
    plt.tight_layout()
    plt.savefig(output_dir / "Fig16_Grouped_Ablation.png", dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / "Fig16_Grouped_Ablation.pdf", format='pdf',
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    console.print("[green]已保存：Fig16_Grouped_Ablation[/green]")


def plot_metrics_heatmap(
    df: pd.DataFrame, 
    output_dir: Path
):
    """融合特征集下各分类器的多指标热图。"""
    set_nature_style()
    
    console.print("[cyan]生成性能指标热图[/cyan]")
    
    target_sets = ['ALL', 'Fusion (3M Framework)']
    df_fusion = df[df['Feature Set'].isin(target_sets)].copy()
    
    if df_fusion.empty:
        console.print("[yellow]未找到融合特征集结果，已跳过[/yellow]")
        return
    
    metrics = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC']
    available_metrics = [m for m in metrics if m in df_fusion.columns]
    
    if not available_metrics:
        return
    
    df_pivot = df_fusion.set_index('Model')[available_metrics]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    im = ax.imshow(df_pivot.values, cmap='YlGnBu', aspect='auto', vmin=0.5, vmax=1.0)
    
    for i in range(len(df_pivot.index)):
        for j in range(len(available_metrics)):
            val = df_pivot.iloc[i, j]
            text_color = 'white' if val > 0.85 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                   color=text_color, fontsize=9, fontweight='bold')
    
    ax.set_xticks(range(len(available_metrics)))
    ax.set_xticklabels(available_metrics, fontsize=9)
    ax.set_yticks(range(len(df_pivot.index)))
    ax.set_yticklabels(df_pivot.index, fontsize=9)
    
    ax.set_xlabel('Metric', fontsize=10)
    ax.set_ylabel('Classifier', fontsize=10)
    ax.set_title('3M Framework: Performance Metrics Heatmap', 
                fontsize=11, fontweight='bold')
    
    cbar = styled_colorbar(im, ax, label='Score', shrink=0.8)
    
    add_panel_label(ax, 'c')
    
    plt.tight_layout()
    plt.savefig(output_dir / "Fig15_Metrics_Heatmap.png", dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / "Fig15_Metrics_Heatmap.pdf", format='pdf',
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    console.print("[green]已保存：Fig15_Metrics_Heatmap[/green]")


def plot_feature_radar(
    df: pd.DataFrame, 
    output_dir: Path
):
    """特征集雷达图。"""
    set_nature_style()
    
    console.print("[cyan]生成特征雷达图[/cyan]")
    
    name_map = {
        'S': 'Geometric',
        'T': 'Topological',
        'D': 'Dynamics (Pert)',
        'ALL': 'Fusion (3M)',
        'Baseline (Raw Stat)': 'Baseline'
    }
    
    df_plot = df.copy()
    df_plot['Feature Set'] = df_plot['Feature Set'].map(lambda x: name_map.get(x, x))
    
    metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC']
    available_metrics = [m for m in metrics if m in df_plot.columns]
    
    if len(available_metrics) < 3:
        console.print("[yellow]指标不足，已跳过雷达图[/yellow]")
        return
    
    df_mean = df_plot.groupby('Feature Set')[available_metrics].mean()
    
    sets_to_plot = [
        'Fusion (3M)', 
        'Dynamics (Pert)', 
        'Topological',
        'Geometric'
    ]
    sets_to_plot = [s for s in sets_to_plot if s in df_mean.index]
    
    if not sets_to_plot:
        return
    
    N = len(available_metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    
    styles = {
        'Fusion (3M)': (NATURE_COLORS['red'], '-', 2.5, 'o'),
        'Dynamics (Pert)': (NATURE_COLORS['green'], '--', 2.0, '^'),
        'Topological': (NATURE_COLORS['blue'], '-.', 1.8, 's'),
        'Geometric': ('#AEC7E8', ':', 1.5, 'd')
    }
    
    for fs in sets_to_plot:
        if fs in df_mean.index:
            values = df_mean.loc[fs].values.flatten().tolist()
            values += values[:1]
            color, ls, lw, marker = styles.get(fs, ('black', '-', 1, 'o'))
            ax.plot(angles, values, color=color, linestyle=ls, linewidth=lw,
                   marker=marker, markersize=5, label=fs)
            ax.fill(angles, values, color=color, alpha=0.08)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available_metrics, fontsize=10)
    
    min_val = df_mean[available_metrics].min().min()
    ax.set_ylim([max(0, min_val - 0.1), 1.02])
    
    ax.set_title("Classifiers Robustness Ablation Analysis", 
                fontsize=11, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.05), fontsize=9,
             frameon=True, fancybox=False, edgecolor=NATURE_COLORS['border'])
    
    plt.tight_layout()
    plt.savefig(output_dir / "Fig17_Feature_Radar.png", dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / "Fig17_Feature_Radar.pdf", format='pdf',
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    console.print("[green]已保存：Fig17_Feature_Radar[/green]")


def plot_roc_comparison(
    roc_data: Dict,
    output_dir: Path
):
    """多特征集 ROC 曲线。"""
    set_nature_style()
    
    if not roc_data:
        console.print("[yellow]无 ROC 数据，已跳过[/yellow]")
        return
    
    console.print("[cyan]生成 ROC 曲线对比图[/cyan]")
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    name_map = {
        'S': 'Geometric',
        'T': 'Topological',
        'D': 'Dynamics',
        'ALL': 'Fusion (3M)',
        'Baseline': 'Baseline'
    }
    
    colors = {
        'Fusion (3M)': NATURE_COLORS['red'],
        'Dynamics': NATURE_COLORS['green'],
        'Topological': NATURE_COLORS['blue'],
        'Geometric': '#AEC7E8',
        'Baseline': '#888888'
    }
    
    for name, data in roc_data.items():
        fpr = data.get('fpr', [])
        tpr = data.get('tpr', [])
        auc = data.get('auc', 0)
        
        if len(fpr) > 0 and len(tpr) > 0:
            color = '#333333'
            for key, c in colors.items():
                if key.lower() in name.lower():
                    color = c
                    break
            
            ax.plot(fpr, tpr, color=color, linewidth=1.8,
                   label=f'{name} (AUC={auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5, label='Random')
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title('ROC Curve Comparison', fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, frameon=True,
             fancybox=False, edgecolor=NATURE_COLORS['border'])
    
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    add_panel_label(ax, 'd')
    
    plt.tight_layout()
    plt.savefig(output_dir / "Fig18_ROC_Comparison.png", dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / "Fig18_ROC_Comparison.pdf", format='pdf',
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    console.print("[green]已保存：Fig18_ROC_Comparison[/green]")


def plot_ablation_summary(
    df: pd.DataFrame,
    output_dir: Path,
    best_model_name: Optional[str] = None
):
    """消融综合图（2×2）。"""
    set_nature_style()
    
    console.print("[cyan]生成消融实验综合图[/cyan]")
    
    name_map = {
        'S': 'Geometric',
        'T': 'Topological',
        'D': 'Dynamics',
        'S+T': 'Geometric\n+ Topological',
        'S+D': 'Geometric\n+ Dynamics',
        'T+D': 'Topological\n+ Dynamics',
        'ALL': 'Geometric\n+ Topological\n+ Dynamics',
        'Baseline (Raw Stat)': 'Baseline'
    }
    
    df_mapped = df.copy()
    df_mapped['Feature Set'] = df_mapped['Feature Set'].map(lambda x: name_map.get(x, x))
    
    fig = plt.figure(figsize=(14, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    ax_a = fig.add_subplot(gs[0, 0])
    
    if best_model_name and best_model_name in df_mapped['Model'].values:
        best_model = best_model_name
    else:
        best_model = 'RF' if 'RF' in df_mapped['Model'].values else df_mapped['Model'].iloc[0]
        
    df_model = df_mapped[df_mapped['Model'] == best_model]
    
    desired_sets = [
         'Baseline', 'Geometric', 'Topological', 'Dynamics', 
         'Geometric\n+ Topological', 'Geometric\n+ Dynamics', 
         'Topological\n+ Dynamics', 'Geometric\n+ Topological\n+ Dynamics'
    ]
    
    df_plot = df_model[df_model['Feature Set'].isin(desired_sets)].copy()
    
    if not df_plot.empty:
        df_plot = df_plot.sort_values(by='Accuracy', ascending=False)
        
        try:
            from .nature_style import PALETTE_MULTI
            n_sets = len(df_plot)
            colors = PALETTE_MULTI[:n_sets]
            if len(colors) < n_sets:
                 colors = colors * (n_sets // len(colors) + 1)
        except ImportError:
            colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(df_plot)))
        
        bars = ax_a.bar(range(len(df_plot)), df_plot['Accuracy'].values,
                       color=colors, edgecolor='white', width=0.65)
        
        for bar, acc in zip(bars, df_plot['Accuracy'].values):
            ax_a.annotate(f'{acc:.4f}', xy=(bar.get_x() + bar.get_width()/2, acc),
                         xytext=(0, 4), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8, fontweight='bold')
                         
        ax_a.set_xticks(range(len(df_plot)))
        ax_a.set_xticklabels(df_plot['Feature Set'].values, rotation=0, ha='center', fontsize=8) 
        
        ax_a.set_ylim([df_plot['Accuracy'].min()-0.05, 1.05])
        ax_a.set_title(f'Feature Contribution ({best_model})', fontsize=11, fontweight='bold')
        ax_a.set_ylabel('Accuracy', fontsize=10)
    
    add_panel_label(ax_a, 'a')
    
    ax_b = fig.add_subplot(gs[0, 1])
    
    target_sets = ['Baseline', 'Dynamics', 'Geometric\n+ Dynamics', 'Geometric\n+ Topological\n+ Dynamics']
    actual_targets = [t for t in target_sets if t in df_mapped['Feature Set'].values]
    
    df_grouped = df_mapped[df_mapped['Feature Set'].isin(actual_targets)].copy()
    
    if not df_grouped.empty:
        df_grouped['Feature Set'] = pd.Categorical(df_grouped['Feature Set'], 
                                                  categories=actual_targets, ordered=True)
        grouped_palette = {}
        if 'Baseline' in actual_targets: grouped_palette['Baseline'] = NATURE_COLORS['gray']
        if 'Dynamics' in actual_targets: grouped_palette['Dynamics'] = '#117864' 
        if 'Geometric\n+ Dynamics' in actual_targets: grouped_palette['Geometric\n+ Dynamics'] = PALETTE_MULTI[1]
        if 'Geometric\n+ Topological\n+ Dynamics' in actual_targets: grouped_palette['Geometric\n+ Topological\n+ Dynamics'] = PALETTE_MULTI[0]
        
        sns.barplot(data=df_grouped, x='Model', y='Accuracy', hue='Feature Set',
                   palette=grouped_palette, edgecolor='white', ax=ax_b, width=0.7)
        
        ax_b.legend(title='', fontsize=8, loc='lower right', ncol=2, frameon=True, edgecolor='#cccccc')
        
        ax_b.set_ylim([0.96, 1.01])
        from matplotlib.ticker import MultipleLocator
        ax_b.yaxis.set_major_locator(MultipleLocator(0.01))
    
    ax_b.set_title('Classifiers Robustness Ablation Analysis', fontsize=11, fontweight='bold')
    ax_b.set_xlabel('Classifier', fontsize=10)
    ax_b.set_ylabel('Accuracy', fontsize=10)
    add_panel_label(ax_b, 'b', x=-0.05)
    
    plt.tight_layout()
    plt.savefig(output_dir / "Fig_Ablation_Summary.png", dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / "Fig_Ablation_Summary.pdf", format='pdf',
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    console.print("[green]已保存：Fig_Ablation_Summary[/green]")


def plot_ablation_results(output_dir: Path, data_file: Optional[Path] = None):
    """生成消融实验全部图表。"""
    console.print("\n[bold cyan]── 消融实验图表 ──[/bold cyan]")
    
    if data_file and data_file.name == 'summary.csv':
        df = load_and_convert_summary(data_file)
        roc_data = {}
    elif data_file:
        df = pd.read_csv(data_file)
        roc_data = {}
    else:
        df, roc_data = load_results(output_dir)
    
    if df is None or df.empty:
        console.print("[red]无法加载消融结果，已跳过可视化[/red]")
        return
    
    best_model = 'RF'
    df_all = df[df['Feature Set'].isin(['ALL', 'Fusion (3M Framework)'])]
    if not df_all.empty:
        best_model = df_all.loc[df_all['Accuracy'].idxmax()]['Model']
        console.print(f"[bold green]消融图使用模型：{best_model}[/bold green]")

    plot_ablation_summary(df, output_dir, best_model_name=best_model)
    plot_ablation_bar_comparison(df, output_dir, model_name=best_model)
    plot_grouped_ablation(df, output_dir)
    plot_metrics_heatmap(df, output_dir)
    plot_feature_radar(df, output_dir)
    
    if roc_data:
        plot_roc_comparison(roc_data, output_dir)
    
    console.print("[bold green]消融图表已全部生成[/bold green]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='消融实验绘图')
    parser.add_argument('--summary-file', type=str, 
                        default='/mnt/3M/chbmit-allchannels/per_patient_results/summary.csv')
    parser.add_argument('--output-dir', type=str, default='./ablation_figures')
    args = parser.parse_args()
    
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    plot_ablation_results(out_path, data_file=Path(args.summary_file))
