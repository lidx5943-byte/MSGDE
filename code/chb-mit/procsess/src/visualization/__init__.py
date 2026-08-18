# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""
可视化模块入口 (Nature Journal Style)
======================================

提供统一的 Nature 子刊级别可视化功能。

Usage:
    from src.visualization import set_nature_style, generate_patient_visualizations
    
    # 设置全局风格
    set_nature_style()
    
    # 使用具体的可视化函数
    from src.visualization import plot_ablation_results
"""

# 核心风格模块
from .nature_style import (
    # 风格设置
    set_nature_style,
    create_figure,
    save_figure,
    add_panel_label,
    
    # 颜色常量
    NATURE_COLORS,
    PALETTE_2CLASS,
    PALETTE_3CLASS,
    PALETTE_MULTI,
    CMAPS,
    FIGURE_SIZES,
    
    # 辅助函数
    add_significance_annotation,
    add_errorbar,
    styled_colorbar,
    
    # 专用绑图函数
    nature_heatmap,
    nature_barplot,
    nature_boxplot,
    nature_lineplot,
    
    # 上下文管理器
    NatureFigure,
)

# 患者分析可视化
from .patient_viz import (
    generate_patient_visualizations,
    plot_patient_accuracy_bar,
    plot_classifier_comparison_heatmap,
    plot_metrics_radar,
    plot_sample_distribution,
    plot_boxplot_metrics,
    plot_performance_summary,
)

# 消融实验可视化
from .ablation_plotter import (
    plot_ablation_results,
    plot_ablation_bar_comparison,
    plot_grouped_ablation,
    plot_metrics_heatmap,
    plot_feature_radar,
    plot_roc_comparison,
    plot_ablation_summary,
)

# 耦合强度×尺度可视化
from .coupling_scale_plotter import (
    plot_coupling_scale_heatmap,
)

# 尺度消融可视化
from .scale_ablation_plotter import (
    plot_scale_ablation_results,
    plot_scale_summary,
)

# 矩阵可视化
from .matrix_viz import (
    plot_adjacency_matrix,
    plot_connectivity_graph,
)

# 特征可视化
from .feature_viz import (
    plot_feature_distribution,
    plot_tsne,
    plot_pca,
    plot_feature_importance,
    plot_feature_comparison,
)

# 微扰可视化
from .perturbation_viz import (
    plot_perturbation_heatmap,
    plot_top_critical_nodes,
    plot_perturbation_summary,
)


__all__ = [
    # 风格
    "set_nature_style",
    "create_figure",
    "save_figure",
    "add_panel_label",
    "NATURE_COLORS",
    "PALETTE_2CLASS",
    "PALETTE_3CLASS",
    "PALETTE_MULTI",
    "CMAPS",
    "FIGURE_SIZES",
    "add_significance_annotation",
    "add_errorbar",
    "styled_colorbar",
    "nature_heatmap",
    "nature_barplot",
    "nature_boxplot",
    "nature_lineplot",
    "NatureFigure",
    
    # 患者分析
    "generate_patient_visualizations",
    "plot_patient_accuracy_bar",
    "plot_classifier_comparison_heatmap",
    "plot_metrics_radar",
    "plot_sample_distribution",
    "plot_boxplot_metrics",
    "plot_performance_summary",
    
    # 消融实验
    "plot_ablation_results",
    "plot_ablation_bar_comparison",
    "plot_grouped_ablation",
    "plot_metrics_heatmap",
    "plot_feature_radar",
    "plot_roc_comparison",
    "plot_ablation_summary",
    
    # 耦合×尺度
    "plot_coupling_scale_heatmap",
    
    # 尺度消融
    "plot_scale_ablation_results",
    "plot_scale_summary",
    
    # 矩阵
    "plot_adjacency_matrix",
    "plot_connectivity_graph",
    
    # 特征
    "plot_feature_distribution",
    "plot_tsne",
    "plot_pca",
    "plot_feature_importance",
    "plot_feature_comparison",
    
    # 微扰
    "plot_perturbation_heatmap",
    "plot_top_critical_nodes",
    "plot_perturbation_summary",
]
