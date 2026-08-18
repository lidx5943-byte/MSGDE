# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""
Nature Journal Visualization Style Module
==========================================

统一的 Nature 子刊级别可视化风格配置。
所有可视化模块应导入此模块以确保风格一致性。

Usage:
    from .nature_style import set_nature_style, NATURE_COLORS, NatureFigure
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from typing import Tuple, Optional, List, Union
from pathlib import Path

# ============================================================================
# Nature Color Palettes (来自 Nature 系列期刊官方配色)
# ============================================================================

NATURE_COLORS = {
    # 主要分类色 (高对比度，适合区分类别)
    # Adjusted user preference: darker/deeper tones
    'red': '#C0392B',        # 深红 (从 #E64B35 调整)
    'blue': '#2E86C1',       # 深蓝 (从 #4DBBD5 调整)
    'green': '#00A087',      # 翠绿 - 第三类别
    'purple': '#3C5488',     # 深蓝紫 - 沉稳色
    'orange': '#E67E22',     # 深橙 (匹配整体色调)
    'gray': '#7F8C8D',       # 深灰
    
    # 功能色
    'highlight': '#C0392B',  # 高亮/关键
    'primary': '#2E86C1',    # 主色
    'secondary': '#00A087',  # 次要
    'accent': '#E67E22',     # 强调
    'muted': '#95A5A6',      # 弱化
    
    # 深浅变体
    'dark_blue': '#1B4F72',
    'light_blue': '#5DADE2',
    'dark_red': '#922B21',
    'light_red': '#E6B0AA',
    
    # 文本和边框
    'text': '#333333',
    'border': '#666666',
    'grid': '#E0E0E0',
}

# 预定义调色板
PALETTE_2CLASS = [NATURE_COLORS['blue'], NATURE_COLORS['red']]
PALETTE_3CLASS = [NATURE_COLORS['green'], NATURE_COLORS['blue'], NATURE_COLORS['red']]
PALETTE_MULTI = [NATURE_COLORS['red'], NATURE_COLORS['blue'], NATURE_COLORS['green'],
                 NATURE_COLORS['purple'], NATURE_COLORS['orange'], NATURE_COLORS['gray']]

# Colormap 推荐
CMAPS = {
    'diverging': 'RdBu_r',      # 分散型 (差异/变化)
    'sequential': 'YlOrRd',     # 顺序型 (强度)
    'sequential_blue': 'Blues', # 顺序蓝
    'accuracy': 'RdYlGn',       # 准确率 (红差绿好)
    'heatmap': 'viridis',       # 通用热图
}

# ============================================================================
# Nature Figure Dimensions (英寸)
# ============================================================================

FIGURE_SIZES = {
    'single_column': (3.5, 2.8),       # 单栏 89mm
    'one_half_column': (5.5, 4.0),     # 1.5栏 140mm  
    'double_column': (7.2, 5.5),       # 双栏 183mm (最常用)
    'full_page': (7.2, 9.5),           # 整页
    
    # 特殊用途
    'square': (4.0, 4.0),
    'wide': (7.2, 3.5),
    'tall': (4.0, 6.0),
}

# ============================================================================
# Core Style Setup Function
# ============================================================================

def set_nature_style(context: str = 'paper'):
    """
    设置 Nature 期刊级别的全局绘图风格。
    
    Parameters
    ----------
    context : str
        绘图上下文，'paper' 或 'poster'
    
    Examples
    --------
    >>> from src.visualization.nature_style import set_nature_style
    >>> set_nature_style()
    >>> plt.plot([1,2,3], [1,4,9])
    """
    
    # 字体大小配置 (Nature 标准: 5-8pt)
    font_scale = 1.0 if context == 'paper' else 1.5
    
    base_params = {
        # === 字体设置 ===
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
        'font.size': 7 * font_scale,
        
        # === 标题和标签 ===
        'axes.titlesize': 9 * font_scale,
        'axes.titleweight': 'bold',
        'axes.titlepad': 8,
        'axes.labelsize': 8 * font_scale,
        'axes.labelweight': 'normal',
        'axes.labelpad': 4,
        
        # === 刻度 ===
        'xtick.labelsize': 7 * font_scale,
        'ytick.labelsize': 7 * font_scale,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        
        # === 图例 ===
        'legend.fontsize': 7 * font_scale,
        'legend.frameon': False,
        'legend.borderaxespad': 0.5,
        'legend.handlelength': 1.5,
        'legend.handletextpad': 0.5,
        
        # === 线条 ===
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.4,
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
        'patch.linewidth': 0.5,
        
        # === 颜色 ===
        'axes.edgecolor': NATURE_COLORS['border'],
        'axes.labelcolor': NATURE_COLORS['text'],
        'xtick.color': NATURE_COLORS['text'],
        'ytick.color': NATURE_COLORS['text'],
        'text.color': NATURE_COLORS['text'],
        
        # === 背景 ===
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        
        # === 边框 (spines) ===
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # === 网格 ===
        'axes.grid': False,
        'grid.color': NATURE_COLORS['grid'],
        'grid.alpha': 0.5,
        
        # === 保存设置 ===
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'savefig.transparent': False,
        
        # === 其他 ===
        'figure.dpi': 150,
        'figure.autolayout': False,
    }
    
    plt.rcParams.update(base_params)
    
    # Seaborn 风格叠加
    sns.set_style("ticks", {
        'axes.edgecolor': NATURE_COLORS['border'],
        'axes.linewidth': 0.8,
    })
    sns.set_context("paper", font_scale=font_scale)
    sns.set_palette(PALETTE_MULTI)


# ============================================================================
# Helper Functions
# ============================================================================

def create_figure(
    nrows: int = 1, 
    ncols: int = 1, 
    figsize: Optional[Tuple[float, float]] = None,
    size_key: str = 'double_column',
    **kwargs
) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
    """
    创建 Nature 风格的 Figure。
    
    Parameters
    ----------
    nrows, ncols : int
        子图行列数
    figsize : tuple, optional
        自定义尺寸 (width, height) in inches
    size_key : str
        预设尺寸: 'single_column', 'double_column', 'full_page', etc.
    **kwargs
        传递给 plt.subplots 的其他参数
        
    Returns
    -------
    fig, axes : Figure and Axes
    """
    set_nature_style()
    
    if figsize is None:
        figsize = FIGURE_SIZES.get(size_key, FIGURE_SIZES['double_column'])
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    fig.set_facecolor('white')
    
    return fig, axes


def add_panel_label(
    ax: plt.Axes, 
    label: str, 
    x: float = -0.12, 
    y: float = 1.08,
    fontsize: int = 12,
    fontweight: str = 'bold'
):
    """
    添加子图标签 (a), (b), (c) 等 - Nature 标准格式。
    
    Parameters
    ----------
    ax : Axes
        目标子图
    label : str
        标签文本，如 'a', 'b', 'c' (不需要括号，函数自动处理)
    x, y : float
        标签位置 (相对于子图，transform=ax.transAxes)
    """
    # 确保小写
    label_text = label.lower()
    
    ax.text(x, y, label_text, 
            transform=ax.transAxes,
            fontsize=fontsize, 
            fontweight=fontweight,
            va='top', 
            ha='left',
            color=NATURE_COLORS['text'])


def add_significance_annotation(
    ax: plt.Axes,
    x1: float, x2: float,
    y: float,
    p_value: float,
    height: float = 0.02
):
    """
    添加显著性标注 (bracket + stars)。
    
    Parameters
    ----------
    ax : Axes
        目标子图
    x1, x2 : float
        横坐标起止位置
    y : float
        标注的 y 位置
    p_value : float
        p 值
    height : float
        bracket 高度
    """
    # 确定星号
    if p_value < 0.001:
        stars = '***'
    elif p_value < 0.01:
        stars = '**'
    elif p_value < 0.05:
        stars = '*'
    else:
        stars = 'n.s.'
    
    # 画 bracket
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], 
            color='black', linewidth=0.8)
    
    # 添加星号
    ax.text((x1 + x2) / 2, y + height, stars,
            ha='center', va='bottom', fontsize=8)


def add_errorbar(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    color: str = None,
    capsize: float = 3,
    capthick: float = 1,
    **kwargs
):
    """
    绘制带误差棒的数据点 (Nature 标准样式)。
    """
    if color is None:
        color = NATURE_COLORS['primary']
    
    ax.errorbar(x, y, yerr=yerr, 
                fmt='o', 
                color=color,
                ecolor=color,
                elinewidth=0.8,
                capsize=capsize,
                capthick=capthick,
                markersize=4,
                **kwargs)


def styled_colorbar(
    mappable,
    ax: plt.Axes,
    label: str = '',
    orientation: str = 'vertical',
    shrink: float = 0.8,
    pad: float = 0.02
):
    """
    添加 Nature 风格的颜色条。
    """
    cbar = plt.colorbar(mappable, ax=ax, orientation=orientation,
                        shrink=shrink, pad=pad)
    cbar.set_label(label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor(NATURE_COLORS['border'])
    
    return cbar


def save_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: Union[str, Path] = '.',
    formats: List[str] = ['png', 'pdf'],
    dpi: int = 300,
    close: bool = True
):
    """
    保存图表为多种格式 (Nature 投稿需要 PDF)。
    
    Parameters
    ----------
    fig : Figure
        要保存的图表
    filename : str
        文件名 (不含扩展名)
    output_dir : str or Path
        输出目录
    formats : list
        输出格式列表，如 ['png', 'pdf']
    dpi : int
        分辨率
    close : bool
        保存后是否关闭图表
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        filepath = output_dir / f"{filename}.{fmt}"
        fig.savefig(filepath, format=fmt, dpi=dpi, 
                    bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"  ✓ Saved: {filepath}")
    
    if close:
        plt.close(fig)


# ============================================================================
# Specialized Plot Functions
# ============================================================================

def nature_heatmap(
    data: np.ndarray,
    ax: plt.Axes = None,
    cmap: str = 'RdBu_r',
    center: float = None,
    annot: bool = True,
    fmt: str = '.2f',
    cbar: bool = True,
    cbar_label: str = '',
    xticklabels: list = None,
    yticklabels: list = None,
    **kwargs
):
    """
    Nature 风格热图。
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # 使用 seaborn heatmap
    hm = sns.heatmap(
        data, 
        ax=ax,
        cmap=cmap,
        center=center,
        annot=annot,
        fmt=fmt,
        annot_kws={'size': 7},
        linewidths=0.5,
        linecolor='white',
        cbar=cbar,
        cbar_kws={'label': cbar_label, 'shrink': 0.8} if cbar else {},
        xticklabels=xticklabels if xticklabels is not None else True,
        yticklabels=yticklabels if yticklabels is not None else True,
        **kwargs
    )
    
    # 美化 colorbar
    if cbar and hm.collections:
        cbar_obj = hm.collections[0].colorbar
        if cbar_obj:
            cbar_obj.ax.tick_params(labelsize=7)
            cbar_obj.outline.set_linewidth(0.5)
    
    return ax


def nature_barplot(
    x: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes = None,
    yerr: np.ndarray = None,
    color: Union[str, List[str]] = None,
    edgecolor: str = 'white',
    width: float = 0.7,
    show_values: bool = True,
    value_fmt: str = '.2f',
    **kwargs
):
    """
    Nature 风格柱状图 (带误差棒和数值标注)。
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    if color is None:
        color = NATURE_COLORS['primary']
    
    bars = ax.bar(x, y, width=width, color=color, edgecolor=edgecolor, 
                  linewidth=0.5, **kwargs)
    
    # 添加误差棒
    if yerr is not None:
        ax.errorbar(x, y, yerr=yerr, fmt='none', 
                    ecolor='black', elinewidth=0.8, capsize=3, capthick=0.8)
    
    # 添加数值标注
    if show_values:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:{value_fmt}}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)
    
    return ax, bars


def nature_boxplot(
    data: list,
    ax: plt.Axes = None,
    labels: list = None,
    colors: list = None,
    show_points: bool = True,
    **kwargs
):
    """
    Nature 风格箱线图 (带散点)。
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    if colors is None:
        colors = PALETTE_MULTI[:len(data)]
    
    bp = ax.boxplot(data, patch_artist=True, labels=labels, **kwargs)
    
    # 美化箱体
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor(NATURE_COLORS['border'])
        patch.set_linewidth(0.8)
    
    # 美化其他元素
    for element in ['whiskers', 'caps']:
        for line in bp[element]:
            line.set_color(NATURE_COLORS['border'])
            line.set_linewidth(0.8)
    
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1.2)
    
    # 添加散点
    if show_points:
        for i, d in enumerate(data):
            jitter = np.random.normal(0, 0.04, size=len(d))
            ax.scatter(np.repeat(i + 1, len(d)) + jitter, d,
                      alpha=0.4, s=15, color=colors[i], zorder=3)
    
    return ax, bp


def nature_lineplot(
    x: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes = None,
    yerr: np.ndarray = None,
    color: str = None,
    marker: str = 'o',
    label: str = None,
    fill_alpha: float = 0.15,
    **kwargs
):
    """
    Nature 风格折线图 (带置信区间填充)。
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    if color is None:
        color = NATURE_COLORS['primary']
    
    line, = ax.plot(x, y, color=color, marker=marker, markersize=4, 
                    linewidth=1.5, label=label, **kwargs)
    
    # 添加置信区间填充
    if yerr is not None:
        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=fill_alpha)
    
    return ax, line


# ============================================================================
# Convenience Class
# ============================================================================

class NatureFigure:
    """
    Nature 风格 Figure 上下文管理器。
    
    Examples
    --------
    >>> with NatureFigure('my_figure', output_dir='./output') as (fig, ax):
    ...     ax.plot([1,2,3], [1,4,9])
    ...     ax.set_xlabel('X')
    """
    
    def __init__(
        self, 
        filename: str,
        output_dir: Union[str, Path] = '.',
        nrows: int = 1,
        ncols: int = 1,
        figsize: Tuple[float, float] = None,
        size_key: str = 'double_column',
        formats: List[str] = ['png', 'pdf']
    ):
        self.filename = filename
        self.output_dir = Path(output_dir)
        self.nrows = nrows
        self.ncols = ncols
        self.figsize = figsize
        self.size_key = size_key
        self.formats = formats
        self.fig = None
        self.axes = None
    
    def __enter__(self):
        self.fig, self.axes = create_figure(
            self.nrows, self.ncols, 
            figsize=self.figsize, 
            size_key=self.size_key
        )
        return self.fig, self.axes
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            save_figure(self.fig, self.filename, self.output_dir, 
                       self.formats, close=True)
        else:
            plt.close(self.fig)
        return False


# ============================================================================
# Module Initialization
# ============================================================================

# 模块导入时自动设置风格
set_nature_style()

if __name__ == "__main__":
    # 测试风格
    print("Testing Nature Style...")
    
    with NatureFigure('nature_style_test', output_dir='/tmp') as (fig, ax):
        x = np.linspace(0, 10, 50)
        y = np.sin(x) + np.random.randn(50) * 0.1
        yerr = np.ones(50) * 0.2
        
        nature_lineplot(x, y, ax=ax, yerr=yerr, label='Test Data')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_title('Nature Style Test')
        ax.legend()
        add_panel_label(ax, 'a')
    
    print("✓ Test completed!")
