# Author: 王梓涵 <wangzh011031@163.com>
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 chb01 患者生成多尺度 Persistence Barcodes 
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# 添加 legacy 目录到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_DIR = PROJECT_ROOT / "legacy"
sys.path.insert(0, str(LEGACY_DIR))
sys.path.insert(0, str(LEGACY_DIR / "step_3_feature"))

from step_3_feature.tda_features import extract_topological_features

# SCI 配色
SCI_COLORS = {
    'primary': '#9BDCFC',    # Light Blue (H1)
    'secondary': '#F0CFEA',  # Light Pink (H0)
    'text': '#000000',
    'border': '#333333'
}

def set_sci_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.linewidth': 0.8,
        'figure.dpi': 300
    })

def main():
    # 数据路径
    patient_id = "chb01"
    matrix_dir = Path("/mnt/3M/chbmit-allchannels/per_patient_results") / patient_id
    cutoff_weight_path = matrix_dir / "cutoff_weight.npy"
    summary_dir = PROJECT_ROOT / "summary_figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading matrices from {cutoff_weight_path}...")
    
    try:
        # 1. 计算/获取条形码 (完全沿用 legacy 逻辑)
        cutoff_matrices = np.load(cutoff_weight_path)
        config = {
            'tda': {
                'topological': {
                    'max_dimension': 1,
                    'max_edge_length': 0.9 
                }
            }
        }
        
        print("Extracting barcodes using legacy TDA logic...")
        _, bars_list = extract_topological_features(
            cutoff_matrices, 
            config=config, 
            num_workers=4,
            verbose=True
        )
        
        # 2. 绘制 5x2 SCI 风格图表
        set_sci_style()
        fig = plt.figure(figsize=(12, 12))
        gs = gridspec.GridSpec(5, 2, hspace=0.5, wspace=0.25)
        
        for idx, bars in enumerate(bars_list):
            row = idx % 5
            col = idx // 5
            ax = fig.add_subplot(gs[row, col])
            
            if len(bars) > 0:
                y_offset = 0

                
                for dim, color in [(1, SCI_COLORS['primary']), (0, SCI_COLORS['secondary'])]:
                    # 筛选并按长度 (death - birth) 排序
                    dim_bars = [b for b in bars if b['dim'] == dim]

                    dim_bars_sorted = sorted(dim_bars, key=lambda x: x['death'] - x['birth'], reverse=True)
                    dim_bars_limited = dim_bars_sorted[:50] # 每个维度取前50条
                    
                    # 绘制
                    for bar in reversed(dim_bars_limited): # 反向遍历，即长条最后画（在 y 较小处）

                        pass
                
                # 重新实现排序绘图逻辑：长条在下
                y_curr = 0
                all_bars_to_plot = []
                for dim, color in [(1, SCI_COLORS['primary']), (0, SCI_COLORS['secondary'])]:
                    dim_bars = [b for b in bars if b['dim'] == dim]
                    sorted_bars = sorted(dim_bars, key=lambda x: x['death'] - x['birth'], reverse=True)
                    for b in sorted_bars[:50]:
                        all_bars_to_plot.append((b, color))
                
                for b, col in all_bars_to_plot:
                    ax.hlines(y_curr, b['birth'], b['death'], colors=col, lw=0.8, alpha=0.8)
                    y_curr += 1
            
            # 美化小图
            ax.set_title(f'Scale {idx+1}', fontsize=10, fontweight='bold', loc='left')
            ax.set_xlim(-0.02, 0.92)
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            if row == 4:
                ax.set_xlabel("Filtration Value", fontsize=9)
            
        # 全局标题和图例
        fig.suptitle(f"Persistent Barcodes (SCI Style: Long Bars Bottom) - {patient_id}", 
                     fontsize=14, fontweight='bold', y=0.98)
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=SCI_COLORS['secondary'], lw=2, label='$H_0$ (Components)'),
            Line2D([0], [0], color=SCI_COLORS['primary'], lw=2, label='$H_1$ (Cycles)')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.94), 
                   ncol=2, frameon=False, fontsize=10)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        save_path = summary_dir / f"{patient_id}_persistence_barcodes_sci_5x2.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to: {save_path}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
