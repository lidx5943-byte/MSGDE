# Author: 王梓涵 <wangzh011031@163.com>
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算并绘制 chb01 患者 10 个尺度子图的贝蒂数 (Beta0, Beta1)
参考 Visual/fig1_graph_construction/figure1_complete.py 的计算方式
"""

import numpy as np
import matplotlib.pyplot as plt
import gudhi
from scipy import sparse
from scipy.linalg import eigh
from pathlib import Path
import os

# =============================================================================
# Helper Functions from Figure 1
# =============================================================================

def compute_pthl_spectrum(simplex_tree):
    """
    计算持久性拓扑拉普拉斯矩阵的谱分析
    参考 figure1_complete.py
    """
    simplices = [s for s, _ in simplex_tree.get_simplices()]
    s0 = sorted([s for s in simplices if len(s) == 1])  # 0-simplex (nodes)
    s1 = sorted([s for s in simplices if len(s) == 2])  # 1-simplex (edges)
    s2 = sorted([s for s in simplices if len(s) == 3])  # 2-simplex (triangles)
    
    n0, n1, n2 = len(s0), len(s1), len(s2)
    id0 = {tuple(s): i for i, s in enumerate(s0)}
    id1 = {tuple(s): i for i, s in enumerate(s1)}
    
    # Boundary operator B1: C1 -> C0
    if n1 > 0:
        rows, cols, data = [], [], []
        for j, edge in enumerate(s1):
            u, v = edge
            rows.extend([id0[(u,)], id0[(v,)]])
            cols.extend([j, j])
            data.extend([-1, 1])
        B1 = sparse.csr_matrix((data, (rows, cols)), shape=(n0, n1))
    else:
        B1 = sparse.csr_matrix((n0, 0))

    # Boundary operator B2: C2 -> C1
    if n2 > 0:
        rows, cols, data = [], [], []
        for k, tri in enumerate(s2):
            u, v, w = tri
            e_signs = [
                (tuple(sorted((v, w))), 1),   # (v,w)
                (tuple(sorted((u, w))), -1),  # (u,w)
                (tuple(sorted((u, v))), 1)    # (u,v)
            ]
            for e, sign in e_signs:
                if e in id1:
                    rows.append(id1[e])
                    cols.append(k)
                    data.append(sign)
        B2 = sparse.csr_matrix((data, (rows, cols)), shape=(n1, n2))
    else:
        B2 = sparse.csr_matrix((n1, 0))

    # L0 = B1 @ B1.T
    L0 = B1 @ B1.T
    
    # L1 = B2 @ B2.T + B1.T @ B1
    L1 = (B2 @ B2.T) + (B1.T @ B1) if n1 > 0 else sparse.csr_matrix((0, 0))
    
    def analyze_spectrum(mat):
        if mat.shape[0] == 0:
            return 0, 0.0
        
        # Dense eigenvalue calculation for small to medium matrices
        vals = eigh(mat.toarray(), eigvals_only=True)
        tol = 1e-5
        zeros = vals[np.abs(vals) < tol]
        betti = len(zeros)
        
        non_zeros = vals[np.abs(vals) >= tol]
        min_lambda = non_zeros.min() if len(non_zeros) > 0 else np.nan
        
        return betti, min_lambda

    b0, l0 = analyze_spectrum(L0)
    b1, l1 = analyze_spectrum(L1)
    
    return b0, l0, b1, l1

# =============================================================================
# Main Execution
# =============================================================================

def main():
    # 路径设置
    patient_id = "chb01"
    data_dir = Path("/mnt/3M/chbmit-allchannels/per_patient_results") / patient_id
    output_dir = Path("summary_figures")
    output_dir.mkdir(exist_ok=True)
    
    matrix_path = data_dir / "cutoff_binary.npy"
    
    if not matrix_path.exists():
        print(f"错误: 找不到数据文件 {matrix_path}")
        return
    
    print(f"正在加载 {patient_id} 的子图数据...")
    subgraphs = np.load(matrix_path)
    num_scales = subgraphs.shape[0]
    num_nodes = subgraphs.shape[1]
    print(f"数据形状: {subgraphs.shape} (Scales={num_scales}, Nodes={num_nodes})")
    
    b0_list = []
    b1_list = []
    l0_list = []
    l1_list = []
    
    for k in range(num_scales):
        print(f"正在处理尺度 {k+1}/{num_scales}...")
        adj = subgraphs[k]
        
        # 构建 Gudhi 单纯复形树
        st = gudhi.SimplexTree()
        # 添加顶点
        for i in range(num_nodes):
            st.insert([i])
        # 添加边
        rows, cols = np.where(adj > 0)
        edge_count = 0
        for u, v in zip(rows, cols):
            if u < v:
                st.insert([u, v])
                edge_count += 1
        
        # 扩展到 2 维 (寻找三角形)，以便计算 B2
        st.expansion(2)
        
        # 使用 PTHL 方法计算 Betti 数和谱间隙
        b0, l0, b1, l1 = compute_pthl_spectrum(st)
        
        b0_list.append(b0)
        b1_list.append(b1)
        l0_list.append(l0)
        l1_list.append(l1)
        
        print(f"  尺度 {k+1}: Edges={edge_count}, Beta0={b0}, Beta1={b1}, L0_min={l0:.4f}, L1_min={l1:.4f}")

    # =============================================================================
    # 可视化 (SCI 风格)
    # =============================================================================
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 300
    })
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    scales = np.arange(1, num_scales + 1)
    
    # Plot Beta0
    ax1.plot(scales, b0_list, marker='o', color='#9BDCFC', linewidth=2, label=r'$\beta_0$ (Connected Components)')
    ax1.set_ylabel(r'Betti Number $\beta_0$', fontsize=12)
    ax1.set_title(f'Topological Analysis of Multi-scale Subgraphs ({patient_id})', fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend()
    
    # Plot Beta1
    ax2.plot(scales, b1_list, marker='s', color='#F0CFEA', linewidth=2, label=r'$\beta_1$ (Fundamental Cycles)')
    ax2.set_ylabel(r'Betti Number $\beta_1$', fontsize=12)
    ax2.set_xlabel('Scale Index (Weak to Strong Connection Quantile)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_xticks(scales)
    ax2.legend()
    
    plt.tight_layout()
    
    save_path = output_dir / f"{patient_id}_betti_analysis.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    
    print(f"\n分析完成!")
    print(f"结果已保存至: {save_path}")
    print(f"和: {save_path.with_suffix('.pdf')}")

if __name__ == "__main__":
    main()
