# Author: 王梓涵 <wangzh011031@163.com>
"""
图构建模块
"""

import numpy as np
from typing import Tuple, Dict, Any
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from ..utils.console import console


def filter_negative_correlations(
    P: np.ndarray,
    copy: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    过滤负相关值，生成连接矩阵 A
    
    A_ij = max(P_ij, 0)
    
    Args:
        P: 相似度矩阵 (N, N)
        copy: 是否复制矩阵 (避免修改原矩阵)
    
    Returns:
        A: 连接矩阵 (N, N)
        stats: 统计信息字典
    """
    console.print("[cyan]过滤负相关值...[/cyan]")
    
    if copy:
        A = P.copy()
    else:
        A = P
    
    # 统计负相关数量
    negative_mask = A < 0
    n_negative = np.sum(negative_mask)
    n_total = A.size - A.shape[0]  # 排除对角线
    
    # 过滤负值
    A[negative_mask] = 0
    
    # 统计边数 (非零非对角元素)
    n_edges_after = np.sum(A > 0) - A.shape[0]  # 排除对角线
    
    stats = {
        "n_negative_filtered": int(n_negative),
        "n_total_pairs": int(n_total),
        "negative_ratio": float(n_negative / n_total) if n_total > 0 else 0.0,
        "n_edges_after": int(n_edges_after // 2),  # 无向图，除以2
    }
    
    console.print(f"  过滤了 {n_negative:,} 个负相关 ({stats['negative_ratio']*100:.2f}%)")
    console.print(f"  剩余 {stats['n_edges_after']:,} 条边")
    
    return A, stats


def apply_gaussian_kernel(
    A: np.ndarray,
    exponent: int = 1,
    sigma: str | float = "auto"
) -> np.ndarray:
    """
    应用高斯核映射，将相似度转换为连接权重
    
    公式: A'_ij = exp(-d_ij^k / sigma^k)
    其中 d_ij = 1 - A_ij (相似度转距离)

    Args:
        A: 连接矩阵 (N, N)，值域 [0, 1]
        exponent: 高斯核指数 k
        sigma: 尺度参数
    
    Returns:
        A': 变换后的连接矩阵 (N, N)
    """
    console.print(f"[cyan]应用高斯核映射 (k={exponent}, exp 模式；仅对 A>0 的边计算，0 保持 0)...[/cyan]")

    N = A.shape[0]
    non_diag_mask = ~np.eye(N, dtype=bool)
    edge_mask = non_diag_mask & (A > 0)

    # 相似度转距离（仅对边）
    D_edges = 1 - A[edge_mask]

    # 计算 sigma（仅基于边的距离分布）
    if sigma == "auto":
        # 使用距离的中位数作为 sigma；若没有边则回退到 1.0
        if D_edges.size > 0:
            sigma_value = float(np.median(D_edges))
            if sigma_value < 1e-10:
                sigma_value = 1.0
        else:
            sigma_value = 1.0
        console.print(f"  自动 sigma (边距离中位数) = {sigma_value:.4f}")
    else:
        sigma_value = float(sigma)

    sigma_k = float(np.power(sigma_value, exponent))

    # 公式: A'_ij = exp(-d_ij^k / sigma^k)，仅填充真实边
    A_prime = np.zeros_like(A, dtype=A.dtype)
    if D_edges.size > 0:
        D_k_edges = np.power(D_edges, exponent)
        A_prime[edge_mask] = np.exp(-D_k_edges / sigma_k)

    # 对角线设为 0（保持与后续图构建一致）
    np.fill_diagonal(A_prime, 0)

    # 验证方向（仅用于提示）
    console.print("  [dim]验证: 相似度 0.9 -> 权重 {:.4f}, 相似度 0.1 -> 权重 {:.4f}[/dim]".format(
        np.exp(-((1 - 0.9) ** exponent) / sigma_k),
        np.exp(-((1 - 0.1) ** exponent) / sigma_k),
    ))

    return A_prime


def partition_by_uniform(
    A: np.ndarray,
    k: int = 10,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    按均匀阈值划分子图（对高斯核映射后的矩阵操作）
    
    阈值 = (max - min) / k 的等分点
    
    Args:
        A: 连接矩阵（高斯核映射后）(N, N)
        k: 划分数量（尺度数）
    
    Returns:
        Cutoff_weight: 子图矩阵簇，保留原值 (K, N, N)
        thresholds: 阈值数组 (K,)
        stats: 每个子图的统计信息
    """
    console.print(f"[cyan]按均匀阈值划分 {k} 个子图（对高斯核映射后的矩阵操作）...[/cyan]")
    
    N = A.shape[0]
    dtype = A.dtype
    
    # 获取非对角线元素
    non_diag_mask = ~np.eye(N, dtype=bool)
    non_diag_values = A[non_diag_mask]
    
    # 获取非零值的范围
    non_zero_values = non_diag_values[non_diag_values > 0]
    
    if len(non_zero_values) == 0:
        console.print("[yellow]警告: 矩阵无非零元素，返回空子图[/yellow]")
        Cutoff_weight = np.zeros((k, N, N), dtype=dtype)
        thresholds = np.zeros(k, dtype=dtype)
        stats = {"subgraphs": [{"threshold": 0, "n_edges": 0, "n_isolated": N} for _ in range(k)]}
        return Cutoff_weight, thresholds, stats
    
    min_val = np.min(non_zero_values)
    max_val = np.max(non_zero_values)
    
    # 计算均匀分布的阈值: (max - min) / k 的等分点
    thresholds = np.linspace(min_val, max_val, k + 1)[1:]
    
    # 诊断信息：统计不同阈值区间的边数
    console.print(f"  值域: [{min_val:.6f}, {max_val:.6f}]")
    console.print(f"  总非零边数: {len(non_zero_values):,}")
    
    # 分析权重分布（用于诊断）
    percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    percentile_values = np.percentile(non_zero_values, percentiles)
    console.print("  [诊断] 权重分布百分位数:")
    for p, val in zip(percentiles, percentile_values):
        n_above = np.sum(non_zero_values >= val)
        console.print(f"    {p:3d}%: {val:.6f} ({n_above:,} 条边 >= 该值)")
    
    console.print(f"  均匀阈值 ({k} 个): {thresholds}")
    
    console.print("  [诊断] 各阈值对应的边数（非零值中）:")
    for i, thresh in enumerate(thresholds):
        n_above = np.sum(non_zero_values >= thresh)
        percentage = (n_above / len(non_zero_values) * 100) if len(non_zero_values) > 0 else 0
        console.print(f"    阈值 {i} ({thresh:.6f}): {n_above:,} 条边 >= 阈值 ({percentage:.4f}%)")
    
    # 初始化子图矩阵
    Cutoff_weight = np.zeros((k, N, N), dtype=dtype)
    
    subgraph_stats = []
    
    progress = None
    task = None
    if show_progress:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )
        progress.start()
        task = progress.add_task("[green]生成子图...", total=k)

    try:
        for i in range(k):
            threshold = thresholds[i]

            subgraph = np.where(A >= threshold, A, 0)
            np.fill_diagonal(subgraph, 0)

            Cutoff_weight[i] = subgraph

            # 统计
            n_edges = np.sum(subgraph > 0) // 2
            degrees = np.sum(subgraph > 0, axis=1)
            n_isolated = np.sum(degrees == 0)

            subgraph_stats.append({
                "index": i,
                "threshold": float(threshold),
                "n_edges": int(n_edges),
                "n_isolated_nodes": int(n_isolated),
                "n_active_nodes": int(N - n_isolated),
            })

            if progress is not None and task is not None:
                progress.update(task, advance=1)
    finally:
        if progress is not None:
            progress.stop()
    
    # 打印子图统计
    console.print("\n[bold]子图统计:[/bold]")
    console.print(f"  {'索引':<6} {'阈值':<15} {'边数':<12} {'孤立节点':<12} {'活跃节点':<12}")
    console.print("  " + "-" * 60)
    for s in subgraph_stats:
        console.print(f"  {s['index']:<6} {s['threshold']:<15.6f} {s['n_edges']:<12} {s['n_isolated_nodes']:<12} {s['n_active_nodes']:<12}")
    
    stats = {"subgraphs": subgraph_stats}
    
    return Cutoff_weight, thresholds, stats


def partition_by_quantile(
    A: np.ndarray,
    k: int = 10,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    按分位数划分子图（对高斯核映射后的矩阵操作）
    
    使用分位数划分保证每个子图的边数大致均匀。
    将非零边按权重排序，然后划分为 k 个等分，每个子图包含一个分位数区间的边。
    
    Args:
        A: 连接矩阵（高斯核映射后）(N, N)
        k: 划分数量（尺度数）
    
    Returns:
        Cutoff_weight: 子图矩阵簇，保留原值 (K, N, N)
        thresholds: 阈值数组 (K,)，每个阈值对应一个分位数
        stats: 每个子图的统计信息
    """
    console.print(f"[cyan]按分位数划分 {k} 个子图（保证边数均匀）...[/cyan]")
    
    N = A.shape[0]
    dtype = A.dtype
    
    # 获取非对角线元素
    non_diag_mask = ~np.eye(N, dtype=bool)
    non_diag_values = A[non_diag_mask]
    
    # 获取非零值的范围
    non_zero_values = non_diag_values[non_diag_values > 0]
    
    if len(non_zero_values) == 0:
        console.print("[yellow]警告: 矩阵无非零元素，返回空子图[/yellow]")
        Cutoff_weight = np.zeros((k, N, N), dtype=dtype)
        thresholds = np.zeros(k, dtype=dtype)
        stats = {"subgraphs": [{"threshold": 0, "n_edges": 0, "n_isolated": N} for _ in range(k)]}
        return Cutoff_weight, thresholds, stats
    
    min_val = np.min(non_zero_values)
    max_val = np.max(non_zero_values)
    
    # 获取非零边的索引（在原始矩阵中的位置）
    non_zero_mask = A > 0
    np.fill_diagonal(non_zero_mask, False)  # 排除对角线
    row_indices, col_indices = np.where(non_zero_mask)
    
    # 获取这些边的权重值
    edge_weights = A[row_indices, col_indices]
    
    # 按权重从大到小排序，获取排序索引
    sorted_indices = np.argsort(edge_weights)[::-1]
    n_edges_total = len(sorted_indices)
    n_edges_per_subgraph = n_edges_total // k
    
    # 计算每个子图应该包含的边数（处理余数）
    edges_per_subgraph = [n_edges_per_subgraph] * k
    remainder = n_edges_total % k
    for i in range(remainder):
        edges_per_subgraph[i] += 1
    
    # 计算每个子图的起始和结束索引
    start_indices = [0]
    for i in range(k):
        start_indices.append(start_indices[-1] + edges_per_subgraph[i])
    
    console.print(f"  值域: [{min_val:.6f}, {max_val:.6f}]")
    console.print(f"  总非零边数: {n_edges_total:,}")
    console.print(f"  每子图边数: {edges_per_subgraph}")
    
    # 初始化子图矩阵
    Cutoff_weight = np.zeros((k, N, N), dtype=dtype)
    
    subgraph_stats = []
    
    progress = None
    task = None
    if show_progress:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )
        progress.start()
        task = progress.add_task("[green]生成子图...", total=k)

    try:
        for i in range(k):
            # 获取属于当前子图的边的排序索引
            subgraph_sorted_indices = sorted_indices[start_indices[i]:start_indices[i+1]]

            # 获取这些边在原始矩阵中的位置
            subgraph_rows = row_indices[subgraph_sorted_indices]
            subgraph_cols = col_indices[subgraph_sorted_indices]

            # 创建子图矩阵
            subgraph = np.zeros_like(A)
            subgraph[subgraph_rows, subgraph_cols] = A[subgraph_rows, subgraph_cols]
            # 对称填充（无向图）
            subgraph[subgraph_cols, subgraph_rows] = A[subgraph_cols, subgraph_rows]
            np.fill_diagonal(subgraph, 0)

            Cutoff_weight[i] = subgraph

            # 计算阈值（用于统计和后续二值化）
            if len(subgraph_sorted_indices) > 0:
                subgraph_weights = edge_weights[subgraph_sorted_indices]
                threshold = float(np.min(subgraph_weights))  # 当前子图的最小权重
                max_weight = float(np.max(subgraph_weights))  # 当前子图的最大权重
            else:
                threshold = 0.0
                max_weight = 0.0

            n_edges = len(subgraph_sorted_indices)
            degrees = np.sum(subgraph > 0, axis=1)
            n_isolated = np.sum(degrees == 0)

            quantile_low = 100.0 * (1.0 - (i + 1) / k)
            quantile_high = 100.0 * (1.0 - i / k)

            subgraph_stats.append({
                "index": i,
                "threshold": threshold,
                "max_weight": max_weight,
                "quantile_low": quantile_low,
                "quantile_high": quantile_high,
                "n_edges": int(n_edges),
                "n_isolated_nodes": int(n_isolated),
                "n_active_nodes": int(N - n_isolated),
            })

            if progress is not None and task is not None:
                progress.update(task, advance=1)
    finally:
        if progress is not None:
            progress.stop()
    
    # 打印子图统计
    console.print("\n[bold]子图统计（分位数划分）:[/bold]")
    console.print(f"  {'索引':<6} {'分位数范围':<15} {'阈值':<15} {'边数':<12} {'孤立节点':<12} {'活跃节点':<12}")
    console.print("  " + "-" * 80)
    for s in subgraph_stats:
        quantile_str = f"[{s['quantile_low']:.1f}%, {s['quantile_high']:.1f}%)"
        console.print(f"  {s['index']:<6} {quantile_str:<15} {s['threshold']:<15.6f} {s['n_edges']:<12} {s['n_isolated_nodes']:<12} {s['n_active_nodes']:<12}")
    
    edge_counts = [s['n_edges'] for s in subgraph_stats]
    if len(edge_counts) > 0:
        mean_edges = np.mean(edge_counts)
        std_edges = np.std(edge_counts)
        cv = std_edges / mean_edges if mean_edges > 0 else 0
        console.print(f"\n  边数均匀性: 均值={mean_edges:.0f}, 标准差={std_edges:.0f}, 变异系数={cv:.4f}")
    
    stats = {"subgraphs": subgraph_stats, "partition_method": "quantile"}
    
    threshold_array = np.array([s['threshold'] for s in subgraph_stats], dtype=dtype)
    
    return Cutoff_weight, threshold_array, stats


def partition_by_knn(
    A: np.ndarray,
    k_list: list = [2, 4, 6, 8, 10]
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    按KNN划分子图 (Top-K per row/col, symmetrized)
    """
    console.print(f"[cyan]按 KNN 划分 {len(k_list)} 个子图...[/cyan]")
    N = A.shape[0]
    
    subgraphs = []
    thresholds = [] 
    
    for k in k_list:
        subgraph = sparsify_matrix(A, 'knn', top_k=k)
        subgraphs.append(subgraph)
        thresholds.append(0.0)
        
    Cutoff_weight = np.array(subgraphs)
    stats = {"method": "knn", "k_values": k_list}
    return Cutoff_weight, np.array(thresholds), stats


def threshold_matrix(A: np.ndarray, threshold: float) -> np.ndarray:
    """阈值二值化"""
    B = np.zeros_like(A)
    B[A > threshold] = 1
    return B

def sparsify_matrix(A: np.ndarray, method: str, threshold: float = 0, top_k: int = 5) -> np.ndarray:
    """稀疏化矩阵 (保留原值)"""
    S = np.zeros_like(A)
    if method == 'threshold':
        S[A > threshold] = A[A > threshold]
    elif method == 'knn':
        N = A.shape[0]
        eff_k = min(top_k, N)
        idx = np.argpartition(A, -eff_k, axis=1)[:, -eff_k:]
        rows = np.arange(N)[:, None]
        S[rows, idx] = A[rows, idx]
        S = np.maximum(S, S.T)
        
    return S

def binarize_cutoff(
    Cutoff_weight: np.ndarray,
    thresholds: np.ndarray
) -> np.ndarray:
    """
    将 Cutoff_weight 二值化为 Cutoff_0_1
    
    对于每个子图，大于阈值的为1，小于等于阈值的为0
    
    Args:
        Cutoff_weight: 子图矩阵簇，保留原值 (K, N, N)
        thresholds: 阈值数组 (K,)
    
    Returns:
        Cutoff_0_1: 二值化子图矩阵簇 (K, N, N)
    """
    console.print("[cyan]生成二值化子图矩阵 Cutoff_0_1...[/cyan]")
    
    K, N, _ = Cutoff_weight.shape
    Cutoff_0_1 = np.zeros_like(Cutoff_weight, dtype=np.float32)
    
    for k in range(K):
        threshold = thresholds[k]
        # 大于阈值的为1，否则为0
        Cutoff_0_1[k] = (Cutoff_weight[k] > threshold).astype(np.float32)
        # 对角线设为0
        np.fill_diagonal(Cutoff_0_1[k], 0)
    
    console.print(f"  生成 {K} 个二值化子图矩阵")
    
    return Cutoff_0_1


def compute_laplacian(
    A: np.ndarray,
    take_abs: bool = True,
    show_progress: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    计算拉普拉斯矩阵
    
    L = D - A
    
    其中 D_ii = Σ_j A_ij 是度矩阵
    
    Args:
        A: 连接矩阵 (N, N) 或子图矩阵簇 (K, N, N)
        take_abs: 是否对元素取绝对值
    
    Returns:
        L: 拉普拉斯矩阵 (N, N) 或拉普拉斯矩阵簇 (K, N, N)
        stats: 统计信息字典
    """
    if A.ndim == 2:
        # 单个矩阵
        console.print("[cyan]计算拉普拉斯矩阵...[/cyan]")
        N = A.shape[0]
        
        # 计算度矩阵 (对角线)
        degrees = np.sum(A, axis=1)
        
        # 拉普拉斯矩阵: L = D - A
        L = np.diag(degrees) - A
        
        # 取绝对值
        if take_abs:
            L = np.abs(L)
        
        # 统计信息
        n_edges = np.sum(A > 0) // 2  # 无向图
        isolated_nodes = np.sum(degrees < 1e-10)
        
        stats = {
            "n_nodes": int(N),
            "n_edges": int(n_edges),
            "n_isolated_nodes": int(isolated_nodes),
            "avg_degree": float(np.mean(degrees)),
            "max_degree": float(np.max(degrees)),
            "min_degree": float(np.min(degrees)),
        }
        
        console.print(f"  节点数: {N}, 边数: {n_edges}, 孤立节点: {isolated_nodes}")
        console.print(f"  平均度: {stats['avg_degree']:.2f}, 最大度: {stats['max_degree']:.2f}")
        
        return L, stats
    elif A.ndim == 3:
        # 多个子图矩阵 (K, N, N)
        K, N, _ = A.shape
        console.print(f"[cyan]计算 {K} 个子图的拉普拉斯矩阵...[/cyan]")
        
        L_matrices = np.zeros_like(A)
        all_stats = []
        
        progress = None
        task = None
        if show_progress:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            )
            progress.start()
            task = progress.add_task("[green]计算拉普拉斯矩阵...", total=K)

        try:
            for k in range(K):
                A_k = A[k]

                # 计算度矩阵
                degrees = np.sum(A_k, axis=1)

                # 拉普拉斯矩阵: L = D - A
                L_k = np.diag(degrees) - A_k

                # 取绝对值
                if take_abs:
                    L_k = np.abs(L_k)

                L_matrices[k] = L_k

                # 统计信息
                n_edges = np.sum(A_k > 0) // 2
                isolated_nodes = np.sum(degrees < 1e-10)

                all_stats.append({
                    "index": k,
                    "n_edges": int(n_edges),
                    "n_isolated_nodes": int(isolated_nodes),
                    "avg_degree": float(np.mean(degrees)),
                    "max_degree": float(np.max(degrees)),
                    "min_degree": float(np.min(degrees)),
                })

                if progress is not None and task is not None:
                    progress.update(task, advance=1)
        finally:
            if progress is not None:
                progress.stop()
        
        # 汇总统计
        stats = {
            "n_subgraphs": K,
            "n_nodes": int(N),
            "subgraphs": all_stats
        }
        
        return L_matrices, stats
    else:
        raise ValueError(f"不支持的矩阵维度: {A.ndim}，期望 2 或 3")


if __name__ == "__main__":
    # 简单测试
    console.print("[bold green]测试图构建模块[/bold green]")
    
    # 生成测试相似度矩阵
    np.random.seed(42)
    N = 100
    P = np.random.randn(N, N)
    P = (P + P.T) / 2  # 对称化
    np.fill_diagonal(P, 1)
    
    # 测试各函数
    A, filter_stats = filter_negative_correlations(P)
    console.print(f"过滤统计: {filter_stats}")
    
    A_prime = apply_gaussian_kernel(A)
    console.print(f"高斯核映射后形状: {A_prime.shape}")
    
    Cutoff_weight, thresholds, subgraph_stats = partition_by_uniform(A_prime, k=5)
    console.print(f"子图矩阵形状: {Cutoff_weight.shape}")
    
    Cutoff_0_1 = binarize_cutoff(Cutoff_weight, thresholds)
    console.print(f"二值化子图矩阵形状: {Cutoff_0_1.shape}")
    
    L_matrices, lap_stats = compute_laplacian(Cutoff_weight)
    console.print(f"拉普拉斯矩阵簇形状: {L_matrices.shape}")
    
    console.print("[bold green]测试完成![/bold green]")
