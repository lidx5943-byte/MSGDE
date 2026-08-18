# Author: 王梓涵 <wangzh011031@163.com>
"""
相似度计算模块
支持分块计算以优化大规模矩阵的内存使用。
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cdist
from typing import Literal, Optional
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from ..utils.console import console


def compute_pearson_channel(
    x_i: np.ndarray, 
    x_j: np.ndarray
) -> float:
    """
    计算两个样本在单个通道上的 Pearson 相关系数
    
    Args:
        x_i: 样本 i 的时间序列 (T,)
        x_j: 样本 j 的时间序列 (T,)
    
    Returns:
        Pearson 相关系数 (如果无法计算则返回 0)
    """
    # 检查标准差是否为零
    if np.std(x_i) < 1e-10 or np.std(x_j) < 1e-10:
        return 0.0
    
    corr = np.corrcoef(x_i, x_j)[0, 1]
    return 0.0 if np.isnan(corr) else corr


def compute_spearman_channel(
    x_i: np.ndarray, 
    x_j: np.ndarray
) -> float:
    """
    计算两个样本在单个通道上的 Spearman 秩相关
    
    Args:
        x_i: 样本 i 的时间序列 (T,)
        x_j: 样本 j 的时间序列 (T,)
    
    Returns:
        Spearman 相关系数 (如果无法计算则返回 0)
    """
    corr, _ = spearmanr(x_i, x_j)
    return 0.0 if np.isnan(corr) else corr


def compute_rbf_similarity(
    x_i: np.ndarray, 
    x_j: np.ndarray, 
    gamma: float = 1.0
) -> float:
    """
    计算两个样本的 RBF 核相似度
    
    RBF(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2)
    
    Args:
        x_i: 样本 i 的特征向量
        x_j: 样本 j 的特征向量
        gamma: RBF 核参数
    
    Returns:
        RBF 相似度值 [0, 1]
    """
    diff = x_i - x_j
    dist_sq = np.dot(diff, diff)
    return np.exp(-gamma * dist_sq)


def compute_similarity_matrix_pearson(
    x_data: np.ndarray,
    chunk_size: int = 500,
    use_float32: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """
    计算 Pearson 相似度矩阵 (优化版本，使用向量化计算)
    
    对每对样本 (i, j)，计算所有通道的平均 Pearson 相关系数:
    P_ij = (1/C) * Σ_c corr(X_i^c, X_j^c)
    
    Args:
        x_data: 输入数据 (N, C, T)
        chunk_size: 分块大小
        use_float32: 是否使用 float32 精度
    
    Returns:
        相似度矩阵 P (N, N)
    """
    N, C, T = x_data.shape
    dtype = np.float32 if use_float32 else np.float64
    
    if show_progress:
        console.print(f"[cyan]计算 Pearson 相似度矩阵: N={N}, C={C}, T={T}[/cyan]")
        console.print("[yellow]注意: 使用原始序列计算，不进行预先标准化[/yellow]")
    
    # 初始化结果矩阵
    P = np.zeros((N, N), dtype=dtype)
    
    # 转换数据类型
    x_data = x_data.astype(dtype)
    
    # 分块计算相关矩阵
    n_chunks = (N + chunk_size - 1) // chunk_size
    total_blocks = n_chunks * (n_chunks + 1) // 2  # 只计算上三角

    progress = None
    calc_task = None
    if show_progress:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        )
        progress.start()
        calc_task = progress.add_task("[green]计算相似度...", total=total_blocks)

    try:
        for i_chunk in range(n_chunks):
            i_start = i_chunk * chunk_size
            i_end = min((i_chunk + 1) * chunk_size, N)

            for j_chunk in range(i_chunk, n_chunks):
                j_start = j_chunk * chunk_size
                j_end = min((j_chunk + 1) * chunk_size, N)

                # 对每个通道计算相关矩阵，然后取平均
                block_sum = np.zeros((i_end - i_start, j_end - j_start), dtype=dtype)

                for c in range(C):
                    # 使用原始数据，在计算时进行标准化（向量化方式）
                    X_i = x_data[i_start:i_end, c, :]  # (chunk_i, T)
                    X_j = x_data[j_start:j_end, c, :]  # (chunk_j, T)

                    # 计算每个样本的均值和标准差
                    mean_i = np.mean(X_i, axis=1, keepdims=True)  # (chunk_i, 1)
                    mean_j = np.mean(X_j, axis=1, keepdims=True)  # (chunk_j, 1)
                    std_i = np.std(X_i, axis=1, keepdims=True)   # (chunk_i, 1)
                    std_j = np.std(X_j, axis=1, keepdims=True)   # (chunk_j, 1)

                    # 标准化（零均值，单位方差）
                    # 使用 np.divide 并设置 where 参数，避免除以接近零的值
                    X_i_centered = X_i - mean_i
                    X_j_centered = X_j - mean_j

                    # 只在标准差足够大时进行除法，否则设为0
                    X_i_norm = np.divide(X_i_centered, std_i, out=np.zeros_like(X_i_centered), where=std_i > 1e-10)
                    X_j_norm = np.divide(X_j_centered, std_j, out=np.zeros_like(X_j_centered), where=std_j > 1e-10)

                    # 计算相关系数矩阵 = X_i_norm @ X_j_norm.T / T
                    corr_block = X_i_norm @ X_j_norm.T / T

                    block_sum += corr_block

                # 取通道平均
                block_avg = block_sum / C

                # 填充到 P 矩阵
                P[i_start:i_end, j_start:j_end] = block_avg

                # 对称填充 (如果不是对角块)
                if i_chunk != j_chunk:
                    P[j_start:j_end, i_start:i_end] = block_avg.T

                if progress is not None and calc_task is not None:
                    progress.update(calc_task, advance=1)
    finally:
        if progress is not None:
            progress.stop()
    
    # 对角线设为 1
    np.fill_diagonal(P, 1.0)
    
    return P


def compute_similarity_matrix_spearman(
    x_data: np.ndarray,
    chunk_size: int = 500,
    use_float32: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """
    计算 Spearman 秩相关相似度矩阵
    
    对每对样本 (i, j)，计算所有通道的平均 Spearman 相关系数
    
    Args:
        x_data: 输入数据 (N, C, T)
        chunk_size: 分块大小
        use_float32: 是否使用 float32 精度
    
    Returns:
        相似度矩阵 P (N, N)
    """
    N, C, T = x_data.shape
    dtype = np.float32 if use_float32 else np.float64
    
    if show_progress:
        console.print(f"[cyan]计算 Spearman 相似度矩阵: N={N}, C={C}, T={T}[/cyan]")
    
    x_ranked = np.zeros_like(x_data, dtype=dtype)
    
    progress = None
    rank_task = None
    if show_progress:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        )
        progress.start()
        rank_task = progress.add_task("[green]计算秩次...", total=N * C)

    try:
        for i in range(N):
            for c in range(C):
                # 使用 scipy.stats.rankdata 或手动计算秩
                from scipy.stats import rankdata
                x_ranked[i, c, :] = rankdata(x_data[i, c, :])
                if progress is not None and rank_task is not None:
                    progress.update(rank_task, advance=1)
    finally:
        if progress is not None:
            progress.stop()
    
    # 使用 Pearson 方法计算秩数据的相关矩阵
    return compute_similarity_matrix_pearson(x_ranked, chunk_size, use_float32, show_progress=show_progress)


def compute_similarity_matrix_rbf(
    x_data: np.ndarray,
    gamma: float = 1.0,
    chunk_size: int = 500,
    use_float32: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """
    计算 RBF 核相似度矩阵
    
    RBF(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2)
    
    Args:
        x_data: 输入数据 (N, C, T)
        gamma: RBF 核参数
        chunk_size: 分块大小
        use_float32: 是否使用 float32 精度
    
    Returns:
        相似度矩阵 P (N, N)
    """
    N, C, T = x_data.shape
    dtype = np.float32 if use_float32 else np.float64
    
    if show_progress:
        console.print(f"[cyan]计算 RBF 相似度矩阵: N={N}, C={C}, T={T}, gamma={gamma}[/cyan]")
    
    # 将数据展平为 (N, C*T)
    x_flat = x_data.reshape(N, -1).astype(dtype)
    
    # 初始化结果矩阵
    P = np.zeros((N, N), dtype=dtype)
    
    n_chunks = (N + chunk_size - 1) // chunk_size
    total_blocks = n_chunks * (n_chunks + 1) // 2

    progress = None
    calc_task = None
    if show_progress:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        )
        progress.start()
        calc_task = progress.add_task("[green]计算 RBF 相似度...", total=total_blocks)

    try:
        for i_chunk in range(n_chunks):
            i_start = i_chunk * chunk_size
            i_end = min((i_chunk + 1) * chunk_size, N)

            for j_chunk in range(i_chunk, n_chunks):
                j_start = j_chunk * chunk_size
                j_end = min((j_chunk + 1) * chunk_size, N)

                # 计算欧氏距离的平方
                X_i = x_flat[i_start:i_end]
                X_j = x_flat[j_start:j_end]

                # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i · x_j
                sq_i = np.sum(X_i ** 2, axis=1, keepdims=True)
                sq_j = np.sum(X_j ** 2, axis=1, keepdims=True)
                dist_sq = sq_i + sq_j.T - 2 * (X_i @ X_j.T)
                dist_sq = np.maximum(dist_sq, 0)  # 避免数值误差导致负值

                # RBF 核
                rbf_block = np.exp(-gamma * dist_sq)

                P[i_start:i_end, j_start:j_end] = rbf_block

                if i_chunk != j_chunk:
                    P[j_start:j_end, i_start:i_end] = rbf_block.T

                if progress is not None and calc_task is not None:
                    progress.update(calc_task, advance=1)
    finally:
        if progress is not None:
            progress.stop()
    
    return P


def compute_similarity_matrix(
    x_data: np.ndarray,
    method: Literal["pearson", "spearman", "rbf"] = "pearson",
    chunk_size: int = 500,
    use_float32: bool = True,
    rbf_gamma: float = 1.0,
    show_progress: bool = True,
) -> np.ndarray:
    """
    统一的相似度矩阵计算入口
    
    Args:
        x_data: 输入数据 (N, C, T)
        method: 相似度计算方法
        chunk_size: 分块大小
        use_float32: 是否使用 float32 精度
        rbf_gamma: RBF 核参数 (仅当 method="rbf" 时使用)
    
    Returns:
        相似度矩阵 P (N, N)
    """
    if method == "pearson":
        return compute_similarity_matrix_pearson(x_data, chunk_size, use_float32, show_progress=show_progress)
    elif method == "spearman":
        return compute_similarity_matrix_spearman(x_data, chunk_size, use_float32, show_progress=show_progress)
    elif method == "rbf":
        return compute_similarity_matrix_rbf(x_data, rbf_gamma, chunk_size, use_float32, show_progress=show_progress)
    else:
        raise ValueError(f"未知的相似度计算方法: {method}")


def compute_similarity_matrix_2d(
    x_data: np.ndarray,
    method: Literal["pearson", "spearman", "rbf"] = "pearson",
    chunk_size: int = 500,
    rbf_gamma: float = 1.0,
    verbose: bool = True
) -> np.ndarray:
    """
    计算 2D 数据的相似度矩阵 (N, T) -> (N, N)
    
    Args:
        x_data: 输入数据 (N, T)
        method: 相似度方法
        chunk_size: 分块大小
        rbf_gamma: RBF 核参数
        verbose: 是否打印进度
        
    Returns:
        相似度矩阵 (N, N)
    """
    N, T = x_data.shape
    # 将 (N, T) 到 (N, 1, T)
    x_data_3d = x_data[:, np.newaxis, :]
    return compute_similarity_matrix(x_data_3d, method, chunk_size, rbf_gamma=rbf_gamma, show_progress=verbose)


if __name__ == "__main__":
    # 简单测试
    console.print("[bold green]测试相似度计算模块[/bold green]")
    
    # 生成测试数据
    np.random.seed(42)
    test_data = np.random.randn(100, 18, 256)  # 100 样本, 18 通道, 256 时间点
    
    # 测试 Pearson
    P_pearson = compute_similarity_matrix(test_data, method="pearson", chunk_size=50)
    console.print(f"Pearson 矩阵形状: {P_pearson.shape}")
    console.print(f"对角线值 (应为1): {P_pearson[0, 0]:.4f}, {P_pearson[50, 50]:.4f}")
    
    console.print("[bold green]测试完成![/bold green]")
