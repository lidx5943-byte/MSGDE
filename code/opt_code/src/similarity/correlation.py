"""
相似度计算模块
==============

提供样本间相似度矩阵的计算功能。

使用示例
--------
>>> from src.similarity.correlation import compute_similarity_matrix
>>> 
>>> # 计算Pearson相关性矩阵
>>> similarity = compute_similarity_matrix(data, method="pearson")
"""

import numpy as np
from scipy import stats
from typing import Union, Dict, List

from ..utils.logger import get_logger, console, create_progress, print_success

logger = get_logger(__name__)


def pearson_correlation(data: np.ndarray) -> np.ndarray:
    """
    计算Pearson相关系数矩阵
    
    对于给定的EEG数据，计算所有样本对之间的Pearson相关系数。
    
    参数
    ----
    data : np.ndarray
        输入EEG数据，形状为 (n_samples, n_channels, n_times)
        注意：此函数由 compute_similarity_matrix 调用，数据已确保为3D
        
    返回
    ----
    np.ndarray
        相关性矩阵，形状为 (n_samples, n_samples)
        
    数学原理
    --------
    1. 对每个通道c，计算样本间的Pearson相关系数
    2. 对所有通道的相关系数取平均
    """
    n_samples, n_channels, n_times = data.shape
    
    # 初始化相关矩阵
    correlation_matrix = np.zeros((n_samples, n_samples))
    valid_channels = 0
    
    # 对每个通道计算相关性
    with create_progress() as progress:
        task = progress.add_task("计算Pearson相关", total=n_channels)
        
        for c in range(n_channels):
            # 提取该通道所有样本的时间序列
            channel_data = data[:, c, :]  # (n_samples, n_times)
            
            # 检查通道是否有有效方差（避免全0或常数序列）
            channel_std = channel_data.std(axis=1)
            if (channel_std < 1e-10).all():
                # 如果所有样本在这个通道上都是常数，跳过该通道
                progress.update(task, advance=1)
                continue
            
            # 标准化（添加小值避免除零）
            channel_mean = channel_data.mean(axis=1, keepdims=True)
            channel_std = channel_data.std(axis=1, keepdims=True) + 1e-8
            channel_data_z = (channel_data - channel_mean) / channel_std
            
            # 计算相关矩阵
            channel_corr = np.corrcoef(channel_data_z)
            
            # 处理NaN值（如果某个样本的方差为0，corrcoef会返回NaN）
            if np.isnan(channel_corr).any():
                # 将对角线设为0
                np.fill_diagonal(channel_corr, 0.0)
                # 将其他NaN值替换为0（表示无相关性）
                channel_corr = np.nan_to_num(channel_corr, nan=0.0)
            
            # 累加
            correlation_matrix += channel_corr
            valid_channels += 1
            
            progress.update(task, advance=1)
    
    # 如果所有通道都无效，返回零矩阵
    if valid_channels == 0:
        console.print("[yellow]警告: 所有通道都无效（全为常数），返回零矩阵[/yellow]")
        return np.zeros((n_samples, n_samples))
    
    # 取平均
    correlation_matrix /= valid_channels
    
    # 最终验证：确保对角线为0
    np.fill_diagonal(correlation_matrix, 0.0)
    
    return correlation_matrix


def spearman_correlation(data: np.ndarray) -> np.ndarray:
    """
    计算Spearman秩相关系数矩阵
    
    参数
    ----
    data : np.ndarray
        输入EEG数据，形状为 (n_samples, n_channels, n_times)
        注意：此函数由 compute_similarity_matrix 调用，数据已确保为3D
        
    返回
    ----
    np.ndarray
        相关性矩阵，形状为 (n_samples, n_samples)
    """
    n_samples, n_channels, n_times = data.shape
    correlation_matrix = np.zeros((n_samples, n_samples))
    valid_channels = 0
    
    with create_progress() as progress:
        task = progress.add_task("计算Spearman相关", total=n_channels)
        
        for c in range(n_channels):
            channel_data = data[:, c, :]
            
            # 检查通道是否有有效方差
            channel_std = channel_data.std(axis=1)
            if (channel_std < 1e-10).all():
                # 如果所有样本在这个通道上都是常数，跳过该通道
                progress.update(task, advance=1)
                continue
            
            # 对每个样本计算秩
            ranked_data = np.apply_along_axis(stats.rankdata, 1, channel_data)
            
            # 计算秩的相关矩阵
            channel_corr = np.corrcoef(ranked_data)
            
            # 处理NaN值
            if np.isnan(channel_corr).any():
                np.fill_diagonal(channel_corr, 0.0)
                channel_corr = np.nan_to_num(channel_corr, nan=0.0)
            
            correlation_matrix += channel_corr
            valid_channels += 1
            
            progress.update(task, advance=1)
    
    # 如果所有通道都无效，返回零矩阵
    if valid_channels == 0:
        console.print("[yellow]警告: 所有通道都无效（全为常数），返回零矩阵[/yellow]")
        return np.zeros((n_samples, n_samples))
    
    correlation_matrix /= valid_channels
    
    # 确保对角线为0
    np.fill_diagonal(correlation_matrix, 0.0)
    
    return correlation_matrix


def cosine_similarity(data: np.ndarray) -> np.ndarray:
    """
    计算余弦相似度矩阵
    
    参数
    ----
    data : np.ndarray
        输入EEG数据，形状为 (n_samples, n_channels, n_times)
        注意：此函数由 compute_similarity_matrix 调用，数据已确保为3D
        
    返回
    ----
    np.ndarray
        相似度矩阵，形状为 (n_samples, n_samples)
    """
    n_samples, n_channels, n_times = data.shape
    similarity_matrix = np.zeros((n_samples, n_samples))
    valid_channels = 0
    
    with create_progress() as progress:
        task = progress.add_task("计算余弦相似度", total=n_channels)
        
        for c in range(n_channels):
            channel_data = data[:, c, :]
            
            # 计算范数
            norms = np.linalg.norm(channel_data, axis=1, keepdims=True)
            norms[norms == 0] = 1e-8  # 防止除零，使用更小的值
            
            # 归一化
            normalized = channel_data / norms
            
            # 计算余弦相似度
            channel_sim = np.dot(normalized, normalized.T)
            
            # 处理可能的NaN值
            if np.isnan(channel_sim).any():
                np.fill_diagonal(channel_sim, 0.0)
                channel_sim = np.nan_to_num(channel_sim, nan=0.0)
            
            similarity_matrix += channel_sim
            valid_channels += 1
            
            progress.update(task, advance=1)
    
    # 如果所有通道都无效，返回零矩阵
    if valid_channels == 0:
        console.print("[yellow]警告: 所有通道都无效，返回零矩阵[/yellow]")
        return np.zeros((n_samples, n_samples))
    
    similarity_matrix /= valid_channels
    
    # 确保对角线为0
    np.fill_diagonal(similarity_matrix, 0.0)
    
    return similarity_matrix


def rbf_kernel_similarity(
    data: np.ndarray,
    gamma: Union[str, float] = "scale",
) -> np.ndarray:
    """
    计算RBF核相似度矩阵
    
    参数
    ----
    data : np.ndarray
        输入EEG数据，形状为 (n_samples, n_channels, n_times)
        注意：此函数由 compute_similarity_matrix 调用，数据已确保为3D
    gamma : str or float
        RBF核参数，"scale"表示使用 1/(n_features * var(X))
        
    返回
    ----
    np.ndarray
        相似度矩阵，形状为 (n_samples, n_samples)
    """
    n_samples, n_channels, n_times = data.shape
    similarity_matrix = np.zeros((n_samples, n_samples))
    valid_channels = 0
    
    with create_progress() as progress:
        task = progress.add_task("计算RBF核相似度", total=n_channels)
        
        for c in range(n_channels):
            channel_data = data[:, c, :]  # (n_samples, n_times)
            
            # 计算gamma
            if gamma == "scale":
                var_ch = channel_data.var()
                if var_ch > 0:
                    gamma_value = 1.0 / (n_times * var_ch)
                else:
                    # 如果方差为0，使用默认值
                    gamma_value = 1.0
            else:
                gamma_value = gamma
            
            # 计算欧氏距离的平方
            sq_dists = np.sum(channel_data**2, axis=1, keepdims=True) + \
                       np.sum(channel_data**2, axis=1) - \
                       2 * np.dot(channel_data, channel_data.T)
            
            # 确保非负
            sq_dists = np.maximum(sq_dists, 0)
            
            # 计算RBF核
            channel_sim = np.exp(-gamma_value * sq_dists)
            
            # 处理可能的NaN或Inf值
            if np.isnan(channel_sim).any() or np.isinf(channel_sim).any():
                np.fill_diagonal(channel_sim, 0.0)
                channel_sim = np.nan_to_num(channel_sim, nan=0.0, posinf=1.0, neginf=0.0)
            
            similarity_matrix += channel_sim
            valid_channels += 1
            
            progress.update(task, advance=1)
    
    # 如果所有通道都无效，返回零矩阵
    if valid_channels == 0:
        console.print("[yellow]警告: 所有通道都无效，返回零矩阵[/yellow]")
        return np.zeros((n_samples, n_samples))
    
    similarity_matrix /= valid_channels
    
    # 确保对角线为0
    np.fill_diagonal(similarity_matrix, 0.0)
    
    return similarity_matrix


def laplacian_kernel_similarity(
    data: np.ndarray,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    计算拉普拉斯核相似度矩阵
    
    参数
    ----
    data : np.ndarray
        输入EEG数据，形状为 (n_samples, n_channels, n_times)
        注意：此函数由 compute_similarity_matrix 调用，数据已确保为3D
    gamma : float
        核参数
        
    返回
    ----
    np.ndarray
        相似度矩阵，形状为 (n_samples, n_samples)
    """
    n_samples, n_channels, n_times = data.shape
    similarity_matrix = np.zeros((n_samples, n_samples))
    valid_channels = 0
    
    with create_progress() as progress:
        task = progress.add_task("计算拉普拉斯核", total=n_channels)
        
        for c in range(n_channels):
            channel_data = data[:, c, :]  # (n_samples, n_times)
            
            # 计算曼哈顿距离
            channel_sim = np.zeros((n_samples, n_samples))
            
            for i in range(n_samples):
                # 计算样本i与所有样本的曼哈顿距离
                distances = np.sum(np.abs(channel_data - channel_data[i]), axis=1)
                channel_sim[i] = np.exp(-gamma * distances)
            
            # 处理可能的NaN或Inf值
            if np.isnan(channel_sim).any() or np.isinf(channel_sim).any():
                np.fill_diagonal(channel_sim, 0.0)
                channel_sim = np.nan_to_num(channel_sim, nan=0.0, posinf=1.0, neginf=0.0)
            
            similarity_matrix += channel_sim
            valid_channels += 1
            
            progress.update(task, advance=1)
    
    # 如果所有通道都无效，返回零矩阵
    if valid_channels == 0:
        console.print("[yellow]警告: 所有通道都无效，返回零矩阵[/yellow]")
        return np.zeros((n_samples, n_samples))
    
    similarity_matrix /= valid_channels
    
    # 确保对角线为0
    np.fill_diagonal(similarity_matrix, 0.0)
    
    return similarity_matrix


def compute_similarity_matrix(
    data: np.ndarray,
    method: str = "pearson",
    **kwargs,
) -> np.ndarray:
    """
    计算样本间相似度矩阵
    
    参数
    ----
    data : np.ndarray
        输入EEG数据，形状为 (n_samples, n_channels, n_times) 或 (n_samples, n_times)
        如果是2D数组，会自动添加通道维度（视为单通道数据）
    method : str
        相似度方法：
        - "pearson": Pearson相关系数
        - "spearman": Spearman秩相关
        - "cosine": 余弦相似度
        - "rbf_kernel": RBF核相似度
        - "laplacian_kernel": 拉普拉斯核相似度
    **kwargs
        方法特定参数
        
    返回
    ----
    np.ndarray
        相似度矩阵，形状为 (n_samples, n_samples)
    """
    console.print(f"[dim]计算相似度矩阵 (方法: {method})[/dim]")
    
    # 兼容2D数据（单通道情况）
    if data.ndim == 2:
        # 数据形状为 (n_samples, n_times)，添加通道维度
        console.print(f"[dim]检测到2D数据 (形状: {data.shape})，添加通道维度...[/dim]")
        data = data[:, np.newaxis, :]  # 扩展为 (n_samples, 1, n_times)
        console.print(f"[dim]扩展后形状: {data.shape}[/dim]")
    elif data.ndim != 3:
        raise ValueError(
            f"数据维度错误: 期望2D (n_samples, n_times) 或 3D (n_samples, n_channels, n_times)，"
            f"但得到 {data.ndim}D 数组，形状为 {data.shape}"
        )
    
    methods = {
        "pearson": pearson_correlation,
        "spearman": spearman_correlation,
        "cosine": cosine_similarity,
        "rbf_kernel": lambda d: rbf_kernel_similarity(d, kwargs.get("gamma", "scale")),
        "laplacian_kernel": lambda d: laplacian_kernel_similarity(d, kwargs.get("gamma", 1.0)),
    }
    
    if method not in methods:
        raise ValueError(f"未知的相似度方法: {method}. 可用: {list(methods.keys())}")
    
    similarity = methods[method](data)
    
    # 最终验证：确保没有NaN或Inf值
    if np.isnan(similarity).any() or np.isinf(similarity).any():
        console.print("[yellow]警告: 相似度矩阵包含NaN或Inf值，正在清理...[/yellow]")
        np.fill_diagonal(similarity, 0.0)
        similarity = np.nan_to_num(similarity, nan=0.0, posinf=1.0, neginf=-1.0)
        # 确保值在合理范围内（对于相关性矩阵，应该在[-1, 1]之间）
        if method in ["pearson", "spearman"]:
            similarity = np.clip(similarity, -1.0, 1.0)
    
    print_success(f"相似度矩阵形状: {similarity.shape}")
    
    return similarity


def compute_multiple_similarities(
    data: np.ndarray,
    methods: List[str] = None,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """
    计算多种相似度矩阵
    
    参数
    ----
    data : np.ndarray
        输入EEG数据，形状为 (n_samples, n_channels, n_times) 或 (n_samples, n_times)
        如果是2D数组，会自动添加通道维度（视为单通道数据）
    methods : List[str]
        要计算的方法列表
    **kwargs
        方法特定参数
        
    返回
    ----
    Dict[str, np.ndarray]
        方法名到相似度矩阵的映射
    """
    if methods is None:
        methods = ["pearson"]
    
    results = {}
    for method in methods:
        results[method] = compute_similarity_matrix(data, method, **kwargs)
    
    return results

