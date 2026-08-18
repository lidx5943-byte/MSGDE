"""
数据标准化模块
==============

提供EEG信号标准化功能。

使用示例
--------
>>> from src.preprocessing.standardize import trialwise_standardize
>>> 
>>> # 按trial标准化
>>> standardized_data = trialwise_standardize(data)
"""

import numpy as np


def trialwise_standardize(
    data: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    按trial和通道进行标准化（零均值单位方差）
    
    对每个trial的每个通道分别进行标准化，使得每个通道的时间序列
    均值为0、标准差为1。这种标准化方法保留了trial内的相对时间模式。
    
    参数
    ----
    data : np.ndarray
        输入EEG数据，形状为 (n_trials, n_channels, n_times) 或 (n_channels, n_times)
    eps : float
        防止除零的极小值，默认1e-6
        
    返回
    ----
    np.ndarray
        标准化后的数据，形状与输入相同
        
    数学原理
    --------
    对于每个trial i 和通道 j：
    X_std[i, j, t] = (X[i, j, t] - mean_t(X[i, j, :])) / std_t(X[i, j, :])
    
    其中 mean_t 和 std_t 是沿时间轴的均值和标准差。
    """
    # 计算沿时间轴的均值和标准差
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    
    # 防止除零
    std[std < eps] = eps
    
    # 标准化：Z = (X - μ) / σ
    standardized_data = (data - mean) / std
    
    return standardized_data


def global_standardize(
    data: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    全局标准化
    
    使用所有数据的均值和标准差进行标准化。
    
    参数
    ----
    data : np.ndarray
        输入数据
    eps : float
        防止除零的极小值
        
    返回
    ----
    np.ndarray
        标准化后的数据
    """
    mean = np.mean(data)
    std = np.std(data)
    
    if std < eps:
        std = eps
    
    return (data - mean) / std


def minmax_normalize(
    data: np.ndarray,
    feature_range: tuple = (0, 1),
    axis: int = -1,
) -> np.ndarray:
    """
    最小-最大归一化
    
    将数据缩放到指定范围。
    
    参数
    ----
    data : np.ndarray
        输入数据
    feature_range : tuple
        目标范围，默认(0, 1)
    axis : int
        沿哪个轴归一化，默认-1（时间轴）
        
    返回
    ----
    np.ndarray
        归一化后的数据
    """
    min_val = np.min(data, axis=axis, keepdims=True)
    max_val = np.max(data, axis=axis, keepdims=True)
    
    # 防止除零
    range_val = max_val - min_val
    range_val[range_val == 0] = 1
    
    # 归一化到[0, 1]
    normalized = (data - min_val) / range_val
    
    # 缩放到目标范围
    a, b = feature_range
    normalized = normalized * (b - a) + a
    
    return normalized


def robust_standardize(
    data: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """
    鲁棒标准化
    
    使用中位数和四分位距进行标准化，对异常值更鲁棒。
    
    参数
    ----
    data : np.ndarray
        输入数据
    axis : int
        沿哪个轴标准化
        
    返回
    ----
    np.ndarray
        标准化后的数据
        
    数学原理
    --------
    X_robust = (X - median(X)) / IQR(X)
    其中 IQR = Q3 - Q1 是四分位距
    """
    median = np.median(data, axis=axis, keepdims=True)
    q1 = np.percentile(data, 25, axis=axis, keepdims=True)
    q3 = np.percentile(data, 75, axis=axis, keepdims=True)
    iqr = q3 - q1
    
    # 防止除零
    iqr[iqr == 0] = 1e-6
    
    return (data - median) / iqr

