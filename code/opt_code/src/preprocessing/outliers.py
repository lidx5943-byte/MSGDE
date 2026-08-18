"""
异常值处理模块
==============

提供EEG信号异常值检测和处理功能：
- MAD方法异常值检测
- 坏trial检测

使用示例
--------
>>> from src.preprocessing.outliers import remove_outliers_mad, detect_bad_trials
>>> 
>>> # 使用MAD方法去除异常值
>>> cleaned_data = remove_outliers_mad(data, threshold=8.0)
>>> 
>>> # 检测坏trial
>>> bad_indices = detect_bad_trials(data, min_variance=1e-6)
"""

import numpy as np
from typing import List, Tuple


def remove_outliers_mad(
    data: np.ndarray,
    threshold: float = 8.0,
    replace_with: str = "median",
) -> np.ndarray:
    """
    基于中位绝对偏差(MAD)的异常值去除
    
    使用MAD方法识别异常值，并将异常值替换为该通道的中位数值。
    MAD方法对异常值具有鲁棒性，比基于标准差的方法更适合处理包含异常值的数据。
    
    参数
    ----
    data : np.ndarray
        输入EEG数据，形状为 (n_trials, n_channels, n_times) 或 (n_channels, n_times)
    threshold : float
        MAD阈值倍数，默认8.0。当标准化MAD分数超过此阈值时，判定为异常值
    replace_with : str
        替换策略，可选 "median"（中位数）或 "nan"
        
    返回
    ----
    np.ndarray
        去除异常值后的数据，形状与输入相同
        
    数学原理
    --------
    1. 对于每个trial i 和通道 j，计算时间序列的中位数 M_ij
    2. 计算中位绝对偏差：MAD_ij = median(|X_ij(t) - M_ij|)
    3. 计算标准化分数：Z_ij(t) = |X_ij(t) - M_ij| / MAD_ij
    4. 如果 Z_ij(t) > threshold，则将 X_ij(t) 替换为 M_ij
    
    注意
    ----
    - 使用中位数而非均值，对异常值具有鲁棒性
    - 避免除零：当MAD为0时设置为极小值1e-6
    """
    # 复制数据以避免修改原始数据
    cleaned_data = data.copy()
    
    # 计算中位数（沿时间轴）
    median = np.median(cleaned_data, axis=-1, keepdims=True)
    
    # 计算中位绝对偏差
    mad = np.median(np.abs(cleaned_data - median), axis=-1, keepdims=True)
    
    # 防止除零
    mad[mad == 0] = 1e-6
    
    # 计算标准化MAD分数
    z_score = np.abs((cleaned_data - median) / mad)
    
    # 识别异常值
    outlier_mask = z_score > threshold
    
    # 替换异常值
    if replace_with == "median":
        # 广播中位数到与数据相同的形状
        median_broadcast = np.broadcast_to(median, cleaned_data.shape)
        cleaned_data[outlier_mask] = median_broadcast[outlier_mask]
    elif replace_with == "nan":
        cleaned_data[outlier_mask] = np.nan
    else:
        raise ValueError(f"未知的替换策略: {replace_with}")
    
    return cleaned_data


def detect_bad_trials(
    data: np.ndarray,
    min_variance: float = 1e-6,
    max_amplitude: float = None,
    nan_threshold: float = 0.1,
) -> List[int]:
    """
    检测坏的trial
    
    识别质量差的trial，包括：
    - 全零或全NaN
    - 方差过小
    - 振幅过大
    - NaN比例过高
    
    参数
    ----
    data : np.ndarray
        输入EEG数据，形状为 (n_trials, n_channels, n_times)
    min_variance : float
        最小方差阈值，低于此值判定为坏trial
    max_amplitude : float, optional
        最大振幅阈值，超过此值判定为坏trial
    nan_threshold : float
        NaN比例阈值，超过此比例判定为坏trial
        
    返回
    ----
    List[int]
        坏trial的索引列表
    """
    n_trials = data.shape[0]
    bad_indices = []
    
    for i in range(n_trials):
        trial = data[i]
        is_bad = False
        
        # 检查是否全为NaN
        if np.all(np.isnan(trial)):
            is_bad = True
        
        # 检查是否全为零
        elif np.all(trial == 0):
            is_bad = True
        
        # 检查方差是否过小
        elif np.nanvar(trial) < min_variance:
            is_bad = True
        
        # 检查振幅是否过大
        elif max_amplitude is not None:
            if np.nanmax(np.abs(trial)) > max_amplitude:
                is_bad = True
        
        # 检查NaN比例
        nan_ratio = np.sum(np.isnan(trial)) / trial.size
        if nan_ratio > nan_threshold:
            is_bad = True
        
        if is_bad:
            bad_indices.append(i)
    
    return bad_indices


def remove_bad_trials(
    data: np.ndarray,
    labels: np.ndarray = None,
    min_variance: float = 1e-6,
    max_amplitude: float = None,
    nan_threshold: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    移除坏的trial
    
    参数
    ----
    data : np.ndarray
        输入EEG数据，形状为 (n_trials, n_channels, n_times)
    labels : np.ndarray, optional
        标签数组，形状为 (n_trials,)
    min_variance : float
        最小方差阈值
    max_amplitude : float, optional
        最大振幅阈值
    nan_threshold : float
        NaN比例阈值
        
    返回
    ----
    Tuple[np.ndarray, np.ndarray, List[int]]
        (清理后的数据, 清理后的标签, 被移除的trial索引)
    """
    # 检测坏trial
    bad_indices = detect_bad_trials(
        data, min_variance, max_amplitude, nan_threshold
    )
    
    # 生成保留的索引
    n_trials = data.shape[0]
    good_indices = [i for i in range(n_trials) if i not in bad_indices]
    
    # 保留好的trial
    cleaned_data = data[good_indices]
    
    if labels is not None:
        cleaned_labels = labels[good_indices]
    else:
        cleaned_labels = None
    
    return cleaned_data, cleaned_labels, bad_indices


def is_bad_trial(
    trial: np.ndarray,
    min_variance: float = 1e-6,
) -> bool:
    """
    判断单个trial是否为坏trial
    
    参数
    ----
    trial : np.ndarray
        单个trial的数据，形状为 (n_channels, n_times)
    min_variance : float
        最小方差阈值
        
    返回
    ----
    bool
        True表示是坏trial
    """
    # 检查是否全为NaN或全为零
    if np.all(np.isnan(trial)) or np.all(trial == 0):
        return True
    
    # 检查方差是否过小
    if np.nanstd(trial) < np.sqrt(min_variance):
        return True
    
    return False

