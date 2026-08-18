"""
基线校正模块
============

提供EEG信号基线校正功能。

使用示例
--------
>>> from src.preprocessing.baseline import baseline_correction
>>> 
>>> # 应用基线校正（使用前0.2秒作为基线）
>>> corrected_data = baseline_correction(data, sfreq=160, baseline_duration=0.2)
"""

import numpy as np
from typing import Optional


def baseline_correction(
    data: np.ndarray,
    sfreq: float = None,
    baseline_duration: float = None,
    baseline_samples: int = None,
) -> np.ndarray:
    """
    基线校正
    
    通过减去基线期的平均值，消除信号中的直流偏移和缓慢漂移。
    这是事件相关电位(ERP)分析的标准预处理步骤。
    
    参数
    ----
    data : np.ndarray
        输入EEG数据，形状为 (n_trials, n_channels, n_times) 或 (n_channels, n_times)
    sfreq : float, optional
        采样频率 (Hz)，与baseline_duration配合使用
    baseline_duration : float, optional
        基线时长（秒），与sfreq配合使用
    baseline_samples : int, optional
        基线采样点数，直接指定时忽略sfreq和baseline_duration
        
    返回
    ----
    np.ndarray
        基线校正后的数据，形状与输入相同
        
    数学原理
    --------
    对于每个trial i 和通道 j，基线校正公式为：
    X_corrected[i, j, t] = X[i, j, t] - mean(X[i, j, 0:baseline_samples])
    
    这样处理后，每个trial的基线期均值为零。
    
    异常
    ----
    ValueError
        如果参数不足以确定基线采样点数
    """
    # 确定基线采样点数
    if baseline_samples is not None:
        n_baseline = baseline_samples
    elif sfreq is not None and baseline_duration is not None:
        n_baseline = int(baseline_duration * sfreq)
    else:
        raise ValueError(
            "必须提供 baseline_samples，或同时提供 sfreq 和 baseline_duration"
        )
    
    # 确保基线采样点数有效
    if n_baseline <= 0:
        raise ValueError(f"基线采样点数必须为正数，当前为 {n_baseline}")
    
    if n_baseline > data.shape[-1]:
        raise ValueError(
            f"基线采样点数 ({n_baseline}) 超过数据时间点数 ({data.shape[-1]})"
        )
    
    # 计算基线均值
    # 对最后一个维度（时间）的前n_baseline个点取均值
    baseline = np.mean(data[..., :n_baseline], axis=-1, keepdims=True)
    
    # 从整个时间序列中减去基线均值
    corrected_data = data - baseline
    
    return corrected_data


def baseline_correction_zscore(
    data: np.ndarray,
    sfreq: float = None,
    baseline_duration: float = None,
    baseline_samples: int = None,
) -> np.ndarray:
    """
    Z-score基线校正
    
    使用基线期的均值和标准差对数据进行标准化。
    
    参数
    ----
    data : np.ndarray
        输入EEG数据
    sfreq : float, optional
        采样频率 (Hz)
    baseline_duration : float, optional
        基线时长（秒）
    baseline_samples : int, optional
        基线采样点数
        
    返回
    ----
    np.ndarray
        Z-score校正后的数据
        
    数学原理
    --------
    X_corrected = (X - mean(baseline)) / std(baseline)
    """
    # 确定基线采样点数
    if baseline_samples is not None:
        n_baseline = baseline_samples
    elif sfreq is not None and baseline_duration is not None:
        n_baseline = int(baseline_duration * sfreq)
    else:
        raise ValueError(
            "必须提供 baseline_samples，或同时提供 sfreq 和 baseline_duration"
        )
    
    # 计算基线均值和标准差
    baseline_data = data[..., :n_baseline]
    baseline_mean = np.mean(baseline_data, axis=-1, keepdims=True)
    baseline_std = np.std(baseline_data, axis=-1, keepdims=True)
    
    # 防止除零
    baseline_std[baseline_std == 0] = 1e-6
    
    # Z-score标准化
    corrected_data = (data - baseline_mean) / baseline_std
    
    return corrected_data
