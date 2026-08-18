"""
信号滤波模块
============

提供EEG信号滤波功能：
- 带通滤波：去除低频漂移和高频噪声
- 陷波滤波：去除工频干扰

使用示例
--------
>>> from src.preprocessing.filters import eeg_filter
>>> 
>>> # 应用标准EEG滤波
>>> filtered_data = eeg_filter(data, sfreq=160, low_freq=8, high_freq=12.5)
"""

import numpy as np
from scipy import signal
from typing import Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


def bandpass_filter(
    data: np.ndarray,
    sfreq: float,
    low_freq: float,
    high_freq: float,
    order: int = 4,
) -> np.ndarray:
    """
    带通滤波
    
    使用Butterworth滤波器实现带通滤波，保留指定频率范围内的信号。
    
    参数
    ----
    data : np.ndarray
        输入数据，形状为 (n_trials, n_channels, n_times) 或 (n_channels, n_times)
    sfreq : float
        采样频率 (Hz)
    low_freq : float
        下截止频率 (Hz)
    high_freq : float
        上截止频率 (Hz)
    order : int
        滤波器阶数，默认4
        
    返回
    ----
    np.ndarray
        滤波后的数据，形状与输入相同
        
    数学原理
    --------
    使用Butterworth带通滤波器，传递函数为：
    H(z) = B(z) / A(z)
    
    在频域 [low_freq, high_freq] 范围内保留信号，其他频率被衰减。
    使用filtfilt实现零相位滤波，避免相位失真。
    """
    # 计算奈奎斯特频率
    nyquist = 0.5 * sfreq
    
    # 归一化截止频率
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # 确保频率在有效范围内
    if low <= 0:
        low = 0.001
    if high >= 1:
        high = 0.999
    
    # 设计Butterworth带通滤波器
    b, a = signal.butter(order, [low, high], btype='band')
    
    # 应用零相位滤波
    filtered_data = signal.filtfilt(b, a, data, axis=-1)
    
    return filtered_data


def notch_filter(
    data: np.ndarray,
    sfreq: float,
    notch_freq: float,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """
    陷波滤波
    
    使用IIR陷波滤波器去除特定频率的干扰（如工频干扰）。
    
    参数
    ----
    data : np.ndarray
        输入数据，形状为 (n_trials, n_channels, n_times) 或 (n_channels, n_times)
    sfreq : float
        采样频率 (Hz)
    notch_freq : float
        陷波频率 (Hz)，通常为50Hz或60Hz
    quality_factor : float
        Q因子，控制陷波的宽度，默认30
        
    返回
    ----
    np.ndarray
        滤波后的数据，形状与输入相同
        
    数学原理
    --------
    使用IIR陷波滤波器，在notch_freq处形成深陷波。
    Q因子越大，陷波越窄；Q因子越小，陷波越宽。
    """
    # 计算奈奎斯特频率
    nyquist = 0.5 * sfreq
    
    # 归一化陷波频率
    w0 = notch_freq / nyquist
    
    # 确保频率在有效范围内
    if w0 >= 1:
        w0 = 0.99
    
    # 设计IIR陷波滤波器
    b, a = signal.iirnotch(w0, quality_factor)
    
    # 应用零相位滤波
    filtered_data = signal.filtfilt(b, a, data, axis=-1)
    
    return filtered_data


def eeg_filter(
    data: np.ndarray,
    sfreq: float,
    low_freq: float = 8.0,
    high_freq: float = 12.5,
    notch_freq: Optional[float] = 50.0,
    filter_order: int = 4,
    notch_q: float = 30.0,
) -> np.ndarray:
    """
    EEG信号标准滤波处理
    
    依次应用带通滤波和陷波滤波，这是EEG信号处理的标准预处理步骤。
    
    参数
    ----
    data : np.ndarray
        输入EEG数据，形状为 (n_trials, n_channels, n_times)
    sfreq : float
        采样频率 (Hz)
    low_freq : float
        带通滤波下截止频率 (Hz)，默认8Hz
    high_freq : float
        带通滤波上截止频率 (Hz)，默认12.5Hz
    notch_freq : float, optional
        陷波频率 (Hz)，默认50Hz。设为None跳过陷波滤波
    filter_order : int
        Butterworth滤波器阶数，默认4
    notch_q : float
        陷波滤波器Q因子，默认30
        
    返回
    ----
    np.ndarray
        滤波后的EEG数据，形状与输入相同
        
    注意
    ----
    - 滤波顺序：先带通滤波，再陷波滤波
    - 使用filtfilt实现零相位滤波，避免相位失真
    """
    # 步骤1：带通滤波
    filtered_data = bandpass_filter(
        data, sfreq, low_freq, high_freq, filter_order
    )
    
    # 步骤2：陷波滤波（如果指定）
    if notch_freq is not None:
        filtered_data = notch_filter(
            filtered_data, sfreq, notch_freq, notch_q
        )
    
    return filtered_data


def highpass_filter(
    data: np.ndarray,
    sfreq: float,
    cutoff_freq: float,
    order: int = 4,
) -> np.ndarray:
    """
    高通滤波
    
    参数
    ----
    data : np.ndarray
        输入数据
    sfreq : float
        采样频率 (Hz)
    cutoff_freq : float
        截止频率 (Hz)
    order : int
        滤波器阶数
        
    返回
    ----
    np.ndarray
        滤波后的数据
    """
    nyquist = 0.5 * sfreq
    normalized_cutoff = cutoff_freq / nyquist
    
    if normalized_cutoff >= 1:
        normalized_cutoff = 0.99
    
    b, a = signal.butter(order, normalized_cutoff, btype='high')
    filtered_data = signal.filtfilt(b, a, data, axis=-1)
    
    return filtered_data


def lowpass_filter(
    data: np.ndarray,
    sfreq: float,
    cutoff_freq: float,
    order: int = 4,
) -> np.ndarray:
    """
    低通滤波
    
    参数
    ----
    data : np.ndarray
        输入数据
    sfreq : float
        采样频率 (Hz)
    cutoff_freq : float
        截止频率 (Hz)
    order : int
        滤波器阶数
        
    返回
    ----
    np.ndarray
        滤波后的数据
    """
    nyquist = 0.5 * sfreq
    normalized_cutoff = cutoff_freq / nyquist
    
    if normalized_cutoff >= 1:
        normalized_cutoff = 0.99
    
    b, a = signal.butter(order, normalized_cutoff, btype='low')
    filtered_data = signal.filtfilt(b, a, data, axis=-1)
    
    return filtered_data
