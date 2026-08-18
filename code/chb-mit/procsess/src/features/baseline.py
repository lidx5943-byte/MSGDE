# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""
基准特征提取模块

从原始信号提取基础统计特征 (Mean, Std, Max, Min, Skewness, Kurtosis)
"""

import numpy as np
from scipy.stats import skew, kurtosis
from concurrent.futures import ProcessPoolExecutor
from ..utils.console import console

def extract_baseline_features_single(sig: np.ndarray) -> np.ndarray:
    """提取单个信号的基础统计特征 (6维)"""
    return np.array([
        np.mean(sig), 
        np.std(sig), 
        np.max(sig), 
        np.min(sig), 
        skew(sig, axis=None), 
        kurtosis(sig, axis=None)
    ], dtype=np.float32)

def extract_baseline_features_batch(x_raw: np.ndarray, num_workers: int = 4) -> np.ndarray:
    """
    批量提取基础统计特征
    
    Args:
        x_raw: (N, T) 或 (N, 1, T)
        
    Returns:
        features: (N, 6)
    """
    if x_raw.ndim == 3:
        x_raw = x_raw.squeeze()
        
    num_samples = x_raw.shape[0]
    features = np.zeros((num_samples, 6), dtype=np.float32)
    console.print(f"[cyan]提取基准特征 (N={num_samples})...[/cyan]")
    
    features[:, 0] = np.mean(x_raw, axis=1)
    features[:, 1] = np.std(x_raw, axis=1)
    features[:, 2] = np.max(x_raw, axis=1)
    features[:, 3] = np.min(x_raw, axis=1)
    features[:, 4] = skew(x_raw, axis=1)
    features[:, 5] = kurtosis(x_raw, axis=1)
    
    return features
