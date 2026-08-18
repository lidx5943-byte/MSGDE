import numpy as np

def trialwise_standardize(data, eps=1e-6):
    """按trial和通道标准化，确保eps为浮点数"""
    eps = float(eps)  # 强制转换为浮点数
    data = np.asarray(data, dtype=np.float64)
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    std[std < eps] = eps
    return (data - mean) / std

def global_standardize(data, eps=1e-6):
    eps = float(eps)
    data = np.asarray(data, dtype=np.float64)
    mean = np.mean(data)
    std = np.std(data)
    if std < eps:
        std = eps
    return (data - mean) / std

def minmax_normalize(data, feature_range=(0, 1), axis=-1):
    data = np.asarray(data, dtype=np.float64)
    min_val = np.min(data, axis=axis, keepdims=True)
    max_val = np.max(data, axis=axis, keepdims=True)
    range_val = max_val - min_val
    range_val[range_val == 0] = 1.0
    normalized = (data - min_val) / range_val
    a, b = feature_range
    return normalized * (b - a) + a

def robust_standardize(data, axis=-1):
    data = np.asarray(data, dtype=np.float64)
    median = np.median(data, axis=axis, keepdims=True)
    q1 = np.percentile(data, 25, axis=axis, keepdims=True)
    q3 = np.percentile(data, 75, axis=axis, keepdims=True)
    iqr = q3 - q1
    iqr[iqr == 0] = 1e-6
    return (data - median) / iqr
