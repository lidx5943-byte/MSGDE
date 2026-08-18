import numpy as np
from typing import List, Tuple

def remove_outliers_mad(data, threshold=8.0, replace_with="median"):
    """基于MAD的异常值去除，确保threshold为浮点数"""
    threshold = float(threshold)
    cleaned_data = np.asarray(data, dtype=np.float64).copy()
    median = np.median(cleaned_data, axis=-1, keepdims=True)
    mad = np.median(np.abs(cleaned_data - median), axis=-1, keepdims=True)
    mad[mad == 0] = 1e-6
    z_score = np.abs((cleaned_data - median) / mad)
    outlier_mask = z_score > threshold
    if replace_with == "median":
        median_broadcast = np.broadcast_to(median, cleaned_data.shape)
        cleaned_data[outlier_mask] = median_broadcast[outlier_mask]
    elif replace_with == "nan":
        cleaned_data[outlier_mask] = np.nan
    else:
        raise ValueError(f"未知的替换策略: {replace_with}")
    return cleaned_data

def detect_bad_trials(data, min_variance=1e-6, max_amplitude=None, nan_threshold=0.1):
    min_variance = float(min_variance)
    if max_amplitude is not None:
        max_amplitude = float(max_amplitude)
    nan_threshold = float(nan_threshold)
    data = np.asarray(data, dtype=np.float64)
    n_trials = data.shape[0]
    bad_indices = []
    for i in range(n_trials):
        trial = data[i]
        is_bad = False
        if np.all(np.isnan(trial)):
            is_bad = True
        elif np.all(trial == 0):
            is_bad = True
        elif np.nanvar(trial) < min_variance:
            is_bad = True
        elif max_amplitude is not None and np.nanmax(np.abs(trial)) > max_amplitude:
            is_bad = True
        elif np.sum(np.isnan(trial)) / trial.size > nan_threshold:
            is_bad = True
        if is_bad:
            bad_indices.append(i)
    return bad_indices

def remove_bad_trials(data, labels=None, min_variance=1e-6, max_amplitude=None, nan_threshold=0.1):
    bad_indices = detect_bad_trials(data, min_variance, max_amplitude, nan_threshold)
    n_trials = data.shape[0]
    good_indices = [i for i in range(n_trials) if i not in bad_indices]
    cleaned_data = data[good_indices]
    if labels is not None:
        cleaned_labels = labels[good_indices]
    else:
        cleaned_labels = None
    return cleaned_data, cleaned_labels, bad_indices

def is_bad_trial(trial, min_variance=1e-6):
    min_variance = float(min_variance)
    trial = np.asarray(trial, dtype=np.float64)
    if np.all(np.isnan(trial)) or np.all(trial == 0):
        return True
    if np.nanstd(trial) < np.sqrt(min_variance):
        return True
    return False
