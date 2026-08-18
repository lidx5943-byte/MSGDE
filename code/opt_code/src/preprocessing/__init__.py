"""
数据预处理模块
==============

提供EEG信号的标准预处理功能：
- filters: 带通滤波和陷波滤波
- baseline: 基线校正
- outliers: 异常值检测和处理
- standardize: 数据标准化
- pipeline: 完整预处理流水线
"""

from .filters import bandpass_filter, notch_filter, eeg_filter
from .baseline import baseline_correction
from .outliers import remove_outliers_mad, detect_bad_trials
from .standardize import trialwise_standardize
from .pipeline import PreprocessingPipeline

__all__ = [
    # filters
    "bandpass_filter",
    "notch_filter", 
    "eeg_filter",
    # baseline
    "baseline_correction",
    # outliers
    "remove_outliers_mad",
    "detect_bad_trials",
    # standardize
    "trialwise_standardize",
    # pipeline
    "PreprocessingPipeline",
]

