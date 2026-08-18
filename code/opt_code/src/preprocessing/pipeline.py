"""
预处理流水线模块
================

提供完整的EEG数据预处理流水线。

使用示例
--------
>>> from src.preprocessing.pipeline import PreprocessingPipeline
>>> from src.config import load_config
>>> 
>>> config = load_config()
>>> pipeline = PreprocessingPipeline(config)
>>> cleaned_data = pipeline.run(data)
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

from .filters import eeg_filter
from .baseline import baseline_correction
from .outliers import remove_outliers_mad, remove_bad_trials
from .standardize import trialwise_standardize

from ..utils.logger import (
    get_logger, console, create_progress,
    print_header, print_step, print_success, print_warning, print_table
)
from ..utils.timer import Timer, TimerGroup
from ..utils.io import save_numpy, load_numpy, ensure_dir


class PreprocessingPipeline:
    """
    EEG数据预处理流水线
    
    执行完整的预处理流程：
    1. 信号滤波（带通+陷波）
    2. 基线校正
    3. 异常值去除
    4. 标准化
    5. 坏trial检测和移除
    
    属性
    ----
    config : Config
        配置对象
    logger : Logger
        日志记录器
        
    使用示例
    --------
    >>> pipeline = PreprocessingPipeline(config)
    >>> cleaned_data = pipeline.run(data)
    >>> 
    >>> # 或分步执行
    >>> filtered = pipeline.filter(data)
    >>> corrected = pipeline.baseline_correct(filtered)
    """
    
    def __init__(self, config=None):
        """
        初始化预处理流水线
        
        参数
        ----
        config : Config, optional
            配置对象，如果为None则使用默认配置
        """
        self.config = config
        self.logger = get_logger("preprocessing")
        self._stats = {}
    
    def run(
        self,
        data: np.ndarray,
        labels: np.ndarray = None,
        sfreq: float = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """
        执行完整的预处理流程
        
        参数
        ----
        data : np.ndarray
            输入EEG数据，形状为 (n_trials, n_channels, n_times)
        labels : np.ndarray, optional
            标签数组
        sfreq : float, optional
            采样频率，如果为None则从配置中读取
            
        返回
        ----
        Tuple[np.ndarray, np.ndarray, Dict]
            (清理后的数据, 清理后的标签, 统计信息字典)
        """
        print_header("EEG数据预处理")
        
        timers = TimerGroup("预处理")
        timers.start_total()
        
        # 获取配置参数
        if self.config is not None:
            preproc_config = self.config.preprocessing
            if sfreq is None:
                sfreq = preproc_config.sampling_rate
        else:
            if sfreq is None:
                sfreq = 160  # 默认采样率
        
        original_shape = data.shape
        console.print(f"[dim]输入数据形状: {original_shape}[/dim]")
        
        total_steps = 5
        
        # 步骤1：信号滤波
        print_step(1, total_steps, "信号滤波（带通+陷波）")
        with timers.timer("信号滤波"):
            data = self._filter(data, sfreq)
        
        # 步骤2：基线校正
        print_step(2, total_steps, "基线校正")
        with timers.timer("基线校正"):
            data = self._baseline_correct(data, sfreq)
        
        # 步骤3：异常值去除
        print_step(3, total_steps, "异常值去除")
        with timers.timer("异常值去除"):
            data = self._remove_outliers(data)
        
        # 步骤4：标准化
        print_step(4, total_steps, "数据标准化")
        with timers.timer("标准化"):
            data = self._standardize(data)
        
        # 步骤5：坏trial移除
        print_step(5, total_steps, "坏trial移除")
        with timers.timer("坏trial移除"):
            data, labels, bad_indices = self._remove_bad_trials(data, labels)
        
        timers.stop_total()
        
        # 统计信息
        self._stats = {
            "original_shape": original_shape,
            "final_shape": data.shape,
            "n_removed_trials": len(bad_indices),
            "bad_trial_indices": bad_indices,
        }
        
        # 打印统计
        print_success("预处理完成")
        stats_rows = [
            ["原始形状", str(original_shape)],
            ["最终形状", str(data.shape)],
            ["移除trial数", str(len(bad_indices))],
        ]
        print_table("预处理统计", ["项目", "值"], stats_rows)
        
        timers.report()
        
        return data, labels, self._stats
    
    def _filter(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        """应用滤波"""
        if self.config is not None:
            filter_config = self.config.preprocessing.filter
            notch_config = self.config.preprocessing.notch
            
            low_freq = filter_config.low_freq
            high_freq = filter_config.high_freq
            filter_order = filter_config.order
            notch_freq = notch_config.freq
            notch_q = notch_config.quality_factor
        else:
            low_freq = 8.0
            high_freq = 12.5
            filter_order = 4
            notch_freq = 50.0
            notch_q = 30
        
        console.print(f"  [dim]带通: {low_freq}-{high_freq} Hz, 陷波: {notch_freq} Hz[/dim]")
        
        return eeg_filter(
            data, sfreq,
            low_freq=low_freq,
            high_freq=high_freq,
            notch_freq=notch_freq,
            filter_order=filter_order,
            notch_q=notch_q,
        )
    
    def _baseline_correct(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        """应用基线校正"""
        if self.config is not None:
            baseline_duration = self.config.preprocessing.baseline.duration
        else:
            baseline_duration = 0.2
        
        console.print(f"  [dim]基线时长: {baseline_duration} 秒[/dim]")
        
        return baseline_correction(
            data, sfreq=sfreq, baseline_duration=baseline_duration
        )
    
    def _remove_outliers(self, data: np.ndarray) -> np.ndarray:
        """去除异常值"""
        if self.config is not None:
            threshold = self.config.preprocessing.outlier.mad_threshold
        else:
            threshold = 8.0
        
        console.print(f"  [dim]MAD阈值: {threshold}[/dim]")
        
        return remove_outliers_mad(data, threshold=threshold)
    
    def _standardize(self, data: np.ndarray) -> np.ndarray:
        """标准化"""
        if self.config is not None:
            eps = self.config.preprocessing.bad_trial.min_std
        else:
            eps = 1e-6
        
        return trialwise_standardize(data, eps=eps)
    
    def _remove_bad_trials(
        self,
        data: np.ndarray,
        labels: np.ndarray = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], list]:
        """移除坏trial"""
        if self.config is not None:
            min_variance = self.config.preprocessing.bad_trial.min_variance
        else:
            min_variance = 1e-6
        
        data, labels, bad_indices = remove_bad_trials(
            data, labels, min_variance=min_variance
        )
        
        if len(bad_indices) > 0:
            print_warning(f"移除了 {len(bad_indices)} 个坏trial")
        
        return data, labels, bad_indices
    
    def filter(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        """单独执行滤波步骤"""
        return self._filter(data, sfreq)
    
    def baseline_correct(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        """单独执行基线校正步骤"""
        return self._baseline_correct(data, sfreq)
    
    def remove_outliers(self, data: np.ndarray) -> np.ndarray:
        """单独执行异常值去除步骤"""
        return self._remove_outliers(data)
    
    def standardize(self, data: np.ndarray) -> np.ndarray:
        """单独执行标准化步骤"""
        return self._standardize(data)
    
    @property
    def stats(self) -> Dict[str, Any]:
        """获取预处理统计信息"""
        return self._stats

