import numpy as np
from typing import Optional, Tuple, Dict, Any
from .filters import eeg_filter
from .baseline import baseline_correction
from .outliers import remove_outliers_mad, remove_bad_trials
from .standardize import trialwise_standardize
from ..utils.logger import get_logger, console, print_step, print_success, print_table
from ..utils.timer import TimerGroup

class ChbmitPreprocessingPipeline:
    def __init__(self, config=None):
        self.config = config
        self.logger = get_logger("preprocessing_chbmit")
        self._stats = {}
    
    def run(self, data: np.ndarray, labels: np.ndarray = None, sfreq: float = 256.0) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        console.print("[bold]CHB-MIT 预处理流水线[/bold]")
        timers = TimerGroup("预处理")
        timers.start_total()
        original_shape = data.shape
        console.print(f"[dim]输入数据形状: {original_shape}[/dim]")
        total_steps = 5
        # 1. 滤波
        print_step(1, total_steps, "信号滤波（带通+陷波）")
        with timers.timer("信号滤波"):
            data = self._filter(data, sfreq)
        # 2. 基线校正
        print_step(2, total_steps, "基线校正（使用前0.2秒）")
        with timers.timer("基线校正"):
            data = self._baseline_correct(data, sfreq)
        # 3. 异常值去除
        print_step(3, total_steps, "异常值去除")
        with timers.timer("异常值去除"):
            data = self._remove_outliers(data)
        # 4. 标准化
        print_step(4, total_steps, "数据标准化")
        with timers.timer("标准化"):
            data = self._standardize(data)
        # 5. 坏段移除
        print_step(5, total_steps, "坏段移除")
        with timers.timer("坏段移除"):
            data, labels, bad_indices = self._remove_bad_segments(data, labels)
        timers.stop_total()
        self._stats = {
            "original_shape": original_shape,
            "final_shape": data.shape,
            "n_removed_segments": len(bad_indices),
            "bad_indices": bad_indices,
        }
        print_success("预处理完成")
        stats_rows = [
            ["原始形状", str(original_shape)],
            ["最终形状", str(data.shape)],
            ["移除段数", str(len(bad_indices))],
        ]
        print_table("预处理统计", ["项目", "值"], stats_rows)
        timers.report()
        return data, labels, self._stats
    
    def _filter(self, data, sfreq):
        # Try to get config, fallback to defaults
        if self.config is not None:
            # Access via _data or get method
            cfg = self.config._data if hasattr(self.config, '_data') else {}
            preproc = cfg.get('preprocessing', {})
            filter_cfg = preproc.get('filter', {})
            notch_cfg = preproc.get('notch', {})
            low = filter_cfg.get('low_freq', 8.0)
            high = filter_cfg.get('high_freq', 12.5)
            order = filter_cfg.get('order', 4)
            notch = notch_cfg.get('freq', 50.0)
            q = notch_cfg.get('quality_factor', 30)
        else:
            low, high, order, notch, q = 8.0, 12.5, 4, 50.0, 30
        console.print(f"  [dim]带通: {low}-{high} Hz, 陷波: {notch} Hz[/dim]")
        return eeg_filter(data, sfreq, low, high, notch, order, q)
    
    def _baseline_correct(self, data, sfreq):
        if self.config is not None:
            cfg = self.config._data if hasattr(self.config, '_data') else {}
            preproc = cfg.get('preprocessing', {})
            baseline_cfg = preproc.get('baseline', {})
            duration = baseline_cfg.get('duration', 0.2)
        else:
            duration = 0.2
        console.print(f"  [dim]基线时长: {duration} 秒[/dim]")
        return baseline_correction(data, sfreq=sfreq, baseline_duration=duration)
    
    def _remove_outliers(self, data):
        if self.config is not None:
            cfg = self.config._data if hasattr(self.config, '_data') else {}
            preproc = cfg.get('preprocessing', {})
            outlier_cfg = preproc.get('outlier', {})
            threshold = outlier_cfg.get('mad_threshold', 8.0)
        else:
            threshold = 8.0
        console.print(f"  [dim]MAD阈值: {threshold}[/dim]")
        return remove_outliers_mad(data, threshold=threshold)
    
    def _standardize(self, data):
        if self.config is not None:
            cfg = self.config._data if hasattr(self.config, '_data') else {}
            preproc = cfg.get('preprocessing', {})
            bad_cfg = preproc.get('bad_trial', {})
            eps = bad_cfg.get('min_std', 1e-6)
        else:
            eps = 1e-6
        return trialwise_standardize(data, eps=eps)
    
    def _remove_bad_segments(self, data, labels):
        if self.config is not None:
            cfg = self.config._data if hasattr(self.config, '_data') else {}
            preproc = cfg.get('preprocessing', {})
            bad_cfg = preproc.get('bad_trial', {})
            min_var = bad_cfg.get('min_variance', 1e-6)
        else:
            min_var = 1e-6
        data, labels, bad = remove_bad_trials(data, labels, min_variance=min_var)
        if len(bad) > 0:
            console.print(f"[yellow]移除了 {len(bad)} 个坏段[/yellow]")
        return data, labels, bad
