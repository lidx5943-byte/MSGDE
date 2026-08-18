"""
可视化模块
==========

提供轨迹和分析结果的可视化功能：
- trajectories: 轨迹和相空间可视化
- diversity: 特征差异性分析可视化
"""

from .trajectories import (
    plot_phase_space,
    plot_trajectory_evolution,
    plot_butterfly_attractor,
)
from .diversity import (
    plot_feature_diversity,
    analyze_feature_diversity,
)

__all__ = [
    # trajectories
    "plot_phase_space",
    "plot_trajectory_evolution",
    "plot_butterfly_attractor",
    # diversity
    "plot_feature_diversity",
    "analyze_feature_diversity",
]

