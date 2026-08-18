"""
动力学分析模块
==============

提供基于混沌振荡器的动力学分析：
- oscillators: Lorenz等混沌振荡器定义
- solvers: 数值求解器（Euler, RK4）
- features: 轨迹特征提取
- pipeline: 完整动力学分析流水线
"""

from .oscillators import LorenzOscillator
from .solvers import EulerSolver, RK4Solver
from .features import extract_trajectory_features, compute_feature_diversity
from .pipeline import DynamicsAnalysisPipeline

__all__ = [
    # oscillators
    "LorenzOscillator",
    # solvers
    "EulerSolver",
    "RK4Solver",
    # features
    "extract_trajectory_features",
    "compute_feature_diversity",
    # pipeline
    "DynamicsAnalysisPipeline",
]

