# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""几何与拓扑（TDA）、Lorenz 动力学及微扰相关特征。"""
from .spectral import extract_spectral_features as calculate_spectral_features
from .tda import extract_topological_features_single_scale as calculate_tda_features
from .tda import extract_topological_features as calculate_tda_features_batch
from .lorenz import (
    LorenzOscillator, 
    LorenzConfig, 
    RosslerOscillator,
    RosslerConfig,
    simulate_lorenz_sparse,
    simulate_rossler_sparse,
    extract_lorenz_features_single_scale as calculate_dynamics_features,
    extract_lorenz_features as calculate_dynamics_features_batch
)
from .metrics import calculate_spectral_metrics, calculate_dynamics_metrics
from .perturbation import run_perturbation_analysis_on_scale

__all__ = [
    "calculate_spectral_features",
    "calculate_tda_features",
    "calculate_tda_features_batch",
    "LorenzOscillator",
    "LorenzConfig",
    "simulate_lorenz_sparse",
    "calculate_dynamics_features",
    "calculate_dynamics_features_batch",
    "calculate_dynamics_metrics",
    "calculate_spectral_metrics",
    "run_perturbation_analysis_on_scale",
]
