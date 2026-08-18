"""
拉普拉斯特征模块
================

提供基于图拉普拉斯矩阵的多尺度特征提取：
- multiscale: 多尺度拉普拉斯矩阵构建
"""

from .multiscale import (
    compute_adjacency_matrix,
    compute_laplacian_matrix,
    compute_multiscale_laplacians,
    LaplacianPipeline,
)

__all__ = [
    "compute_adjacency_matrix",
    "compute_laplacian_matrix",
    "compute_multiscale_laplacians",
    "LaplacianPipeline",
]

