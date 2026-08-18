# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""
矩阵生成模块
"""

from .similarity import compute_similarity_matrix, compute_similarity_matrix_2d
from .graph_builder import (
    filter_negative_correlations,
    apply_gaussian_kernel,
    partition_by_uniform,
    partition_by_quantile,
    binarize_cutoff,
    compute_laplacian
)
from .report_generator import generate_matrix_report

__all__ = [
    "compute_similarity_matrix",
    "compute_similarity_matrix_2d",
    "filter_negative_correlations",
    "apply_gaussian_kernel",
    "partition_by_uniform",
    "partition_by_quantile",
    "binarize_cutoff",
    "compute_laplacian",
    "generate_matrix_report",
]
