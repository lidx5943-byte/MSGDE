"""
相似度计算模块
==============

提供样本间相似度矩阵的计算功能：
- correlation: 多种相关性度量（Pearson, Spearman, Cosine等）
"""

from .correlation import (
    compute_similarity_matrix,
    pearson_correlation,
    spearman_correlation,
    cosine_similarity,
)

__all__ = [
    "compute_similarity_matrix",
    "pearson_correlation",
    "spearman_correlation",
    "cosine_similarity",
]

