# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""单矩阵上的几何/微扰指标与 Lorenz 动力学指标。"""

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from typing import Dict, Any, List, Optional, Callable

try:
    from ripser import ripser as _ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False

def calculate_spectral_metrics(matrix: np.ndarray, metrics_list: List[str]) -> np.ndarray:
    """基于图拉普拉斯 L 的几何统计量；`metrics_list` 见实现内分支。"""
    N = matrix.shape[0]
    
    degrees = np.sum(matrix, axis=1)
    L = np.diag(degrees) - matrix
    
    results = []
    
    trace_L = np.sum(degrees)
    
    for m in metrics_list:
        if m == "sum":
            results.append(trace_L)

        elif m == "mean":
            results.append(trace_L / N)

        elif m == "std" or m == "var":
            sum_sq_L = np.sum(degrees**2) + np.sum(matrix**2)
            mean_sq = sum_sq_L / N
            mean = trace_L / N
            
            var = mean_sq - mean**2
            if var < 0: var = 0.0
            
            if m == "var":
                results.append(var)
            else:
                results.append(np.sqrt(var))
                
        elif m == "max":
            try:
                vals = eigsh(L, k=1, which='LA', return_eigenvectors=False, ncv=10)
                results.append(vals[0])
            except Exception:
                results.append(np.max(np.sum(np.abs(L), axis=1)))

        elif m == "min_nonzero":
            try:
                vals = eigsh(L, k=2, which='SM', return_eigenvectors=False, ncv=15)
                vals_sorted = np.sort(vals)
                nonzeros = vals_sorted[vals_sorted > 1e-8]
                if len(nonzeros) > 0:
                    results.append(nonzeros[0])
                else:
                    results.append(0.0)
            except Exception:
                results.append(0.0)

        elif m == "tda":
            if not HAS_RIPSER:
                results.extend([0.0, 0.0, 0.0])
                continue

            try:
                N_mat = matrix.shape[0]
                PERSISTENCE_THRESHOLD = 0.00002
                MAX_EDGES = 150000

                # 提取上三角非零边
                A_coo = sparse.coo_matrix(matrix)
                edge_mask = (A_coo.row < A_coo.col) & (A_coo.data > 0)
                rows_e = A_coo.row[edge_mask]
                cols_e = A_coo.col[edge_mask]
                vals_e = A_coo.data[edge_mask]

                if len(vals_e) == 0:
                    results.extend([0.0, 0.0, 0.0])
                    continue

                if len(vals_e) > MAX_EDGES:
                    top_k = np.argpartition(vals_e, -MAX_EDGES)[-MAX_EDGES:]
                    rows_e = rows_e[top_k]
                    cols_e = cols_e[top_k]
                    vals_e = vals_e[top_k]

                min_v, max_v = vals_e.min(), vals_e.max()
                if max_v > min_v:
                    normed = (vals_e - min_v) / (max_v - min_v)
                    distances = np.clip(1.0 - normed, 0.0, 1.0).astype(np.float32)
                else:
                    distances = np.full(len(vals_e), 0.001, dtype=np.float32)

                sp_dist = sparse.csr_matrix(
                    (distances, (rows_e, cols_e)),
                    shape=(N_mat, N_mat),
                    dtype=np.float32
                )
                sp_dist = sp_dist + sp_dist.T

                import warnings as _w
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    rips_result = _ripser(
                        sp_dist,
                        maxdim=1,
                        distance_matrix=True,
                        thresh=1.0
                    )

                diagrams = rips_result['dgms']
                h0_death = 0
                h1_birth = 0
                h1_death = 0

                if len(diagrams) > 0:
                    for b, d in diagrams[0]:
                        if d != np.inf and (d - b) > PERSISTENCE_THRESHOLD:
                            h0_death += 1

                if len(diagrams) > 1:
                    for b, d in diagrams[1]:
                        if d != np.inf and (d - b) > PERSISTENCE_THRESHOLD:
                            h1_birth += 1
                            h1_death += 1

                results.extend([float(h0_death), float(h1_birth), float(h1_death)])

            except Exception:
                results.extend([0.0, 0.0, 0.0])


    return np.array(results, dtype=np.float32)

def calculate_dynamics_metrics(
    matrix: np.ndarray, 
    lorenz_solver_func: Callable, 
    config: Dict[str, Any]
) -> np.ndarray:
    """Lorenz 轨迹 x 分量的全局统计量 [mean, max, min, median, var, std]。"""
    result = lorenz_solver_func(matrix, config['config'], config['oscillator'], config['random_seed'])
    
    x_stats = result['stats']['x']
    return np.array([
        x_stats['mean'], x_stats['max'], x_stats['min'], 
        x_stats['median'], x_stats['var'], x_stats['std']
    ], dtype=np.float32)
