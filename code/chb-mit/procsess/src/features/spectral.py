# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""多尺度拉普拉斯迹统计（几何特征）；稀疏实现与并行。"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sp_linalg
from typing import Tuple
import warnings
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn, TimeRemainingColumn

from ..utils.console import console, print_info as logger


def compute_laplacian_eigenvalues_sparse(adj_matrix: np.ndarray, k: int = 20) -> np.ndarray:
    """稀疏拉普拉斯前 k 个最小特征值（SM）。"""
    N = adj_matrix.shape[0]

    A = sparse.csr_matrix(adj_matrix, dtype=np.float64)

    A = (A + A.T) / 2

    degrees = np.array(np.abs(A).sum(axis=1)).flatten()

    D = sparse.diags(degrees, format='csr')
    L = D - A

    nnz = L.nnz

    if nnz == 0 or nnz < 10:
        return np.zeros(min(k, N))

    k_actual = min(k, N - 2)
    if k_actual < 2:
        k_actual = 2

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eigenvalues = sp_linalg.eigsh(
                L, k=k_actual, which='SM',
                return_eigenvectors=False,
                tol=1e-3, maxiter=100
            )
            return np.sort(np.real(eigenvalues))
    except Exception:
        return np.array([0.0, np.min(degrees[degrees > 0]) if np.any(degrees > 0) else 0.0,
                        np.max(degrees), np.mean(degrees), np.std(degrees)])


def extract_spectral_features_single(adj_matrix: np.ndarray) -> np.ndarray:
    """由 L 的迹与 Frobenius 项得 [sum, mean, std]，不显式求特征值。"""
    N = adj_matrix.shape[0]

    degrees = np.abs(adj_matrix).sum(axis=1)
    trace_L = np.sum(degrees)

    mean_lambda = trace_L / N

    sum_sq_L = np.sum(degrees**2) + np.sum(adj_matrix**2)
    mean_sq = sum_sq_L / N

    var_lambda = mean_sq - (mean_lambda**2)
    if var_lambda < 0: var_lambda = 0.0
    std_lambda = np.sqrt(var_lambda)

    return np.array([trace_L, mean_lambda, std_lambda], dtype=np.float32)


def extract_spectral_features_single_scale_matrix(
    scale_matrix: np.ndarray,
    k_eigenvalues: int = 20
) -> np.ndarray:
    """单尺度 3 维几何特征 [sum, mean, std]。"""
    return extract_spectral_features_single(scale_matrix)


def _process_scale_spectral(args: Tuple) -> Tuple[int, np.ndarray]:
    scale_idx, scale_matrix = args
    features = extract_spectral_features_single(scale_matrix)
    return scale_idx, features


def extract_spectral_features(
    subgraphs: np.ndarray,
    num_workers: int = 4,
    verbose: bool = True,
    k_eigenvalues: int = 20
) -> np.ndarray:
    """并行提取各尺度几何特征并沿列拼接。"""
    num_scales, num_samples, _ = subgraphs.shape

    all_features = np.zeros((num_samples, num_scales * 3), dtype=np.float32)

    if verbose:
        console.print(f"[cyan]提取几何特征: {num_scales}个尺度, {num_samples}个样本[/cyan]")

    start_time = time.time()

    args_list = [(i, subgraphs[i]) for i in range(num_scales)]

    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]

    if verbose:
        with Progress(*progress_columns, console=console) as progress:
            task = progress.add_task("[yellow]几何特征", total=num_scales)

            if num_workers > 1 and num_scales > 1:
                with ProcessPoolExecutor(max_workers=min(num_workers, num_scales)) as executor:
                    futures = {executor.submit(_process_scale_spectral, args): args[0] for args in args_list}
                    for future in as_completed(futures):
                        idx, features = future.result()
                        start_idx = idx * 3
                        all_features[:, start_idx:start_idx+3] = features
                        progress.update(task, advance=1, description=f"[yellow]几何尺度 {idx} 完成")
            else:
                for scale_idx in range(num_scales):
                    idx, features = _process_scale_spectral(args_list[scale_idx])
                    start_idx = idx * 3
                    all_features[:, start_idx:start_idx+3] = features
                    progress.update(task, advance=1, description=f"[yellow]几何尺度 {idx} 完成")
    else:
        if num_workers > 1 and num_scales > 1:
            with ProcessPoolExecutor(max_workers=min(num_workers, num_scales)) as executor:
                futures = {executor.submit(_process_scale_spectral, args): args[0] for args in args_list}
                for future in as_completed(futures):
                    idx, features = future.result()
                    start_idx = idx * 3
                    all_features[:, start_idx:start_idx+3] = features
        else:
            for scale_idx in range(num_scales):
                idx, features = _process_scale_spectral(args_list[scale_idx])
                start_idx = idx * 3
                all_features[:, start_idx:start_idx+3] = features

    total_elapsed = time.time() - start_time
    logger(f"✓ 几何特征提取完成: {all_features.shape}, 耗时: {total_elapsed:.1f}s")
    return all_features


if __name__ == "__main__":
    console.print("[yellow]运行几何特征提取测试...[/yellow]")

    np.random.seed(42)
    test_data = np.zeros((3, 100, 100), dtype=np.float32)
    for i in range(3):
        n_edges = 100
        rows = np.random.randint(0, 100, n_edges)
        cols = np.random.randint(0, 100, n_edges)
        test_data[i, rows, cols] = np.random.rand(n_edges)
        test_data[i] = (test_data[i] + test_data[i].T) / 2

    features = extract_spectral_features(test_data, verbose=True)
    console.print(f"[green]✓ 完成! 形状: {features.shape}[/green]")
