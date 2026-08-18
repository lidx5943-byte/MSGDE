# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""
TDA 拓扑特征提取
"""

import numpy as np
import math
from typing import Tuple, Dict, Any, List
import warnings
from scipy import sparse
from scipy.linalg import eigh
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn, TimeRemainingColumn
from ..utils.console import console

# 尝试导入 ripser
try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
    console.print("[yellow]警告: ripser 未安装，请运行 pip install ripser[/yellow]")

# 定义条形码数据类型
dt = np.dtype([('dim', int), ('birth', float), ('death', float)])


# 尝试导入 Gudhi（用于 PTHL 拉普拉斯方法）
try:
    import gudhi
    HAS_GUDHI = True
except ImportError:
    HAS_GUDHI = False
    console.print("[yellow]警告: Gudhi 未安装，PTHL 模式将不可用，请运行 pip install gudhi[/yellow]")

def _compute_rips_single_scale(args: Tuple) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    单个尺度的 Rips 复形计算（稀疏矩阵优化）
    """
    scale_idx, adjacency_matrix, max_dim, thresh = args
    
    # 简单的进程内日志
    try:
        nnz = adjacency_matrix.getnnz() if sparse.issparse(adjacency_matrix) else np.count_nonzero(adjacency_matrix)
        # print(f"DEBUG: Scale {scale_idx} start, edges={nnz}") 
    except:
        pass
        
    if not HAS_RIPSER:
        return scale_idx, np.zeros(3, dtype=np.float32), np.array([], dtype=dt)
    
    N = adjacency_matrix.shape[0]
    
    try:
        # 提取非零边 - 优化提取方式
        if sparse.issparse(adjacency_matrix):
            A_coo = adjacency_matrix.tocoo()
        else:
            A_coo = sparse.coo_matrix(adjacency_matrix)
        
        # 只保留上三角且大于0的边
        mask = (A_coo.row < A_coo.col) & (A_coo.data > 0)
        rows = A_coo.row[mask]
        cols = A_coo.col[mask]
        values = A_coo.data[mask]
        
        if len(values) == 0:
            return scale_idx, np.zeros(3, dtype=np.float32), np.array([], dtype=dt)

        MAX_EDGES = 150000 
        if len(values) > MAX_EDGES:

            top_k_indices = np.argpartition(values, -MAX_EDGES)[-MAX_EDGES:]
            
            rows = rows[top_k_indices]
            cols = cols[top_k_indices]
            values = values[top_k_indices]

        if len(values) > 0:
            min_val = np.min(values)
            max_val = np.max(values)
            if max_val > min_val:
                # 归一化到 [0, 1]
                normalized = (values - min_val) / (max_val - min_val)
                distances = 1.0 - normalized
                # 确保距离非负且合理
                distances = np.clip(distances, 0.0, 1.0)
            else:
                # 所有值相同，设为很小的距离（表示所有连接强度相同）
                distances = np.full(len(values), 0.001, dtype=np.float32)
        else:
            distances = np.array([], dtype=np.float32)
        
        # 只保留距离 <= thresh 的边
        valid_mask = distances <= thresh
        rows = rows[valid_mask]
        cols = cols[valid_mask]
        distances = distances[valid_mask]
        
        if len(distances) == 0:
            return scale_idx, np.zeros(3, dtype=np.float32), np.array([], dtype=dt)
        
        # 创建稀疏距离矩阵（上三角）
        sparse_dist = sparse.csr_matrix(
            (distances, (rows, cols)), 
            shape=(N, N),
            dtype=np.float32
        )
        sparse_dist = sparse_dist + sparse_dist.T 
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ripser(
                sparse_dist, 
                maxdim=max_dim, 
                distance_matrix=True,
                thresh=thresh
            )
        
        diagrams = result['dgms']
        
        # 提取条形码和特征
        bars_list = []
        # 初始化统计特征
        tda_stats = {
            'dim0_death': [],
            'dim1_birth': [],
            'dim1_death': []
        }
        
        # Dim 0
        if len(diagrams) > 0:
            dgm0 = diagrams[0]
            for b, d in dgm0:
                if d != np.inf and d - b > 0.00002:
                    bars_list.append((0, float(b), float(d)))
                    tda_stats['dim0_death'].append(float(d))
        
        # Dim 1
        if len(diagrams) > 1:
            dgm1 = diagrams[1]
            for b, d in dgm1:
                if d != np.inf and d - b > 0.00002:
                    bars_list.append((1, float(b), float(d)))
                    tda_stats['dim1_birth'].append(float(b))
                    tda_stats['dim1_death'].append(float(d))
        
        
        f1 = float(len(tda_stats['dim0_death'])) # 在提取循环里已经过滤了 inf
        f2 = float(len(tda_stats['dim1_birth'])) 
        f3 = float(len(tda_stats['dim1_death'])) # 同上
        
        features = np.array([f1, f2, f3], dtype=np.float32)
        
        return scale_idx, features, np.array(bars_list, dtype=dt) if bars_list else np.array([], dtype=dt)
        
    except Exception as e:
        console.print(f"[yellow]ripser 失败 (尺度 {scale_idx}): {e}[/yellow]")
        return scale_idx, np.zeros(3, dtype=np.float32), np.array([], dtype=dt)


def extract_topological_features_single_scale(
    scale_matrix: np.ndarray,
    config: Dict[str, Any] | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    提取单个尺度的拓扑特征 (3维) + 条形码

    返回
    ----
    features: np.ndarray, shape (3,), dtype float32
    bars: np.ndarray, dtype=dt
    """
    tda_config = (config.get('tda', {}).get('topological', {}) if config else {})
    max_dim = tda_config.get('max_dimension', 1)
    thresh = tda_config.get('max_edge_length', 1.0)
    method = tda_config.get('method', 'ripser').lower()

    # 用 CSR 以复用 worker 的稀疏优化路径
    if not sparse.issparse(scale_matrix):
        scale_matrix = sparse.csr_matrix(scale_matrix)

    if method == "pthl":
        _, feats, bars = _compute_pthl_single_scale((0, scale_matrix, max_dim, thresh))
    else:
        _, feats, bars = _compute_rips_single_scale((0, scale_matrix, max_dim, thresh))

    return feats.astype(np.float32), bars


def extract_topological_features(
    cutoff_matrices: np.ndarray,
    config: Dict[str, Any] = None,
    num_workers: int = 4,
    verbose: bool = True
) -> Tuple[np.ndarray, List]:
    """
    提取所有尺度的拓扑特征（ripser + 稀疏矩阵）
    """
    n_scales = cutoff_matrices.shape[0]
    n_samples = cutoff_matrices.shape[1]
    
    if verbose:
        console.print(f"[cyan]提取拓扑特征 (ripser+稀疏, {n_scales} 个尺度)...[/cyan]")
        if not HAS_RIPSER:
            console.print("[yellow]警告: ripser 未安装[/yellow]")
    
    # 获取配置参数
    tda_config = config.get('tda', {}).get('topological', {}) if config else {}
    max_dim = tda_config.get('max_dimension', 1)
    thresh = tda_config.get('max_edge_length', 1.0)
    method = tda_config.get('method', 'ripser').lower()
    
    # 提前转换稀疏格式，显著减少并行传参开销
    sparse_subgraphs = []
    for i in range(n_scales):
        if not sparse.issparse(cutoff_matrices[i]):
            sparse_subgraphs.append(sparse.csr_matrix(cutoff_matrices[i]))
        else:
            sparse_subgraphs.append(cutoff_matrices[i])
    
    args_list = [(i, sparse_subgraphs[i], max_dim, thresh) for i in range(n_scales)]
    
    del cutoff_matrices
    gc.collect()
    
    scale_features = None 
    bars_list = [None] * n_scales
    
    safe_workers = min(num_workers, 5)
    if verbose and num_workers > 5:
        console.print(f"[yellow]提示: 为防止内存交换，将并行进程数从 {num_workers} 限制为 {safe_workers}[/yellow]")
    
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
            task = progress.add_task("[green]TDA 特征", total=n_scales)
            
            if safe_workers > 1 and n_scales > 1:
                # 并行模式
                worker = _compute_pthl_single_scale if method == "pthl" else _compute_rips_single_scale
                with ProcessPoolExecutor(max_workers=min(safe_workers, n_scales)) as executor:
                    futures = {executor.submit(worker, args): args[0] for args in args_list}
                    
                    for future in as_completed(futures):
                        idx, features, bars = future.result()
                        if scale_features is None:
                            scale_features = np.zeros((n_scales, len(features)), dtype=np.float32)
                        scale_features[idx] = features
                        bars_list[idx] = bars
                        
                        nnz = sparse_subgraphs[idx].nnz
                        progress.update(task, advance=1, description=f"[green]TDA 尺度 {idx} 完成 (边数={nnz:,})")
            else:
                # 顺序模式
                worker = _compute_pthl_single_scale if method == "pthl" else _compute_rips_single_scale
                for i in range(n_scales):
                    idx, features, bars = worker(args_list[i])
                    if scale_features is None:
                        scale_features = np.zeros((n_scales, len(features)), dtype=np.float32)
                    scale_features[idx] = features
                    bars_list[idx] = bars
                    nnz = sparse_subgraphs[i].nnz
                    progress.update(task, advance=1, description=f"[green]TDA 尺度 {i} 完成 (边数={nnz:,})")
    else:
        # 非 verbose 模式
        if safe_workers > 1 and n_scales > 1:
            worker = _compute_pthl_single_scale if method == "pthl" else _compute_rips_single_scale
            with ProcessPoolExecutor(max_workers=min(safe_workers, n_scales)) as executor:
                futures = {executor.submit(worker, args): args[0] for args in args_list}
                for future in as_completed(futures):
                    idx, features, bars = future.result()
                    if scale_features is None:
                        scale_features = np.zeros((n_scales, len(features)), dtype=np.float32)
                    scale_features[idx] = features
                    bars_list[idx] = bars
        else:
            worker = _compute_pthl_single_scale if method == "pthl" else _compute_rips_single_scale
            for i in range(n_scales):
                idx, features, bars = worker(args_list[i])
                if scale_features is None:
                    scale_features = np.zeros((n_scales, len(features)), dtype=np.float32)
                scale_features[idx] = features
                bars_list[idx] = bars
    
    if scale_features is None:
        scale_features = np.zeros((n_scales, 3), dtype=np.float32)
    
    flat_features = scale_features.flatten()  # (n_scales * 3,)
    all_features = np.tile(flat_features, (n_samples, 1))
    
    if verbose:
        console.print(f"[green]✓ 拓扑特征提取完成: {all_features.shape}[/green]")
    
    return all_features, bars_list


def _compute_pthl_single_scale(args: Tuple) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    单个尺度的 PTHL 拉普拉斯方法：
    - 使用 Gudhi 构建单纯复形 (0,1,2-simplex)
    - 构造边界算子 B1, B2
    - 计算 Hodge 拉普拉斯 L0 = B1 B1^T, L1 = B2 B2^T + B1^T B1
    - 提取 3 维特征: [beta0, beta1, lambda1_min_nonzero]
    """
    scale_idx, adjacency_matrix, max_dim, thresh = args  
    if not HAS_GUDHI:
        return scale_idx, np.zeros(3, dtype=np.float32), np.array([], dtype=dt)

    # 稀疏矩阵转为稠密邻接矩阵
    if sparse.issparse(adjacency_matrix):
        adj = adjacency_matrix.toarray()
    else:
        adj = np.asarray(adjacency_matrix)

    num_nodes = adj.shape[0]

    # 构建 Gudhi SimplexTree
    st = gudhi.SimplexTree()
    for i in range(num_nodes):
        st.insert([i])

    rows, cols = np.where(adj > 0)
    for u, v in zip(rows, cols):
        if u < v:
            st.insert([u, v])

    # 扩展到 2-单形以便构造 B2
    st.expansion(2)

    # 从 simplex tree 提取 0,1,2 维单形
    simplices = [s for s, _ in st.get_simplices()]
    s0 = sorted([s for s in simplices if len(s) == 1])
    s1 = sorted([s for s in simplices if len(s) == 2])
    s2 = sorted([s for s in simplices if len(s) == 3])

    n0, n1, n2 = len(s0), len(s1), len(s2)
    id0 = {tuple(s): i for i, s in enumerate(s0)}
    id1 = {tuple(sorted(s)): i for i, s in enumerate(s1)}

    # 边界算子 B1: C1 -> C0
    if n1 > 0:
        rows_b1, cols_b1, data_b1 = [], [], []
        for j, edge in enumerate(s1):
            u, v = edge
            rows_b1.extend([id0[(u,)], id0[(v,)]])
            cols_b1.extend([j, j])
            data_b1.extend([-1, 1])
        B1 = sparse.csr_matrix((data_b1, (rows_b1, cols_b1)), shape=(n0, n1))
    else:
        B1 = sparse.csr_matrix((n0, 0))

    # 边界算子 B2: C2 -> C1
    if n2 > 0:
        rows_b2, cols_b2, data_b2 = [], [], []
        for k, tri in enumerate(s2):
            u, v, w = tri
            e_signs = [
                (tuple(sorted((v, w))), 1),   # (v,w)
                (tuple(sorted((u, w))), -1),  # (u,w)
                (tuple(sorted((u, v))), 1)    # (u,v)
            ]
            for e, sign in e_signs:
                if e in id1:
                    rows_b2.append(id1[e])
                    cols_b2.append(k)
                    data_b2.append(sign)
        B2 = sparse.csr_matrix((data_b2, (rows_b2, cols_b2)), shape=(n1, n2))
    else:
        B2 = sparse.csr_matrix((n1, 0))

    # Hodge 拉普拉斯
    L0 = B1 @ B1.T
    if n1 > 0:
        L1 = (B2 @ B2.T) + (B1.T @ B1)
    else:
        L1 = sparse.csr_matrix((0, 0))

    def _analyze_spectrum(mat: sparse.csr_matrix) -> Tuple[int, float]:
        if mat.shape[0] == 0:
            return 0, 0.0
        vals = eigh(mat.toarray(), eigvals_only=True)
        tol = 1e-5
        zeros = vals[np.abs(vals) < tol]
        betti = len(zeros)
        non_zeros = vals[np.abs(vals) >= tol]
        min_lambda = float(non_zeros.min()) if len(non_zeros) > 0 else 0.0
        return int(betti), min_lambda

    beta0, lambda0 = _analyze_spectrum(L0)
    beta1, lambda1 = _analyze_spectrum(L1)


    features = np.array(
        [float(beta0), float(beta1), float(lambda1)],
        dtype=np.float32
    )
    bars = np.array([], dtype=dt)
    return scale_idx, features, bars





if __name__ == "__main__":
    console.print("[bold]测试 TDA 特征提取[/bold]")
    
    np.random.seed(42)
    test_matrices = np.random.rand(3, 100, 100).astype(np.float32)
    for i in range(3):
        test_matrices[i] = (test_matrices[i] + test_matrices[i].T) / 2
        np.fill_diagonal(test_matrices[i], 0)
    
    features, bars = extract_topological_features(test_matrices)
    console.print(f"特征形状: {features.shape}")
