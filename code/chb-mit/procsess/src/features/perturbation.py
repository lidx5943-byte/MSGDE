# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""Step 4 节点移除微扰。"""
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from ..utils.console import console
from .metrics import calculate_spectral_metrics, calculate_dynamics_metrics
from .lorenz import simulate_lorenz_sparse

def _perturb_batch_worker(args):
    """批量样本微扰任务"""
    node_indices, scale_idx, mat_w, mat_b, config_payload = args
    results = []
    
    spec_list = config_payload['spec_list']
    
    for node_idx in node_indices:
        # 移除节点
        mask = np.ones(mat_w.shape[0], dtype=bool)
        mask[node_idx] = False
        
        sub_mat_w = mat_w[mask][:, mask]
        sub_mat_b = mat_b[mask][:, mask]
        
        spec_metrics = calculate_spectral_metrics(sub_mat_w, spec_list)

        dim_dyn = config_payload.get('dim_dyn', 0)
        if dim_dyn > 0:
            solver_func = config_payload.get('solver_func', simulate_lorenz_sparse)
            dyn_metrics = calculate_dynamics_metrics(sub_mat_b, solver_func, config_payload)
        else:
            dyn_metrics = np.array([], dtype=np.float32)
        
        results.append((node_idx, spec_metrics, dyn_metrics))
        
    return results

def run_perturbation_analysis_on_scale(
    scale_idx: int,
    mat_w: np.ndarray,
    mat_b: np.ndarray,
    config_payload: Dict[str, Any],
    n_workers: int = 10,
    batch_size: int = 50,
    dim_spec: int = 0,
    dim_dyn: int = 0,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """单尺度上逐节点移除；返回几何指标与动力学指标矩阵。"""
    N = mat_w.shape[0]
    perturbed_spec = np.zeros((N, dim_spec), dtype=np.float32)
    perturbed_dyn = np.zeros((N, dim_dyn), dtype=np.float32)
    n_workers = max(1, int(n_workers))
    batch_size = max(1, int(batch_size))

    num_batches = int(np.ceil(N / batch_size))
    config_payload['dim_dyn'] = dim_dyn

    def _run(progress: Progress | None = None, task: int | None = None) -> None:
        indices = np.arange(N)

        if n_workers == 1 or num_batches == 1:
            for i in range(num_batches):
                batch_indices = indices[i * batch_size : (i + 1) * batch_size]
                batch_results = _perturb_batch_worker((batch_indices, scale_idx, mat_w, mat_b, config_payload))
                for node_idx, spec, dyn in batch_results:
                    perturbed_spec[node_idx] = spec
                    perturbed_dyn[node_idx] = dyn
                    if progress is not None and task is not None:
                        progress.advance(task)
        else:
            with ProcessPoolExecutor(max_workers=min(n_workers, num_batches)) as executor:
                futures = []

                for i in range(num_batches):
                    batch_indices = indices[i * batch_size : (i + 1) * batch_size]
                    futures.append(executor.submit(
                        _perturb_batch_worker,
                        (batch_indices, scale_idx, mat_w, mat_b, config_payload)
                    ))

                for future in as_completed(futures):
                    batch_results = future.result()
                    for node_idx, spec, dyn in batch_results:
                        perturbed_spec[node_idx] = spec
                        perturbed_dyn[node_idx] = dyn
                        if progress is not None and task is not None:
                            progress.advance(task)

    if show_progress:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Perturbation"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"尺度 {scale_idx} 微扰分析", total=N)
            _run(progress, task)
    else:
        _run()

    return perturbed_spec, perturbed_dyn
