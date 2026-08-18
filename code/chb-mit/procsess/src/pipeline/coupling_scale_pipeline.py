# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""耦合强度×尺度消融：多尺度矩阵、Lorenz 与 12 维微扰特征、分类与热图。"""

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from ..utils.console import (
    console, print_header, print_step, print_success, print_error, print_info, print_warning
)
from ..features.lorenz import (
    LorenzConfig, LorenzOscillator, simulate_lorenz_sparse
)
from ..features.metrics import calculate_spectral_metrics, calculate_dynamics_metrics
from ..classification.models import get_models
from ..classification.evaluation import evaluate_model_cv
from ..visualization.coupling_scale_plotter import plot_coupling_scale_heatmap


@dataclass
class ExperimentResult:
    """单次实验结果。"""
    coupling_strength: float
    scale_idx: int
    accuracy: float
    auc: float = 0.0
    error: str = ""


def extract_lorenz_features_for_scale(
    scale_matrix: np.ndarray,
    coupling_strength: float,
    config: Dict[str, Any]
) -> np.ndarray:
    """单尺度邻接矩阵上 Lorenz 仿真，返回每节点特征 (N, 18)。"""
    lorenz_cfg = config.get("lorenz", {})

    lz_config = LorenzConfig(
        coupling_strength=coupling_strength,
        coupling_mode=lorenz_cfg.get("coupling_mode", "xyz_all"),
        dt=lorenz_cfg.get("dt", 0.01),
        total_steps=lorenz_cfg.get("total_steps", 100),
        steady_steps=lorenz_cfg.get("steady_steps", 50),
        initial_range=lorenz_cfg.get("initial_range", 1.0),
        sparsity_threshold=lorenz_cfg.get("sparsity_threshold", 1e-6)
    )

    oscillator = LorenzOscillator()

    result = simulate_lorenz_sparse(
        adj_matrix=scale_matrix,
        config=lz_config,
        oscillator=oscillator,
        random_seed=config.get("experiment", {}).get("random_seed", 42)
    )

    node_features = result.get('per_node_features', np.zeros((scale_matrix.shape[0], 18)))

    return node_features


def run_single_experiment(
    coupling_strength: float,
    scale_idx: int,
    scale_matrix: np.ndarray,
    y: np.ndarray,
    config: Dict[str, Any]
) -> ExperimentResult:
    """单组（耦合强度×尺度）分类评估。"""
    try:
        features = extract_lorenz_features_for_scale(scale_matrix, coupling_strength, config)

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        clf_config = config.get("classification", {})
        classifier_name = clf_config.get("classifier", "RF")

        models = get_models(random_seed=config.get("experiment", {}).get("random_seed", 42))
        model = models.get(classifier_name)

        if model is None:
            return ExperimentResult(coupling_strength, scale_idx, 0.0, 0.0, f"未找到分类器: {classifier_name}")

        from sklearn.base import clone
        clf = clone(model)

        metrics, _ = evaluate_model_cv(
            clf, features, y,
            cv_splits=clf_config.get("n_folds", 5),
            model_name=classifier_name,
            feature_set_name=f"C{coupling_strength:.1f}_S{scale_idx}"
        )

        return ExperimentResult(
            coupling_strength=coupling_strength,
            scale_idx=scale_idx,
            accuracy=metrics['Accuracy'],
            auc=metrics.get('AUC', 0.0)
        )

    except Exception as e:
        return ExperimentResult(coupling_strength, scale_idx, 0.0, 0.0, str(e))


def _static_work(args):
    """单节点删点后的几何指标（供多进程）。"""
    node_idx, matrix, metrics_list, n_nodes = args
    mask = np.ones(n_nodes, dtype=bool)
    mask[node_idx] = False
    sub_mat = matrix[mask][:, mask]
    return calculate_spectral_metrics(sub_mat, metrics_list)

def precompute_static_perturbation(
    cutoff_weight: np.ndarray,
    config: Dict[str, Any]
) -> List[np.ndarray]:
    """各尺度静态微扰：几何+TDA，每尺度 6 维差分特征。"""
    n_scales = cutoff_weight.shape[0]
    n_nodes = cutoff_weight.shape[1]
    static_feats_list = []
    metrics_list = ["sum", "mean", "std", "tda"]

    print_info(f"预计算 {n_scales} 个尺度的静态微扰（几何+TDA）…")

    for s_idx in range(n_scales):
        mat_w = cutoff_weight[s_idx]
        base_metrics = calculate_spectral_metrics(mat_w, metrics_list)

        tasks = [(i, mat_w, metrics_list, n_nodes) for i in range(n_nodes)]
        with ProcessPoolExecutor(max_workers=config.get("parallel", {}).get("n_workers", 8)) as executor:
            perturbed_metrics = np.array(list(executor.map(_static_work, tasks)))

        static_diff = np.abs(base_metrics - perturbed_metrics)
        static_feats_list.append(static_diff)
        print_info(f"尺度 {s_idx+1} 静态微扰完成")

    return static_feats_list


def _total_experiment_work(args):
    """单组（耦合×尺度）：LOO 微扰，拼接几何+TDA+动力学后交叉验证。"""
    (c_val, s_idx, mat_b, static_diff, y, lorenz_cfg, classifier_name, n_folds, random_seed) = args

    from ..features.lorenz import LorenzConfig, LorenzOscillator, simulate_lorenz_sparse
    from sklearn.base import clone

    n_nodes = mat_b.shape[0]
    oscillator = LorenzOscillator()
    lorenz_params = {
        'config': LorenzConfig(coupling_strength=c_val, total_steps=lorenz_cfg.get("total_steps", 100), steady_steps=lorenz_cfg.get("steady_steps", 50)),
        'oscillator': oscillator,
        'random_seed': random_seed
    }

    base_dyn = calculate_dynamics_metrics(mat_b, simulate_lorenz_sparse, lorenz_params)

    dyn_diffs = []
    for i in range(n_nodes):
        mask = np.ones(n_nodes, dtype=bool)
        mask[i] = False
        sub_mat = mat_b[mask][:, mask]
        perturbed_dyn = calculate_dynamics_metrics(sub_mat, simulate_lorenz_sparse, lorenz_params)
        dyn_diffs.append(np.abs(base_dyn - perturbed_dyn))

    dyn_perturbation = np.array(dyn_diffs)

    X_combined = np.hstack([static_diff, dyn_perturbation])

    models = get_models(random_seed=random_seed)
    model = clone(models.get(classifier_name))
    metrics, _ = evaluate_model_cv(model, X_combined, y, cv_splits=n_folds)

    return {
        'CouplingStrength': float(c_val),
        'Scale': int(s_idx + 1),
        'Accuracy': metrics['Accuracy'],
        'AUC': metrics.get('AUC', 0.0)
    }

def run_coupling_scale_ablation_pipeline(config: Dict[str, Any]):
    """耦合×尺度 12 维微扰消融主流程。"""
    print_header("耦合强度×尺度消融（12 维微扰）")

    start_time = time.time()
    output_dir = Path(config.get("output", {}).get("save_dir", "./output/coupling_scale_ablation"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print_step(1, 4, "加载数据与预计算静态特征")
    data_cfg = config.get("data", {})
    y = np.load(data_cfg.get("labels_path"))
    cutoff_weight = np.load(data_cfg.get("cutoff_weight_path"))
    cutoff_0_1 = np.load(data_cfg.get("cutoff_0_1_path"))
    n_scales = cutoff_weight.shape[0]

    static_perturbations = precompute_static_perturbation(cutoff_weight, config)

    print_step(2, 4, "检查进度并分配任务")

    results_cache_path = output_dir / "results_perturbation_12d.csv"
    all_results = []
    done_keys = set()

    if results_cache_path.exists():
        try:
            df_cache = pd.read_csv(results_cache_path)
            for _, row in df_cache.iterrows():
                all_results.append(row.to_dict())
                done_keys.add((round(float(row['CouplingStrength']), 2), int(row['Scale'])))
            print_info(f"已从缓存恢复 {len(done_keys)} 条结果")
        except Exception as e:
            print_warning(f"读取缓存失败: {e}")

    c_cfg = config.get("coupling", {})
    coupling_values = np.round(np.arange(c_cfg.get("start", 0.0), c_cfg.get("end", 1.0) + 0.0001, c_cfg.get("step", 0.01)), 2)

    lorenz_cfg = config.get("lorenz", {})
    classifier_name = config.get("classification", {}).get("classifier", "RF")
    n_folds = config.get("classification", {}).get("n_folds", 5)
    random_seed = config.get("experiment", {}).get("random_seed", 42)
    n_workers = config.get("parallel", {}).get("n_workers", 8)

    all_tasks = []
    for c_val in coupling_values:
        for s_idx in range(n_scales):
            key = (round(float(c_val), 2), int(s_idx + 1))
            if key not in done_keys:
                all_tasks.append((
                    c_val, s_idx, cutoff_0_1[s_idx],
                    static_perturbations[s_idx], y,
                    lorenz_cfg, classifier_name, n_folds, random_seed
                ))

    print_info(f"待执行任务数: {len(all_tasks)}")

    if len(all_tasks) > 0:
        print_step(3, 4, "并行实验")

        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task_id = progress.add_task("[cyan]进度", total=len(all_tasks))

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(_total_experiment_work, task) for task in all_tasks]

                for future in as_completed(futures):
                    res = future.result()
                    all_results.append(res)
                    progress.advance(task_id)

                    if len(all_results) % 50 == 0:
                        pd.DataFrame(all_results).to_csv(results_cache_path, index=False)
    else:
        print_success("全部结果已在缓存中，跳过计算。")

    print_step(4, 4, "汇总与可视化")
    df_final = pd.DataFrame(all_results).sort_values(['Scale', 'CouplingStrength'])
    df_final.to_csv(output_dir / "coupling_scale_perturbation_final.csv", index=False)

    plot_coupling_scale_heatmap(df_final, output_dir, config)

    total_time = time.time() - start_time
    print_header(f"完成，耗时 {total_time:.1f} s")

