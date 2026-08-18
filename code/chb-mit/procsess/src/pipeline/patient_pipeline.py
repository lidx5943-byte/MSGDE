# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""单患者：构图、微扰特征与分类评估。"""

import os
import time
import numpy as np
import pandas as pd
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..utils.console import (
    console, print_header, print_step, print_success, print_error, print_warning, print_info
)
from ..matrix.similarity import compute_similarity_matrix_2d
from ..matrix.graph_builder import (
    filter_negative_correlations, apply_gaussian_kernel,
    partition_by_quantile, partition_by_uniform, binarize_cutoff
)
from ..features.lorenz import (
    simulate_lorenz_sparse, LorenzConfig, LorenzOscillator,
    simulate_rossler_sparse, RosslerConfig, RosslerOscillator
)
from ..features.perturbation import run_perturbation_analysis_on_scale
from ..classification.evaluation import evaluate_model_cv
from ..classification.models import get_models, build_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib

@dataclass
class PatientResult:
    patient_id: str
    n_samples: int = 0
    n_class0: int = 0
    n_class1: int = 0
    feature_dim: int = 0
    classifier_results: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class PatientFeatureBundle:
    features: np.ndarray
    labels: np.ndarray
    feature_dims: Dict[str, List[int]]
    group_masks: Dict[str, np.ndarray]
    cutoff_weight: Optional[np.ndarray] = None
    cutoff_binary: Optional[np.ndarray] = None
    similarity: Optional[np.ndarray] = None

def load_patient_data(patient_dir: str) -> tuple:
    x_path = os.path.join(patient_dir, "x_data.npy")
    y_path = os.path.join(patient_dir, "y_labels.npy")
    
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"x_data.npy not found in {patient_dir}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"y_labels.npy not found in {patient_dir}")
        
    x = np.load(x_path)
    y = np.load(y_path)
    return x, y

def compute_similarity_and_graphs(x_data: np.ndarray, config: Dict[str, Any]) -> tuple:
    """计算样本相似度并构建多尺度图。"""
    sim_cfg = config.get("similarity", {})
    graph_cfg = config.get("graph", {})
    
    # 1. Similarity
    method = sim_cfg.get("method", "pearson")
    # x_data: (N, T) -> (N, N)
    P = compute_similarity_matrix_2d(
        x_data, 
        method=method, 
        chunk_size=sim_cfg.get("chunk_size", 200),
        rbf_gamma=sim_cfg.get("rbf_gamma", 1.0),
        verbose=False
    )
    
    # 2. Filter Negative
    A, _ = filter_negative_correlations(P)
    
    # 3. Gaussian Kernel
    A_prime = apply_gaussian_kernel(A, exponent=graph_cfg.get("gaussian_exponent", 1))
    
    # 4. Partition
    n_scales = graph_cfg.get("n_scales", 10)
    mode = graph_cfg.get("partition_method", "quantile")
    
    if mode == "quantile":
        Cutoff_weight, thresholds, _ = partition_by_quantile(A_prime, k=n_scales, show_progress=False)
    else:
        Cutoff_weight, thresholds, _ = partition_by_uniform(A_prime, k=n_scales, show_progress=False)
        
    # 5. Binarize
    Cutoff_0_1 = binarize_cutoff(Cutoff_weight, thresholds)
    
    return P, Cutoff_weight, Cutoff_0_1


def build_graphs(x_data: np.ndarray, config: Dict[str, Any]) -> tuple:
    """构建多尺度图。"""
    _, Cutoff_weight, Cutoff_0_1 = compute_similarity_and_graphs(x_data, config)
    return Cutoff_weight, Cutoff_0_1


def get_perturbation_spec_list(config: Dict[str, Any]) -> List[str]:
    pert_cfg = config.get("perturbation", {})
    spec_list = list(pert_cfg.get("spectral_metrics", ["sum", "min_nonzero", "max", "mean", "std"]))
    if pert_cfg.get("use_tda", True) and "tda" not in spec_list:
        spec_list.append("tda")
    return spec_list


def build_feature_group_masks(
    total_dim: int,
    config: Dict[str, Any],
    feature_dims: Optional[Dict[str, List[int]]] = None
) -> Dict[str, np.ndarray]:
    """根据特征布局构建 S/T/D 掩码。"""
    feature_dims = feature_dims or {}
    spec_list = get_perturbation_spec_list(config)
    n_scales = config.get("graph", {}).get("n_scales", 10)

    if "D" in feature_dims:
        dim_dyn_total = feature_dims["D"][1] - feature_dims["D"][0]
    else:
        dim_dyn_total = 0

    final_mask_D_direct = np.zeros(total_dim, dtype=bool)
    if dim_dyn_total > 0:
        final_mask_D_direct[:dim_dyn_total] = True

    len_s = 0
    len_t = 0
    for metric_name in spec_list:
        if metric_name == "tda":
            len_t += 3
        else:
            len_s += 1

    dim_pert_total = total_dim - dim_dyn_total
    dim_per_scale = dim_pert_total // n_scales if n_scales > 0 else dim_pert_total
    dim_d_pert = max(0, dim_per_scale - (len_s + len_t))

    pat_s_bits = [False] * dim_per_scale
    pat_t_bits = [False] * dim_per_scale
    pat_d_bits = [False] * dim_per_scale

    curr = 0
    for metric_name in spec_list:
        if metric_name == "tda":
            for _ in range(3):
                if curr < dim_per_scale:
                    pat_t_bits[curr] = True
                curr += 1
        else:
            if curr < dim_per_scale:
                pat_s_bits[curr] = True
            curr += 1

    for _ in range(dim_d_pert):
        if curr < dim_per_scale:
            pat_d_bits[curr] = True
        curr += 1

    full_pat_s = np.tile(pat_s_bits, n_scales)
    full_pat_t = np.tile(pat_t_bits, n_scales)
    full_pat_d_pert = np.tile(pat_d_bits, n_scales)

    final_mask_S = np.zeros(total_dim, dtype=bool)
    final_mask_S[dim_dyn_total:] = full_pat_s[: total_dim - dim_dyn_total]

    final_mask_T = np.zeros(total_dim, dtype=bool)
    final_mask_T[dim_dyn_total:] = full_pat_t[: total_dim - dim_dyn_total]

    final_mask_D_pert = np.zeros(total_dim, dtype=bool)
    final_mask_D_pert[dim_dyn_total:] = full_pat_d_pert[: total_dim - dim_dyn_total]

    final_mask_D = final_mask_D_direct | final_mask_D_pert

    return {
        "D": final_mask_D,
        "D_Direct": final_mask_D_direct,
        "D_Pert": final_mask_D_pert,
        "S": final_mask_S,
        "T": final_mask_T,
    }


def save_feature_group_arrays(features: np.ndarray, group_masks: Dict[str, np.ndarray], output_dir: Path) -> None:
    """按 S/T/D 分组保存特征。"""
    np.save(output_dir / "features_S.npy", features[:, group_masks["S"]])
    np.save(output_dir / "features_T.npy", features[:, group_masks["T"]])
    np.save(output_dir / "features_D.npy", features[:, group_masks["D"]])


def persist_patient_feature_bundle(bundle: PatientFeatureBundle, output_dir: Path) -> None:
    """将已计算好的特征包写入磁盘。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    if bundle.cutoff_weight is not None:
        np.save(output_dir / "cutoff_weight.npy", bundle.cutoff_weight)
    if bundle.cutoff_binary is not None:
        np.save(output_dir / "cutoff_binary.npy", bundle.cutoff_binary)
    if bundle.similarity is not None:
        np.save(output_dir / "similarity.npy", bundle.similarity)
    np.save(output_dir / "features.npy", bundle.features)
    np.save(output_dir / "labels.npy", bundle.labels)
    save_feature_group_arrays(bundle.features, bundle.group_masks, output_dir)
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"dims": bundle.feature_dims, "feature_dim": int(bundle.features.shape[1])}, f, indent=2)


def load_patient_feature_bundle(
    bundle_dir: Path,
    config: Dict[str, Any],
    include_similarity: bool = False,
) -> PatientFeatureBundle:
    """从磁盘载入单患者特征缓存。"""
    feat_path = bundle_dir / "features.npy"
    lbl_path = bundle_dir / "labels.npy"
    meta_path = bundle_dir / "meta.json"

    if not feat_path.exists() or not lbl_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"特征缓存不完整: {bundle_dir}")

    features = np.load(feat_path)
    labels = np.load(lbl_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_dims = meta.get("dims", {"Perturbation": [0, int(features.shape[1])]})
    group_masks = build_feature_group_masks(features.shape[1], config, feature_dims)

    cutoff_weight = np.load(bundle_dir / "cutoff_weight.npy") if (bundle_dir / "cutoff_weight.npy").exists() else None
    cutoff_binary = np.load(bundle_dir / "cutoff_binary.npy") if (bundle_dir / "cutoff_binary.npy").exists() else None
    similarity = None
    if include_similarity and (bundle_dir / "similarity.npy").exists():
        similarity = np.load(bundle_dir / "similarity.npy")

    return PatientFeatureBundle(
        features=features,
        labels=labels,
        feature_dims=feature_dims,
        group_masks=group_masks,
        cutoff_weight=cutoff_weight,
        cutoff_binary=cutoff_binary,
        similarity=similarity,
    )


def compute_patient_feature_bundle(
    x_data: np.ndarray,
    labels: np.ndarray,
    config: Dict[str, Any],
    output_dir: Optional[Path] = None,
    use_cache: Optional[bool] = None,
    save_artifacts: bool = True,
    return_similarity: bool = False,
) -> PatientFeatureBundle:
    """构建或读取单患者特征及其中间产物。"""
    if use_cache is None:
        use_cache = config.get("output", {}).get("use_feature_cache", True)

    feat_path = output_dir / "features.npy" if output_dir else None
    lbl_path = output_dir / "labels.npy" if output_dir else None
    meta_path = output_dir / "meta.json" if output_dir else None

    if (
        use_cache and output_dir and feat_path.exists() and lbl_path.exists() and meta_path.exists()
    ):
        cached_bundle = load_patient_feature_bundle(output_dir, config, include_similarity=return_similarity)
        if len(cached_bundle.labels) == len(labels):
            return cached_bundle

    similarity, cutoff_weight, cutoff_binary = compute_similarity_and_graphs(x_data, config)
    features = extract_perturbation_features(cutoff_weight, cutoff_binary, config)
    feature_dims = {"Perturbation": [0, int(features.shape[1])]}
    group_masks = build_feature_group_masks(features.shape[1], config, feature_dims)

    bundle = PatientFeatureBundle(
        features=features,
        labels=labels,
        feature_dims=feature_dims,
        group_masks=group_masks,
        cutoff_weight=cutoff_weight if save_artifacts or return_similarity else None,
        cutoff_binary=cutoff_binary if save_artifacts or return_similarity else None,
        similarity=similarity if return_similarity else None,
    )
    if output_dir and save_artifacts:
        persist_patient_feature_bundle(bundle, output_dir)
    return bundle

def extract_direct_dynamics(Cutoff_0_1: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """提取直接动力学特征 (无微扰)"""
    lorenz_cfg = config.get("lorenz", {})
    num_scales, num_samples, _ = Cutoff_0_1.shape
    seed = config.get("experiment", {}).get("random_seed", 42)
    dim_dyn = 18 # 3 * 6 stats
    final_features = np.zeros((num_samples, num_scales * dim_dyn), dtype=np.float32)
    
    for s in range(num_scales):
        dyn_type = lorenz_cfg.get("type", "lorenz").lower()
        if dyn_type == "rossler":
            dyn_config = RosslerConfig(
                coupling_strength=lorenz_cfg.get("coupling_strength", 0.1),
                coupling_mode=lorenz_cfg.get("coupling_mode", "x_only"),
                dt=lorenz_cfg.get("dt", 0.01),
                total_steps=lorenz_cfg.get("total_steps", 3000), 
                steady_steps=lorenz_cfg.get("steady_steps", 1000),
            )
            oscillator = RosslerOscillator(
                 a=lorenz_cfg.get('a', 0.2),
                 b=lorenz_cfg.get('b', 0.2),
                 c=lorenz_cfg.get('c', 5.7),
            )
            res = simulate_rossler_sparse(Cutoff_0_1[s], dyn_config, oscillator, seed + s)
        else:
            dyn_config = LorenzConfig(
                coupling_strength=lorenz_cfg.get("coupling_strength", 0.42),
                coupling_mode=lorenz_cfg.get("coupling_mode", "xyz_all"),
                dt=lorenz_cfg.get("dt", 0.01),
                total_steps=lorenz_cfg.get("total_steps", 3000), 
                steady_steps=lorenz_cfg.get("steady_steps", 1000),
            )
            oscillator = LorenzOscillator(
                 delta=lorenz_cfg.get('delta', 10.0),
                 gamma=lorenz_cfg.get('gamma', 60.0),
                 beta=lorenz_cfg.get('beta', 8.0/3.0),
                 rk=lorenz_cfg.get('rk', 7.0),
            )
            res = simulate_lorenz_sparse(Cutoff_0_1[s], dyn_config, oscillator, seed + s)
            
        feat = res['per_node_features']
        col_start = s * dim_dyn
        final_features[:, col_start : col_start + dim_dyn] = feat
        
    return final_features

def extract_perturbation_features(Cutoff_weight: np.ndarray, Cutoff_0_1: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """微扰特征：几何 + TDA + 动力学。"""
    pert_cfg = config.get("perturbation", {})
    lorenz_cfg = config.get("lorenz", {})
    num_scales, num_samples, _ = Cutoff_weight.shape
    
    spec_list = pert_cfg.get("spectral_metrics", ["sum", "min_nonzero", "max", "mean", "std"])
    if pert_cfg.get("use_tda", True) and "tda" not in spec_list:
        spec_list.append("tda")
        
    dim_spec = 0
    for m in spec_list:
        dim_spec += 3 if m == 'tda' else 1

    dim_dyn = 6
        
    dim_total = dim_spec + dim_dyn
    final_features = np.zeros((num_samples, num_scales * dim_total), dtype=np.float32)
    
    # Params for dynamics
    dyn_type = lorenz_cfg.get("type", "lorenz").lower()
    if dyn_type == "rossler":
        dyn_config = RosslerConfig(
            coupling_strength=lorenz_cfg.get("coupling_strength", 0.1),
            coupling_mode=lorenz_cfg.get("coupling_mode", "x_only"),
            dt=lorenz_cfg.get("dt", 0.01),
            total_steps=lorenz_cfg.get("total_steps", 3000), 
            steady_steps=lorenz_cfg.get("steady_steps", 1000),
        )
        oscillator = RosslerOscillator(
             a=lorenz_cfg.get('a', 0.2),
             b=lorenz_cfg.get('b', 0.2),
             c=lorenz_cfg.get('c', 5.7),
        )
        solver_func = simulate_rossler_sparse
    else:
        dyn_config = LorenzConfig(
            coupling_strength=lorenz_cfg.get("coupling_strength", 0.42),
            coupling_mode=lorenz_cfg.get("coupling_mode", "xyz_all"),
            dt=lorenz_cfg.get("dt", 0.01),
            total_steps=lorenz_cfg.get("total_steps", 3000), 
            steady_steps=lorenz_cfg.get("steady_steps", 1000),
        )
        oscillator = LorenzOscillator(
             delta=lorenz_cfg.get('delta', 10.0),
             gamma=lorenz_cfg.get('gamma', 60.0),
             beta=lorenz_cfg.get('beta', 8.0/3.0),
             rk=lorenz_cfg.get('rk', 7.0),
        )
        solver_func = simulate_lorenz_sparse
    
    config_payload = {
        'spec_list': spec_list,
        'config': dyn_config,
        'oscillator': oscillator,
        'random_seed': config.get("experiment", {}).get("random_seed", 42),
        'dim_dyn': dim_dyn,
        'solver_func': solver_func
    }
    
    from ..features.metrics import calculate_spectral_metrics, calculate_dynamics_metrics
    
    pert_parallel_cfg = config.get("parallel", {})
    pert_num_workers = int(pert_parallel_cfg.get("feature_inner_workers", config.get("perturbation", {}).get("num_workers", 4)))
    pert_batch_size = int(config.get("perturbation", {}).get("batch_size", 50))
    pert_show_progress = bool(config.get("perturbation", {}).get("show_progress", True))
    total_node_evals = int(num_scales * num_samples)
    if not pert_show_progress:
        print_info(
            f"微扰特征阶段开始: scales={num_scales}, samples={num_samples}, "
            f"node_evals={total_node_evals}, inner_workers={pert_num_workers}, batch_size={pert_batch_size}"
        )

    stage_start = time.perf_counter()
    for s in range(num_scales):
        scale_start = time.perf_counter()
        mat_w = Cutoff_weight[s]
        mat_b = Cutoff_0_1[s]
        if not pert_show_progress:
            print_info(f"微扰尺度 {s + 1}/{num_scales} 开始")
        
        # Calculate Baseline
        base_spec = calculate_spectral_metrics(mat_w, spec_list)
        base_dyn = calculate_dynamics_metrics(mat_b, solver_func, config_payload) # (dim_dyn,)
        
        pert_spec, pert_dyn = run_perturbation_analysis_on_scale(
            s, mat_w, mat_b, config_payload,
            n_workers=pert_num_workers,
            batch_size=pert_batch_size,
            dim_spec=dim_spec,
            dim_dyn=dim_dyn,
            show_progress=pert_show_progress,
        )
        
        delta_spec = np.abs(base_spec - pert_spec)
        delta_dyn = np.abs(base_dyn - pert_dyn)
        
        col_start = s * dim_total
        final_features[:, col_start : col_start + dim_spec] = delta_spec
        final_features[:, col_start + dim_spec : col_start + dim_total] = delta_dyn

        if not pert_show_progress:
            elapsed_scale = time.perf_counter() - scale_start
            elapsed_total = time.perf_counter() - stage_start
            scales_done = s + 1
            avg_per_scale = elapsed_total / scales_done
            eta_sec = max(0.0, (num_scales - scales_done) * avg_per_scale)
            print_info(
                f"微扰尺度 {scales_done}/{num_scales} 完成: "
                f"scale_elapsed={elapsed_scale:.1f}s, total_elapsed={elapsed_total:.1f}s, eta={eta_sec:.1f}s"
            )
        
    return final_features

def classify_patient(
    features: np.ndarray,
    labels: np.ndarray,
    config: Dict[str, Any],
    feature_dims: Dict[str, List[int]],
    output_dir: Optional[Path] = None,
    eval_features: Optional[np.ndarray] = None,
    evaluation_mode: str = "within_domain",
    patient_id: Optional[str] = None,
    pretrained_params_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    分类并进行消融实验 (支持参数调优)
    feature_dims: {'D': [start, end], 'S': [start, end], 'T': ...}
    """
    results = {}
    
    total_dim = features.shape[1]
    masks_map = build_feature_group_masks(total_dim, config, feature_dims)
    
    experiments = {
        "S": ["S"], 
        "T": ["T"], 
        "D": ["D"], 
        "S+T": ["S", "T"], 
        "S+D": ["S", "D"], 
        "T+D": ["T", "D"], 
        "ALL": ["S", "T", "D"]
    }
    
    random_seed = config.get("experiment", {}).get("random_seed", 42)
    models = get_models(random_seed)
    cls_cfg = config.get("classification", {})
    parameter_strategy = str(cls_cfg.get("parameter_strategy", "grid_search")).lower()
    fallback_to_default = bool(cls_cfg.get("fallback_to_default", True))
    
    from ..classification.evaluation import evaluate_model_cv, evaluate_model_cv_transfer
    from sklearn.base import clone

    def _resolve_pretrained_params_root() -> Optional[Path]:
        if pretrained_params_dir is not None:
            return Path(pretrained_params_dir)
        root = cls_cfg.get("pretrained_params_root")
        if not root or not patient_id:
            return None
        subdir = cls_cfg.get("pretrained_params_subdir", "models")
        root_path = Path(os.path.expandvars(os.path.expanduser(str(root))))
        return root_path / patient_id / subdir

    resolved_pretrained_dir = _resolve_pretrained_params_root()

    def _load_pretrained_params(exp_name: str, model_name: str) -> Optional[Dict[str, Any]]:
        if resolved_pretrained_dir is None:
            return None
        pm_path = resolved_pretrained_dir / f"{exp_name}_{model_name}_params.json"
        if not pm_path.exists():
            return None
        with open(pm_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "params" in payload and isinstance(payload["params"], dict):
            return payload["params"]
        return payload
    
    for exp_name, comps in experiments.items():
        # Combine masks
        combined_mask = np.zeros(total_dim, dtype=bool)
        for c in comps:
            combined_mask |= masks_map[c]
            
        if not np.any(combined_mask):
            continue
            
        X_sub = features[:, combined_mask]
        X_eval_sub = eval_features[:, combined_mask] if eval_features is not None else X_sub
        
        # Prepare models dir
        models_dir = None
        if output_dir:
            models_dir = output_dir / "models"
            models_dir.mkdir(exist_ok=True, parents=True)
        
        for model_name, model in models.items():
            clf_cfg = cls_cfg.get("classifiers", {}).get(model_name, {})
            tuned_combos = cls_cfg.get("tuned_feature_combinations", ["ALL", "D+S", "D+T"])
            do_grid = clf_cfg.get("enabled", True) and clf_cfg.get("grid_search", False) and (exp_name in tuned_combos)
            param_grid_raw = clf_cfg.get("params", {})

            selected_model = None
            selected_params = None
            selected_param_source = "default"
            final_metrics = None

            if parameter_strategy == "pretrained":
                pretrained_params = _load_pretrained_params(exp_name, model_name)
                if pretrained_params is not None:
                    selected_model = build_model(model_name, random_seed=random_seed, params=pretrained_params)
                    selected_params = pretrained_params
                    selected_param_source = "pretrained"
                    if evaluation_mode == "train_clean_test_noisy" and eval_features is not None:
                        final_metrics, _ = evaluate_model_cv_transfer(
                            clone(selected_model), X_sub, X_eval_sub, labels, cv_splits=5
                        )
                    else:
                        final_metrics, _ = evaluate_model_cv(
                            clone(selected_model), X_sub, labels, cv_splits=5
                        )
                elif not fallback_to_default:
                    raise FileNotFoundError(
                        f"未找到预训练参数: patient={patient_id}, exp={exp_name}, model={model_name}"
                    )

            if final_metrics is None and do_grid and param_grid_raw and parameter_strategy == "grid_search":
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', model)
                ])

                param_grid = {f"clf__{k}": v for k, v in param_grid_raw.items()}
                gs = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=1, verbose=0)
                gs.fit(X_sub, labels)

                best_pipe = gs.best_estimator_
                final_clf = best_pipe.named_steps['clf']
                best_params = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}

                clf_tuned = clone(final_clf)
                if evaluation_mode == "train_clean_test_noisy" and eval_features is not None:
                    metrics_tuned, _ = evaluate_model_cv_transfer(
                        clf_tuned, X_sub, X_eval_sub, labels, cv_splits=5
                    )
                else:
                    metrics_tuned, _ = evaluate_model_cv(clf_tuned, X_sub, labels, cv_splits=5)

                metrics_default = None
                if evaluation_mode == "train_clean_test_noisy" and eval_features is not None:
                    metrics_default, _ = evaluate_model_cv_transfer(
                        clone(model), X_sub, X_eval_sub, labels, cv_splits=5
                    )
                else:
                    metrics_default, _ = evaluate_model_cv(clone(model), X_sub, labels, cv_splits=5)

                if metrics_tuned.get("Accuracy", 0) >= metrics_default.get("Accuracy", 0):
                    final_metrics = metrics_tuned
                    selected_model = final_clf
                    selected_params = best_params
                    selected_param_source = "grid_search"
                else:
                    final_metrics = metrics_default
                    selected_model = clone(model)
                    selected_param_source = "default"

            if final_metrics is None:
                selected_model = clone(model)
                if evaluation_mode == "train_clean_test_noisy" and eval_features is not None:
                    final_metrics, _ = evaluate_model_cv_transfer(
                        clone(selected_model), X_sub, X_eval_sub, labels, cv_splits=5
                    )
                else:
                    final_metrics, _ = evaluate_model_cv(clone(selected_model), X_sub, labels, cv_splits=5)

            results[f"{exp_name}_{model_name}"] = final_metrics

            if models_dir:
                md_path = models_dir / f"{exp_name}_{model_name}.joblib"
                joblib.dump(selected_model, md_path)
                pm_path = models_dir / f"{exp_name}_{model_name}_params.json"
                payload = {
                    "parameter_source": selected_param_source,
                    "params": selected_params or {},
                }
                with open(pm_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
            
    return results

def process_single_patient(
    patient_id: str,
    patient_dir: str,
    config: Dict[str, Any],
    output_dir: Path
) -> PatientResult:
    
    result = PatientResult(patient_id=patient_id)
    pat_out = output_dir / patient_id
    pat_out.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Load
        x, y = load_patient_data(patient_dir)
        result.n_samples = len(y)
        result.n_class0 = int(np.sum(y==0))
        result.n_class1 = int(np.sum(y==1))
        
        if result.n_class0 < 5 or result.n_class1 < 5:
            result.error = "Insufficient samples"
            return result
            
        feature_bundle = compute_patient_feature_bundle(
            x, y, config, output_dir=pat_out, use_cache=config.get("output", {}).get("use_feature_cache", True)
        )
        result.feature_dim = feature_bundle.features.shape[1]
        
        # 4. Classify
        clf_results = classify_patient(
            feature_bundle.features,
            y,
            config,
            feature_bundle.feature_dims,
            output_dir=pat_out,
        )
        result.classifier_results = clf_results
        
        # Save Result JSON
        with open(pat_out / "result_metrics.json", "w") as f:
            # Convert numpy types
            def ser(o):
                if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
                    np.int16, np.int32, np.int64, np.uint8,
                    np.uint16, np.uint32, np.uint64)): return int(o)
                if isinstance(o, (np.float_, np.float16, np.float32, np.float64)): return float(o)
                return str(o)
            json.dump(asdict(result), f, default=ser, indent=2)
            
    except Exception as e:
        result.error = str(e)
        console.print_exception()
        
    return result

def run_patient_analysis_pipeline(config: Dict[str, Any]):
    print_header("Step 6: 单患者分析")
    
    data_cfg = config.get("data", {})
    patients_dir = Path(data_cfg.get("patients_data_dir", ""))
    output_dir = Path(config.get("output", {}).get("output_dir", "./output/patients"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not patients_dir.exists():
        print_error(f"患者数据目录不存在: {patients_dir}")
        return

    # Scan patients
    patients = [p.name for p in patients_dir.iterdir() if p.is_dir() and (p/"x_data.npy").exists()]
    patients.sort()
    
    # Filter
    whitelist = data_cfg.get("patients", "all")
    if whitelist != "all":
        patients = [p for p in patients if p in whitelist]
    exclude = data_cfg.get("exclude_patients", [])
    patients = [p for p in patients if p not in exclude]
    
    print_info(f"待处理患者数: {len(patients)}")
    
    results = []
    
    # Parallel
    par_cfg = config.get("parallel", {})
    max_workers = par_cfg.get("max_patient_workers", 4)
    do_parallel = par_cfg.get("patient_parallel", True)
    
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    if do_parallel and len(patients) > 1:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Processing patients...", total=len(patients))
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_single_patient, pid, str(patients_dir/pid), config, output_dir): pid for pid in patients}
                
                for future in as_completed(futures):
                    pid = futures[future]
                    try:
                        res = future.result()
                        results.append(res)
                        
                        # Find best accuracy for logging
                        best_acc = 0.0
                        if res.classifier_results:
                            for key, val in res.classifier_results.items():
                                if "ALL_" in key and "Accuracy" in val:
                                    acc = val["Accuracy"]
                                    if acc > best_acc: best_acc = acc
                        
                        status = "✓" if not res.error else "✗"
                    except Exception as exc:
                        print_error(f"Patient {pid} failed: {exc}")
                    
                    progress.advance(task)
    else:
        for pid in patients:
            try:
                results.append(process_single_patient(pid, str(patients_dir / pid), config, output_dir))
            except Exception as exc:
                print_error(f"Patient {pid} failed: {exc}")
            
    # Summary
    df_data = []
    for r in results:
        d = asdict(r)
        d.pop('classifier_results') 
        for exp_key, metrics_dict in r.classifier_results.items():
            for metric_k, metric_v in metrics_dict.items():

                if isinstance(metric_v, (int, float)):
                    d[f"{exp_key}_{metric_k}"] = metric_v
        df_data.append(d)
        
    if df_data:
        df = pd.DataFrame(df_data)
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if "patient_id" in numeric_cols:
            numeric_cols.remove("patient_id")
        mean_row = df[numeric_cols].mean()
        mean_row["patient_id"] = "AVERAGE"
        
        df_final = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
        
        cols = ["patient_id"] + [c for c in df_final.columns if c != "patient_id"]
        df_final = df_final[cols]
        
        df_final.to_csv(output_dir / "summary.csv", index=False)
        print_success(f"分析完成，汇总已保存 (含平均值): {output_dir}/summary.csv")
    else:
        print_warning("未生成任何结果数据")
