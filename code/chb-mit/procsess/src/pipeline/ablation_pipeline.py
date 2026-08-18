# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""消融：按配置组合几何/拓扑/动力学等特征并交叉验证。"""

import os
import time
import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
from typing import Dict, Any

from ..utils.console import (
    console, print_header, print_step, print_success, print_error, print_warning, print_info
)
from ..features.baseline import extract_baseline_features_batch
from ..classification import get_models, evaluate_model_cv
from ..visualization.ablation_plotter import plot_ablation_results

def load_data_for_ablation(config: Dict[str, Any]) -> tuple:
    """加载消融所需的特征矩阵与标签。"""
    data_cfg = config.get("data", {})

    patient_dir = data_cfg.get("patient_dir")
    if patient_dir and os.path.exists(patient_dir):
        print_info(f"单病人目录: {patient_dir}")
        patient_dir = Path(patient_dir)

        x_raw_path = patient_dir / "x_data.npy"
        labels_path = patient_dir / "labels.npy"
        if not labels_path.exists():
            labels_path = patient_dir / "y_labels.npy"

        patient_id = patient_dir.name
        per_patient_results_dir = patient_dir.parent if "per_patient_results" in str(patient_dir) else patient_dir.parent / "per_patient_results"
        patient_result_dir = per_patient_results_dir / patient_id

        dyn_feat = None
        tda_feat = None
        pert_feat = None

        if patient_result_dir.exists():
            features_path = patient_result_dir / "features.npy"
            meta_path = patient_result_dir / "meta.json"

            if features_path.exists() and meta_path.exists():
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                        feature_dims = meta.get("dims", {})

                    all_features = np.load(features_path)
                    print_info(f"单病人特征: {all_features.shape}")

                    if "D" in feature_dims:
                        d_start, d_end = feature_dims["D"]
                        dyn_feat = all_features[:, d_start:d_end]
                        print_info(f"Dynamics: {dyn_feat.shape}")

                    if "Perturbation" in feature_dims:
                        p_start, p_end = feature_dims["Perturbation"]
                        pert_feat = all_features[:, p_start:p_end]
                        print_info(f"Perturbation: {pert_feat.shape}")

                except Exception as e:
                    print_warning(f"单病人特征加载失败: {e}")

        if dyn_feat is None and pert_feat is None:
            print_warning("未找到单病人特征，请先运行单病人分析")
    else:
        x_raw_path = data_cfg.get("x_data_path")
        dyn_feat_path = data_cfg.get("dynamics_path") or data_cfg.get("dynamics_features_path")
        tda_feat_path = data_cfg.get("tda_path") or data_cfg.get("tda_features_path")
        pert_feat_path = data_cfg.get("perturbation_path") or data_cfg.get("perturbation_features_path")
        labels_path = data_cfg.get("labels_path") or data_cfg.get("y_labels_path")

        if dyn_feat_path and os.path.exists(dyn_feat_path):
            dyn_feat = np.load(dyn_feat_path)
            print_info(f"Dynamics 特征: {dyn_feat.shape}")
        else:
            print_warning(f"未找到 Dynamics: {dyn_feat_path}")
            dyn_feat = None

        if tda_feat_path and os.path.exists(tda_feat_path):
            tda_feat = np.load(tda_feat_path)
            print_info(f"TDA 特征: {tda_feat.shape}")
        else:
            print_warning(f"未找到 TDA: {tda_feat_path}")
            tda_feat = None

        if pert_feat_path and os.path.exists(pert_feat_path):
            pert_feat = np.load(pert_feat_path)
            print_info(f"Perturbation 特征: {pert_feat.shape}")
        else:
            print_warning(f"未找到 Perturbation: {pert_feat_path}")
            pert_feat = None

    if x_raw_path and os.path.exists(x_raw_path):
        x_raw = np.load(x_raw_path, mmap_mode='r')
        if x_raw.ndim == 3:
            x_raw = x_raw.squeeze()
        print_info(f"原始数据: {x_raw.shape}")
    else:
        print_warning(f"未指定或不存在原始数据: {x_raw_path}")
        x_raw = None

    if labels_path and os.path.exists(labels_path):
        y = np.load(labels_path)
        print_info(f"标签: {y.shape}")
    else:
        print_error(f"未找到标签: {labels_path}")
        y = None

    return x_raw, dyn_feat, tda_feat, pert_feat, y

def run_ablation_pipeline(config: Dict[str, Any]):
    """运行消融流水线。"""
    print_header("消融实验")

    start_time = time.time()

    output_config = config.get("output", {})
    output_dir = Path(output_config.get("save_dir", "./output/ablation"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print_step(1, 5, "加载数据")
    x_raw, dyn_feat, tda_feat, pert_feat, y = load_data_for_ablation(config)

    if y is None:
        return

    base_components = {}

    if dyn_feat is not None:
        base_components["D_global"] = dyn_feat

    if pert_feat is not None:
        try:
            num_scales = 10
            dim_per_scale = 12

            pert_reshaped = pert_feat.reshape(pert_feat.shape[0], num_scales, dim_per_scale)

            p_geo = pert_reshaped[:, :, 0:3].reshape(pert_feat.shape[0], -1)
            base_components["S"] = p_geo

            p_topo = pert_reshaped[:, :, 3:6].reshape(pert_feat.shape[0], -1)
            base_components["T"] = p_topo

            p_dyn = pert_reshaped[:, :, 6:12].reshape(pert_feat.shape[0], -1)
            base_components["D_pert"] = p_dyn

        except Exception as e:
            print_error(f"Perturbation 拆分失败: {e}")

    logical_components = {}

    if "S" in base_components:
        logical_components["S"] = base_components["S"]

    if "T" in base_components:
        logical_components["T"] = base_components["T"]

    if "D_pert" in base_components:
        logical_components["D"] = base_components["D_pert"]

    feature_sets = {}

    if x_raw is not None:
        try:
            baseline_feat = extract_baseline_features_batch(x_raw)
            feature_sets["Baseline"] = baseline_feat
        except Exception as e:
            print_warning(f"Baseline 特征失败: {e}")

    ablation_cfg = config.get("ablation", {})
    experiments = ablation_cfg.get("experiments", [])

    if not experiments:
        print_warning("未配置 ablation.experiments，使用默认全集。")
        if logical_components:
            all_feats = list(logical_components.values())
            feature_sets["ALL (Default)"] = np.hstack(all_feats)
    else:
        for exp in experiments:
            exp_name = exp.get("name")
            components = exp.get("components", [])

            parts_to_combine = []
            valid_exp = True

            for comp_key in components:
                if comp_key in logical_components:
                    parts_to_combine.append(logical_components[comp_key])
                else:
                    print_warning(f"实验「{exp_name}」缺少组件「{comp_key}」。")
                    valid_exp = False

            if valid_exp and parts_to_combine:
                sizes = [p.shape[0] for p in parts_to_combine]
                if len(set(sizes)) > 1:
                    print_error(f"实验「{exp_name}」样本行数不一致: {sizes}")
                    continue

                feature_sets[exp_name] = np.hstack(parts_to_combine)

    print_info(f"特征组合: {list(feature_sets.keys())}")

    print_step(3, 5, "运行评估")

    models = get_models(random_seed=config.get("experiment", {}).get("random_seed", 42))
    results = []
    roc_data = {}

    data_stats = {}
    for fs_name, X in feature_sets.items():
        data_stats[fs_name] = {
            "Shape": X.shape,
            "NaN": int(np.isnan(X).sum()),
            "Inf": int(np.isinf(X).sum())
        }
        feature_sets[fs_name] = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    with open(output_dir / "ablation_debug.json", "w") as f:
        json.dump({"Data_Stats": data_stats}, f, indent=4)

    total_tasks = len(feature_sets) * len(models)
    current_task = 0

    for fs_name, X in feature_sets.items():
        for model_name, model in models.items():
            current_task += 1
            start_task = time.time()

            from sklearn.base import clone
            clf = clone(model)

            metrics, roc_info = evaluate_model_cv(
                clf, X, y,
                cv_splits=5,
                model_name=model_name,
                feature_set_name=fs_name
            )

            if roc_info:
                if model_name not in roc_data:
                    roc_data[model_name] = {}
                roc_data[model_name][fs_name] = roc_info

            results.append(metrics)

            elapsed = time.time() - start_task
            console.print(f"[{current_task}/{total_tasks}] {fs_name} + {model_name}: Acc={metrics['Accuracy']:.4f} ({elapsed:.1f}s)")

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "ablation_results.csv", index=False)
    joblib.dump(roc_data, output_dir / "roc_curves_data.pkl")

    print_success(f"结果: {output_dir}/ablation_results.csv")

    print_step(5, 5, "生成可视化")
    try:
        plot_ablation_results(output_dir)
        print_success("可视化完成")
    except Exception as e:
        print_error(f"可视化失败: {e}")

    print_time = time.time() - start_time
    print_success(f"消融完成，耗时 {print_time:.2f} s")
