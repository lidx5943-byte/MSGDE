# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""
尺度消融实验流水线 (Scale Ablation Pipeline)

功能：
1. 加载微扰特征 (Perturbation Features)
2. 按尺度拆分特征 (Scale-wise splitting)
3. 对每个尺度和每个分类器运行评估
4. 汇总所有尺度的指标结果并保存
"""

import os
import time
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, List

from ..utils.console import (
    console, print_header, print_step, print_success, print_error, print_info
)
from ..classification import get_models, evaluate_model_cv
from ..visualization.scale_ablation_plotter import plot_scale_ablation_results

def run_scale_ablation_pipeline(config: Dict[str, Any]):
    """运行尺度消融实验"""
    print_header("尺度消融实验 (Scale-wise Ablation)")
    
    start_time = time.time()
    
    # 1. 准备配置和输出目录
    data_cfg = config.get("data", {})
    scale_cfg = config.get("scale", {})
    output_cfg = config.get("output", {})
    
    output_dir = Path(output_cfg.get("save_dir", "./output/scale_ablation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. 加载数据
    print_step(1, 4, "加载特征与标签")
    
    pert_feat_path = Path(data_cfg.get("perturbation_features_path"))
    labels_path = Path(data_cfg.get("labels_path"))
    
    if not pert_feat_path.exists():
        print_error(f"微扰特征文件不存在: {pert_feat_path}")
        return
        
    # 自动推断标签路径 (适配单病人目录结构)
    if not labels_path.exists():
        # 尝试同目录下的 labels.npy
        potential_labels = pert_feat_path.parent / "labels.npy"
        if potential_labels.exists():
            labels_path = potential_labels
            print_info(f"自动定位标签文件: {labels_path}")
        else:
            print_error(f"标签文件不存在: {labels_path}")
            return
            
    X_all = np.load(pert_feat_path)
    y = np.load(labels_path)
    
    print_info(f"Loaded Features: {X_all.shape}")
    print_info(f"Loaded Labels: {y.shape}")
    
    # 3. 按尺度拆分特征
    print_step(2, 4, "特征预处理与分尺度拆分")
    
    n_scales = scale_cfg.get("n_scales", 10)
    dim_per_scale = scale_cfg.get("dim_per_scale", 12)
    
    # 检查维度是否匹配
    expected_dim = n_scales * dim_per_scale
    if X_all.shape[1] != expected_dim:
        print_error(f"特征维度 ({X_all.shape[1]}) 与预期 ({expected_dim}) 不符!")
        # 尝试自适应调整，如果可能的话
        if X_all.shape[1] % n_scales == 0:
            dim_per_scale = X_all.shape[1] // n_scales
            print_warning(f"自适应调整 dim_per_scale 为: {dim_per_scale}")
        else:
            return

    # 数据归一化 (NaN 处理)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 4. 运行分尺度评估
    print_step(3, 4, "执行分尺度分类评估")
    
    models_dict = get_models(random_seed=config.get("experiment", {}).get("random_seed", 42))
    
    # 只取配置中启用的模型 (如果有配置的话)
    active_classifiers = config.get("classification", {}).get("classifiers", {})
    models = {}
    for name, model in models_dict.items():
        if active_classifiers.get(name, {}).get("enabled", True):
            models[name] = model
            
    all_results = []
    
    total_tasks = n_scales * len(models)
    current_task = 0
    
    for s_idx in range(n_scales):
        # 提取当前尺度的特征
        start_col = s_idx * dim_per_scale
        end_col = (s_idx + 1) * dim_per_scale
        X_scale = X_all[:, start_col:end_col]
        
        console.print(f"\n[bold cyan]分析尺度 {s_idx+1}/{n_scales} (特征维度: {X_scale.shape[1]})[/bold cyan]")
        
        for model_name, model in models.items():
            current_task += 1
            start_task_time = time.time()
            
            # 使用副本模型
            from sklearn.base import clone
            clf = clone(model)
            
            # 运行评估
            metrics, _ = evaluate_model_cv(
                clf, X_scale, y,
                cv_splits=config.get("classification", {}).get("n_folds", 5),
                model_name=model_name,
                feature_set_name=f"Scale_{s_idx}"
            )
            
            # 记录结果，添加尺度索引
            metrics['Scale'] = s_idx + 1 # 从 1 开始计数
            all_results.append(metrics)
            
            elapsed = time.time() - start_task_time
            console.print(f"  - [Scale {s_idx+1}] {model_name}: Acc={metrics['Accuracy']:.4f} ({elapsed:.1f}s)")
            
    # 5. 保存并可视化
    print_step(4, 4, "保存结果与生成可视化图表")
    
    # 保存 CSV
    df = pd.DataFrame(all_results)
    results_csv = output_dir / "scale_ablation_results.csv"
    df.to_csv(results_csv, index=False)
    print_success(f"结果已保存至: {results_csv}")
    
    # 生成可视化
    if output_cfg.get("generate_plots", True):
        try:
            plot_scale_ablation_results(df, output_dir, config)
            print_success("可视化图表生成成功")
        except Exception as e:
            print_error(f"生成图表失败: {e}")
            import traceback
            traceback.print_exc()
            
    total_elapsed = time.time() - start_time
    print_header(f"尺度消融实验完成! 总耗时: {total_elapsed:.1f}s")
