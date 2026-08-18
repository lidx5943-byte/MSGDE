# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""Step 4：节点移除微扰，输出 |Δ| 特征。"""

import os
import time
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any

from ..utils.console import (
    console, print_header, print_step, print_success, print_error, print_warning, print_info
)
from ..features.perturbation import run_perturbation_analysis_on_scale
from ..features.metrics import calculate_spectral_metrics, calculate_dynamics_metrics
from ..features.lorenz import (
    simulate_lorenz_sparse, LorenzConfig, LorenzOscillator,
    simulate_rossler_sparse, RosslerConfig, RosslerOscillator
)

def run_perturbation_pipeline(config: Dict[str, Any]):
    print_header("Step 4: 微扰特性分析")
    
    start_time = time.time()
    
    # 1. 准备路径
    print_step(1, 5, "准备数据和输出目录")
    
    data_cfg = config.get("data", {})
    subgraphs_path = data_cfg.get("subgraphs_path")
    weights_path = data_cfg.get("weights_path")
    labels_path = data_cfg.get("labels_path")
    output_dir = Path(data_cfg.get("output_dir", "./output/perturbation"))
    
    if not subgraphs_path or not os.path.exists(subgraphs_path):
        print_error(f"子图文件不存在: {subgraphs_path}")
        return
    if not weights_path or not os.path.exists(weights_path):
        print_error(f"权重文件不存在: {weights_path}")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. 加载数据
    print_step(2, 5, f"加载数据")
    try:
        # mmap_mode 节省内存
        subgraphs_0_1 = np.load(subgraphs_path, mmap_mode='r')
        subgraphs_weight = np.load(weights_path, mmap_mode='r')
        if labels_path and os.path.exists(labels_path):
            labels = np.load(labels_path)
            np.save(output_dir / "labels.npy", labels)
            
        num_scales, num_samples, _ = subgraphs_0_1.shape
        print_success(f"数据加载成功: {num_scales}个尺度, {num_samples}个样本")
    except Exception as e:
        print_error(f"数据加载失败: {e}")
        return

    # 3. 准备配置参数
    lorenz_cfg = config.get('lorenz', {})
    proc_cfg = config.get('processing', {})
    metrics_cfg = config.get('metrics', {})
    
    spec_list = metrics_cfg.get('spectral', ["sum", "min_nonzero", "max", "mean", "std"])
    dyn_list = metrics_cfg.get('dynamics', ["mean", "max", "min", "median", "var", "std"])
    
    # 计算维度
    dim_spec = 0
    for m in spec_list:
        if m == 'tda':
            dim_spec += 3
        else:
            dim_spec += 1
            
    dim_dyn = len(dyn_list) # 假定 dyn_list 的长度对应维度，通常是 6
    dim_total = dim_spec + dim_dyn
    
    print_info(f"特征维度: 几何={dim_spec}, 动力学={dim_dyn}, 总计={dim_total}/尺度")
    
    # 构建动力学对象
    dyn_type = lorenz_cfg.get("type", "lorenz").lower()
    if dyn_type == "rossler":
        dyn_config_obj = RosslerConfig(
            coupling_strength=lorenz_cfg.get('coupling_strength', 0.1),
            coupling_mode=lorenz_cfg.get('coupling_mode', 'x_only'),
            dt=lorenz_cfg.get('dt', 0.01),
            total_steps=lorenz_cfg.get('total_steps', 3000),
            steady_steps=lorenz_cfg.get('steady_steps', 1000),
        )
        oscillator_obj = RosslerOscillator(
             a=lorenz_cfg.get('a', 0.2),
             b=lorenz_cfg.get('b', 0.2),
             c=lorenz_cfg.get('c', 5.7),
        )
        solver_func = simulate_rossler_sparse
    else:
        dyn_config_obj = LorenzConfig(
            coupling_strength=lorenz_cfg.get('coupling_strength', 0.42),
            coupling_mode=lorenz_cfg.get('coupling_mode', 'xyz_all'),
            dt=lorenz_cfg.get('dt', 0.01),
            total_steps=lorenz_cfg.get('total_steps', 3000),
            steady_steps=lorenz_cfg.get('steady_steps', 1000),
        )
        oscillator_obj = LorenzOscillator(
             delta=lorenz_cfg.get('delta', 10.0),
             gamma=lorenz_cfg.get('gamma', 60.0),
             beta=lorenz_cfg.get('beta', 8.0/3.0),
             rk=lorenz_cfg.get('rk', 7.0),
        )
        solver_func = simulate_lorenz_sparse
    
    config_payload = {
        'spec_list': spec_list,
        'config': dyn_config_obj,
        'oscillator': oscillator_obj,
        'random_seed': config.get('experiment', {}).get('random_seed', 42),
        'solver_func': solver_func,
        'dim_dyn': dim_dyn
    }
    
    # 4. 执行微扰分析
    final_features = np.zeros((num_samples, num_scales * dim_total), dtype=np.float32)
    
    n_workers = proc_cfg.get('n_workers', 10)
    batch_size = proc_cfg.get('batch_size', 50)
    
    for s_idx in range(num_scales):
        print_step(4, 5, f"处理尺度 {s_idx + 1}/{num_scales}")
        
        mat_w = subgraphs_weight[s_idx]
        mat_b = subgraphs_0_1[s_idx]
        
        console.print("  [dim]计算全图基准指标...[/dim]")

        base_spec = calculate_spectral_metrics(mat_w, spec_list)
        base_dyn = calculate_dynamics_metrics(mat_b, solver_func, config_payload)
        
        # --- 微扰分析 ---
        perturbed_spec, perturbed_dyn = run_perturbation_analysis_on_scale(
            s_idx, mat_w, mat_b, config_payload,
            n_workers=n_workers,
            batch_size=batch_size,
            dim_spec=dim_spec,
            dim_dyn=dim_dyn
        )
        
        # --- 计算差异 ---
        # 广播基准值: (dim,) -> (N, dim)
        delta_spec = np.abs(base_spec - perturbed_spec)
        delta_dyn = np.abs(base_dyn - perturbed_dyn)
        
        # 拼接
        col_start = s_idx * dim_total
        final_features[:, col_start : col_start + dim_spec] = delta_spec
        final_features[:, col_start + dim_spec : col_start + dim_total] = delta_dyn
        
        print_success(f"尺度 {s_idx} 完成")

    # 5. 保存结果
    np.save(output_dir / "perturbation_features.npy", final_features)
    print_success(f"微扰特征已保存: {final_features.shape}")
    
    print_time = time.time() - start_time
    print_success(f"Step 4 完成! 总耗时: {print_time:.2f} 秒")
