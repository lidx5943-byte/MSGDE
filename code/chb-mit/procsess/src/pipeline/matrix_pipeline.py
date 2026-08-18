# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""
矩阵生成流水线
"""

import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from ..utils.console import (
    console, print_header, print_step, print_success, print_error, print_warning, print_info
)
from ..matrix.similarity import compute_similarity_matrix, compute_similarity_matrix_2d
from ..matrix.graph_builder import (
    filter_negative_correlations,
    apply_gaussian_kernel,
    partition_by_uniform,
    partition_by_quantile,
    binarize_cutoff,
    compute_laplacian
)
from ..matrix.report_generator import generate_matrix_report, print_summary as print_report_summary


def run_matrix_generation_pipeline(config: Dict[str, Any]):
    """
    运行矩阵生成流水线
    
    Args:
        config: 配置字典
    """
    print_header("Step 2: 矩阵生成")
    
    start_time = time.time()
    
    # 1. 准备路径
    print_step(1, 9, "准备数据和输出目录")
    
    data_config = config.get("data", {})
    x_path = data_config.get("x_data_path")
    
    output_config = config.get("output", {})
    save_dir = Path(output_config.get("save_dir", "./output"))
    save_intermediate = output_config.get("save_intermediate", True)
    
    if not x_path or not os.path.exists(x_path):
        print_error(f"输入数据文件不存在: {x_path}")
        return
        
    save_dir.mkdir(parents=True, exist_ok=True)
    print_info(f"输出目录: {save_dir}")
    
    # 2. 加载数据
    print_step(2, 9, f"加载数据: {x_path}")
    try:
        x_data = np.load(x_path, mmap_mode='r')
        print_success(f"数据加载成功: {x_data.shape}, dtype={x_data.dtype}")
        
        pass
    except Exception as e:
        print_error(f"数据加载失败: {e}")
        return
    
    # 获取参数
    sim_config = config.get("similarity", {})
    method = sim_config.get("method", "pearson")
    rbf_gamma = float(sim_config.get("rbf_gamma", 1.0))
    chunk_size = int(sim_config.get("chunk_size", 500))
    
    graph_config = config.get("graph", {})  
    if not graph_config and "subgraph" in config:
        graph_config = config["subgraph"]
        
    partition_method = graph_config.get("partition_method", "quantile")
    k = int(graph_config.get("subgraph_count", graph_config.get("k", 10)))
    
    gaussian_config = config.get("gaussian_kernel", {})
    exponent = int(gaussian_config.get("exponent", 1))
    sigma = gaussian_config.get("sigma", "auto")
    
    # 3. 计算相似度矩阵
    print_step(3, 9, f"计算相似度矩阵 ({method})")
    try:
        # 根据数据维度选择相应的函数
        if x_data.ndim == 2:
            # 2D 数据 (N, T) - 单通道或多通道已展平
            print_info(f"检测到2D数据 (shape={x_data.shape})，使用 compute_similarity_matrix_2d")
            P = compute_similarity_matrix_2d(
                x_data, 
                method=method, 
                chunk_size=chunk_size, 
                rbf_gamma=rbf_gamma,
                verbose=True
            )
        elif x_data.ndim == 3:
            # 3D 数据 (N, C, T) - 多通道数据
            print_info(f"检测到3D数据 (shape={x_data.shape})，使用 compute_similarity_matrix")
            P = compute_similarity_matrix(
                x_data, 
                method=method, 
                chunk_size=chunk_size, 
                rbf_gamma=rbf_gamma
            )
        else:
            print_error(f"不支持的数据维度: {x_data.ndim}，期望 2 或 3")
            return
        
        if save_intermediate:
            np.save(save_dir / "similarity_matrix.npy", P)
            print_info("已保存: similarity_matrix.npy")
            
    except Exception as e:
        print_error(f"相似度计算失败: {e}")
        import traceback
        traceback.print_exc()
        return
        
    # 4. 过滤负相关
    print_step(4, 9, "过滤负相关")
    A, filter_stats = filter_negative_correlations(P)
    
    # 5. 高斯核映射
    print_step(5, 9, "应用高斯核映射")
    A_prime = apply_gaussian_kernel(A, exponent=exponent, sigma=sigma)
    
    if save_intermediate:
        np.save(save_dir / "Gaussian_matrix.npy", A_prime)
        print_info("已保存: Gaussian_matrix.npy")
        
    # 6. 子图划分
    print_step(6, 9, f"子图划分 (k={k}, method={partition_method})")
    
    if partition_method == "uniform":
        Cutoff_weight, thresholds, subgraph_stats = partition_by_uniform(A_prime, k=k)
    else:  # quantile
        Cutoff_weight, thresholds, subgraph_stats = partition_by_quantile(A_prime, k=k)
        
    np.save(save_dir / "Cutoff_weight.npy", Cutoff_weight)
    np.save(save_dir / "thresholds.npy", thresholds)
    print_info("已保存: Cutoff_weight.npy, thresholds.npy")
    
    # 7. 二值化
    print_step(7, 9, "子图二值化")
    Cutoff_0_1 = binarize_cutoff(Cutoff_weight, thresholds)
    
    np.save(save_dir / "Cutoff_0_1.npy", Cutoff_0_1)
    print_success(f"已保存核心文件: Cutoff_0_1.npy (shape={Cutoff_0_1.shape})")
    
    # 8. 计算拉普拉斯矩阵
    print_step(8, 9, "计算拉普拉斯矩阵")
    L_matrices, lap_stats = compute_laplacian(Cutoff_weight)
    
    np.save(save_dir / "L_matrices.npy", L_matrices)
    print_success(f"已保存核心文件: L_matrices.npy (shape={L_matrices.shape})")
    
    # 9. 生成报告
    print_step(9, 9, "生成报告")
    execution_time = time.time() - start_time
    
    matrices_info = {
        "Similarity Matrix": {"shape": P.shape, "dtype": str(P.dtype), "size_mb": P.nbytes / 1024**2},
        "Gaussian Matrix": {"shape": A_prime.shape, "dtype": str(A_prime.dtype), "size_mb": A_prime.nbytes / 1024**2},
        "Cutoff Weight": {"shape": Cutoff_weight.shape, "dtype": str(Cutoff_weight.dtype), "size_mb": Cutoff_weight.nbytes / 1024**2},
        "Cutoff 0/1": {"shape": Cutoff_0_1.shape, "dtype": str(Cutoff_0_1.dtype), "size_mb": Cutoff_0_1.nbytes / 1024**2},
        "Laplacian Matrices": {"shape": L_matrices.shape, "dtype": str(L_matrices.dtype), "size_mb": L_matrices.nbytes / 1024**2},
    }
    
    report_text = generate_matrix_report(
        save_dir,
        config,
        matrices_info,
        filter_stats,
        lap_stats,
        subgraph_stats,
        thresholds,
        execution_time
    )
    
    print_report_summary(report_text)
    print_success(f"Step 2 完成! 总耗时: {execution_time:.2f} 秒")
