# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""Step 3：子图矩阵上几何、TDA、Lorenz 特征提取与保存。"""

import os
import time
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from ..utils.console import (
    console, print_header, print_step, print_success, print_error, print_warning, print_info
)
from ..features import (
    calculate_spectral_features,
    calculate_tda_features_batch,
    calculate_dynamics_features_batch as calculate_lorenz_features_batch
)


def run_feature_extraction_pipeline(config: Dict[str, Any]):
    """按配置运行特征提取。"""
    print_header("Step 3: 特征提取")

    start_time = time.time()

    print_step(1, 3, "准备数据和输出目录")

    data_config = config.get("data", {})
    subgraphs_path = data_config.get("subgraphs_path")
    output_dir = Path(data_config.get("output_dir", "./output/features"))

    if not subgraphs_path or not os.path.exists(subgraphs_path):
        print_error(f"子图文件不存在: {subgraphs_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print_step(2, 3, f"加载子图: {subgraphs_path}")
    try:
        subgraphs = np.load(subgraphs_path, mmap_mode='r')
        K, N, _ = subgraphs.shape
        print_success(f"子图加载成功: {subgraphs.shape}, K={K}, N={N}")
    except Exception as e:
        print_error(f"子图加载失败: {e}")
        return

    tda_config = config.get("tda", {})
    lorenz_config = config.get("lorenz", {})
    processing_config = config.get("processing", {})

    n_workers = int(processing_config.get("num_scale_workers", 4))
    verbose = processing_config.get("verbose", True)

    if tda_config.get("spectral", {}).get("enabled", True):
        print_step(3, 3, "提取特征...")
        print_info("子步骤: 提取几何特征 (Geometric)")
        try:
            spectral_features = calculate_spectral_features(
                subgraphs,
                num_workers=n_workers,
                verbose=verbose
            )
            np.save(output_dir / "spectral_features.npy", spectral_features)
            print_success(f"几何特征已保存: {spectral_features.shape}")
        except Exception as e:
            print_error(f"几何特征提取失败: {e}")
            spectral_features = None
    else:
        spectral_features = None

    if tda_config.get("topological", {}).get("enabled", True):
        print_info("子步骤: 提取拓扑特征 (TDA)")
        try:
            tda_features, bars_list = calculate_tda_features_batch(
                subgraphs,
                config=config,
                num_workers=n_workers,
                verbose=verbose
            )
            np.save(output_dir / "tda_features.npy", tda_features)
            np.save(output_dir / "tda_barcodes.npy", np.array(bars_list, dtype=object))
            print_success(f"TDA 特征已保存: {tda_features.shape}")
        except Exception as e:
            print_error(f"TDA 特征提取失败: {e}")
            tda_features = None
    else:
        tda_features = None

    if lorenz_config.get("enabled", True):
        print_info("子步骤: 提取 Lorenz 动力学特征")
        try:
            traj_dir = output_dir / "trajectories" if config.get("lorenz", {}).get("trajectory_save", {}).get("enabled", False) else None
            if traj_dir:
                traj_dir.mkdir(exist_ok=True)

            lorenz_features, lorenz_trajectories = calculate_lorenz_features_batch(
                subgraphs,
                config=config,
                num_workers=n_workers,
                verbose=verbose,
                trajectory_output_dir=str(traj_dir) if traj_dir else None
            )
            np.save(output_dir / "dynamics_features.npy", lorenz_features)
            print_success(f"Lorenz 特征已保存: {lorenz_features.shape}")
        except Exception as e:
            print_error(f"Lorenz 特征提取失败: {e}")
            lorenz_features = None
    else:
        lorenz_features = None

    print_time = time.time() - start_time
    print_success(f"Step 3 完成! 总耗时: {print_time:.2f} 秒")
