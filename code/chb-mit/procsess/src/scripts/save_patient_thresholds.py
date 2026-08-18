# Author: 王梓涵 <wangzh011031@163.com>
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 Step 6 的配置，为单个患者重新构建多尺度图，
并将每个尺度对应的阈值数组 `thresholds.npy` 保存在
per_patient_results/<patient_id>/ 目录下。

默认针对 chb01，可通过 --patient_id 指定其它患者。
"""

import sys
import os
import argparse
import yaml
from pathlib import Path

import numpy as np

# 将项目根目录加入 sys.path，便于导入 src.*
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.matrix.similarity import compute_similarity_matrix_2d
from src.matrix.graph_builder import (
    filter_negative_correlations,
    apply_gaussian_kernel,
    partition_by_quantile,
    partition_by_uniform,
)
from src.utils.console import print_error, print_info, print_success


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        # 兼容相对路径：从项目根目录再找一次
        alt = PROJECT_ROOT / config_path
        if alt.exists():
            config_path = alt
        else:
            raise FileNotFoundError(f"配置文件未找到: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_thresholds_for_patient(x: np.ndarray, config: dict) -> np.ndarray:
    """
    完全复用 patient_pipeline.build_graphs 中的逻辑，
    但显式返回 thresholds。
    """
    sim_cfg = config.get("similarity", {})
    graph_cfg = config.get("graph", {})

    method = sim_cfg.get("method", "pearson")

    # 计算相似度矩阵 (N, T) -> (N, N)
    P = compute_similarity_matrix_2d(
        x,
        method=method,
        chunk_size=sim_cfg.get("chunk_size", 200),
        rbf_gamma=sim_cfg.get("rbf_gamma", 1.0),
        verbose=False,
    )

    # 过滤负相关
    A, _ = filter_negative_correlations(P)

    # 高斯核映射
    A_prime = apply_gaussian_kernel(
        A,
        exponent=graph_cfg.get("gaussian_exponent", 1),
    )

    # 子图划分
    n_scales = graph_cfg.get("n_scales", 10)
    mode = graph_cfg.get("partition_method", "quantile")

    if mode == "quantile":
        _, thresholds, _ = partition_by_quantile(A_prime, k=n_scales)
    else:
        _, thresholds, _ = partition_by_uniform(A_prime, k=n_scales)

    return thresholds


def main():
    parser = argparse.ArgumentParser(
        description="为指定患者保存多尺度 cutoff 分位数阈值 thresholds.npy"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/per_patient.yaml",
        help="与单患者分析一致的配置文件路径",
    )
    parser.add_argument(
        "--patient_id",
        type=str,
        default="chb01",
        help="患者 ID（例如 chb01）",
    )

    args = parser.parse_args()

    try:
        config = load_config(Path(args.config))
    except Exception as e:
        print_error(f"读取配置失败: {e}")
        return

    data_cfg = config.get("data", {})
    patients_dir = Path(data_cfg.get("patients_data_dir", ""))
    if not patients_dir.exists():
        print_error(f"患者数据目录不存在: {patients_dir}")
        return

    patient_id = args.patient_id
    patient_data_dir = patients_dir / patient_id
    x_path = patient_data_dir / "x_data.npy"

    if not x_path.exists():
        print_error(f"未找到 x_data.npy: {x_path}")
        return

    # 加载 x_data
    print_info(f"加载患者 {patient_id} 的 x_data: {x_path}")
    x = np.load(x_path)

    # 计算阈值
    print_info("根据配置重新构建多尺度图并计算分位数阈值 thresholds...")
    thresholds = compute_thresholds_for_patient(x, config)

    # 输出目录：与单患者分析统一
    out_cfg = config.get("output", {})
    base_out_dir = Path(out_cfg.get("output_dir", "./output/patients"))
    patient_out_dir = base_out_dir / patient_id
    patient_out_dir.mkdir(parents=True, exist_ok=True)

    save_path = patient_out_dir / "thresholds.npy"
    np.save(save_path, thresholds)

    print_success(f"已保存阈值数组 thresholds.npy 到: {save_path}")
    print_info(f"thresholds 形状: {thresholds.shape}")


if __name__ == "__main__":
    main()

