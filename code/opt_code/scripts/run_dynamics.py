#!/usr/bin/env python3
"""
动力学分析脚本
=============

执行EEG动力学分析。

使用方法
--------
python run_dynamics.py --config config/config.yaml --laplacian data/laplacian_family.npy

参数说明
--------
--config : 配置文件路径
--laplacian : 拉普拉斯矩阵族数据路径
--output : 输出目录（可选）
--name : 实验名称（可选）
--n-scales : 分析尺度数量（可选）
--n-jobs : 并行作业数（可选）
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.config import load_config
from src.dynamics.pipeline import DynamicsAnalysisPipeline
from src.utils.logger import (
    print_header, print_success, print_error, print_panel, console
)
from src.utils.io import load_numpy, Experiment


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="EEG动力学分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/config.yaml",
        help="配置文件路径",
    )
    
    parser.add_argument(
        "--laplacian", "-l",
        type=str,
        required=True,
        help="拉普拉斯矩阵族数据路径 (npy文件)",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output",
        help="输出基础目录",
    )
    
    parser.add_argument(
        "--name", "-n",
        type=str,
        default="dynamics",
        help="实验名称",
    )
    
    parser.add_argument(
        "--n-scales",
        type=int,
        default=None,
        help="分析尺度数量",
    )
    
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="并行作业数（-1表示使用所有核心）",
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建实验
    exp = Experiment(
        name=args.name,
        base_dir=args.output,
        config_path=args.config,
        description="EEG动力学分析"
    )
    
    # 记录运行命令
    exp.log_command(sys.argv)
    exp.log_input("laplacian", args.laplacian)
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 加载拉普拉斯矩阵族
        console.print(f"\n[bold]加载拉普拉斯数据: {args.laplacian}[/bold]")
        laplacian_family = load_numpy(args.laplacian)
        console.print(f"[dim]数据形状: {laplacian_family.shape}[/dim]")
        
        # 创建流水线（使用实验目录）
        pipeline = DynamicsAnalysisPipeline(config, output_dir=str(exp.root_dir))
        
        # 执行分析
        dynamics_data, trajectories = pipeline.run(
            laplacian_family,
            n_scales=args.n_scales,
            n_jobs=args.n_jobs,
        )
        
        # 记录输出
        exp.log_output("dynamics_data", exp.output_dir / "dynamics_data.npy")
        exp.log_output("trajectories", exp.output_dir / "trajectories.npy")
        
        # 记录指标
        exp.log_metric("feature_shape", list(dynamics_data.shape))
        exp.log_metric("trajectory_shape", list(trajectories.shape))
        
        # 完成实验
        exp.finish("completed")
        
    except Exception as e:
        exp.finish("failed")
        print_error(f"分析失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
