#!/usr/bin/env python3
"""
多尺度过滤脚本
==============

生成拉普拉斯矩阵族并保存到本地npy文件。
这个脚本只需要运行一次，生成的拉普拉斯矩阵族会被所有动力学演化任务共享。

使用方法
--------
python multiscale_filter.py --config config.yaml

参数说明
--------
--config : 配置文件路径（默认: config.yaml）
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import yaml
from typing import Dict, Any
from datetime import datetime

# 添加参考代码路径到sys.path
sys.path.insert(0, str(Path("/mnt/home/jiangj33/eegcode/opt_code")))

from src.config import load_config
from src.laplacian.multiscale import LaplacianPipeline
from src.utils.io import load_numpy, save_numpy, ensure_dir
from src.utils.timer import Timer
from src.utils.logger import get_logger
from rich.console import Console
import logging

logger = get_logger(__name__)


def load_custom_config(config_path: str) -> Dict[str, Any]:
    """加载自定义配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def get_laplacian_family_path(laplacian_output_path: str, K: int, sigma: float) -> Path:
    """
    获取拉普拉斯矩阵族的保存路径
    
    参数
    ----
    laplacian_output_path : str
        拉普拉斯矩阵族保存路径
    K : int
        拉普拉斯尺度数量
    sigma : float
        高斯核参数
        
    返回
    ----
    Path
        拉普拉斯矩阵族的保存路径
    """
    output_path = Path(laplacian_output_path)
    ensure_dir(output_path)
    return output_path / f"laplacian_family_K{K}_sigma{sigma:.1f}.npy"


def run_multiscale_filter(config_path: str = "config.yaml"):
    """运行多尺度过滤，生成拉普拉斯矩阵族"""
    console = Console()
    
    console.print()
    console.rule("[bold magenta]多尺度过滤 - 拉普拉斯矩阵族生成[/bold magenta]")
    console.print()
    
    # 加载自定义配置
    console.print(f"[bold]加载配置文件: {config_path}[/bold]")
    custom_config = load_custom_config(config_path)
    
    # 获取配置参数
    correlation_matrix_path = custom_config['paths']['correlation_matrix_path']
    laplacian_output_path = custom_config['paths']['laplacian_output_path']
    K = custom_config['experiment']['K']  # 拉普拉斯尺度数量
    sigma = custom_config['laplacian'].get('sigma', 3.0) if 'laplacian' in custom_config else 3.0
    
    console.print(f"\n[bold]配置信息:[/bold]")
    console.print(f"  相关系数矩阵: {correlation_matrix_path}")
    console.print(f"  拉普拉斯矩阵族保存路径: {laplacian_output_path}")
    console.print(f"  拉普拉斯尺度数 K: {K}")
    console.print(f"  高斯核参数 σ: {sigma}")
    
    # 检查相关系数矩阵是否存在
    if not Path(correlation_matrix_path).exists():
        console.print(f"[bold red]✗[/bold red] 错误: 相关系数矩阵文件不存在: {correlation_matrix_path}")
        sys.exit(1)
    
    # 加载相关系数矩阵
    console.print(f"\n[bold]加载相关系数矩阵: {correlation_matrix_path}[/bold]")
    with Timer("加载相关系数矩阵"):
        correlation_matrix = load_numpy(correlation_matrix_path)
    console.print(f"[dim]矩阵形状: {correlation_matrix.shape}[/dim]")
    
    # 验证矩阵
    if correlation_matrix.ndim != 2:
        console.print(f"[bold red]✗[/bold red] 错误: 相关系数矩阵应为2维，当前为 {correlation_matrix.ndim} 维")
        sys.exit(1)
    if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
        console.print(f"[bold red]✗[/bold red] 错误: 相关系数矩阵应为方阵，当前形状: {correlation_matrix.shape}")
        sys.exit(1)
    
    # 获取拉普拉斯矩阵族保存路径
    laplacian_family_path = get_laplacian_family_path(laplacian_output_path, K, sigma)
    
    # 检查是否已存在
    if laplacian_family_path.exists():
        console.print(f"\n[bold yellow]⚠[/bold yellow] 拉普拉斯矩阵族已存在: {laplacian_family_path.name}")
        console.print(f"[dim]如需重新生成，请先删除该文件[/dim]")
        
        # 验证文件
        try:
            laplacian_family = load_numpy(str(laplacian_family_path))
            console.print(f"[bold green]✓[/bold green] 文件验证成功，形状: {laplacian_family.shape}")
            console.print(f"[bold green]✓[/bold green] 拉普拉斯矩阵族已就绪，可直接用于动力学演化")
            return
        except Exception as e:
            console.print(f"[bold red]✗[/bold red] 文件验证失败: {e}")
            console.print(f"[dim]将重新生成...[/dim]")
    
    # 创建拉普拉斯配置
    reference_config_path = "/mnt/gs21/scratch/jiangj33/ldx/case17/multi_main/config.yaml"
    laplacian_config = load_config(reference_config_path)
    laplacian_config.laplacian.sigma = sigma
    laplacian_config.laplacian.n_scales = K
    
    # 生成拉普拉斯矩阵族
    console.print(f"\n[bold]生成拉普拉斯矩阵族 (K={K}, σ={sigma})[/bold]")
    with Timer("拉普拉斯矩阵族生成"):
        laplacian_pipeline = LaplacianPipeline(laplacian_config)
        laplacian_family = laplacian_pipeline.run(
            correlation_matrix,
            save_results=False  # 不保存中间结果
        )
    
    console.print(f"[dim]拉普拉斯矩阵族形状: {laplacian_family.shape}[/dim]")
    
    # 保存拉普拉斯矩阵族
    console.print(f"\n[bold]保存拉普拉斯矩阵族到: {laplacian_family_path}[/bold]")
    ensure_dir(laplacian_family_path.parent)
    save_numpy(laplacian_family, str(laplacian_family_path))
    
    console.print()
    console.print(f"[bold green]✓[/bold green] 多尺度过滤完成！")
    console.print(f"[dim]拉普拉斯矩阵族已保存: {laplacian_family_path}[/dim]")
    console.print(f"[dim]现在可以运行动力学演化脚本了[/dim]")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="多尺度过滤 - 生成拉普拉斯矩阵族",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="配置文件路径（默认: config.yaml）",
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    try:
        run_multiscale_filter(args.config)
    except Exception as e:
        console = Console()
        console.print(f"[bold red]✗[/bold red] 多尺度过滤失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
