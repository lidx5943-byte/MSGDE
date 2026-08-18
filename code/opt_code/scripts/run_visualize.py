#!/usr/bin/env python3
"""
可视化脚本
=========

生成轨迹和特征差异性的可视化图像。

使用方法
--------
python run_visualize.py --config config/config.yaml --trajectories data/trajectories.npy

参数说明
--------
--config : 配置文件路径
--trajectories : 轨迹数据路径
--output : 输出目录（可选）
--node : 节点索引（可选）
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.config import load_config
from src.visualization.trajectories import (
    plot_phase_space,
    plot_trajectory_evolution,
    plot_butterfly_attractor,
    plot_3d_trajectory,
)
from src.visualization.diversity import analyze_all_scales
from src.utils.logger import (
    print_header, print_success, print_error, print_panel, console
)
from src.utils.io import load_numpy, ensure_dir


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="轨迹可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/config.yaml",
        help="配置文件路径",
    )
    
    parser.add_argument(
        "--trajectories", "-t",
        type=str,
        required=True,
        help="轨迹数据路径 (npy文件)",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出目录",
    )
    
    parser.add_argument(
        "--node", "-n",
        type=int,
        default=0,
        help="可视化的节点索引",
    )
    
    parser.add_argument(
        "--scale", "-s",
        type=int,
        default=0,
        help="可视化的尺度索引",
    )
    
    parser.add_argument(
        "--butterfly",
        action="store_true",
        help="生成蝴蝶图序列",
    )
    
    parser.add_argument(
        "--diversity",
        action="store_true",
        help="生成特征差异性分析",
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="生成所有可视化",
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print_panel(
        "轨迹可视化\n\n"
        f"配置文件: {args.config}\n"
        f"轨迹数据: {args.trajectories}",
        title="可视化脚本",
        style="blue"
    )
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 加载轨迹数据
        console.print(f"\n[bold]加载轨迹数据: {args.trajectories}[/bold]")
        trajectories = load_numpy(args.trajectories)
        console.print(f"[dim]数据形状: {trajectories.shape}[/dim]")
        
        # 确定输出目录
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = Path(args.trajectories).parent / "figures"
        ensure_dir(output_dir)
        
        # 确定是否生成所有
        do_all = args.all or (not args.butterfly and not args.diversity)
        
        # 处理数据形状
        if trajectories.ndim == 4:
            # (n_scales, n_nodes, n_times, 3)
            scale_traj = trajectories[args.scale]
            console.print(f"[dim]选择尺度 {args.scale + 1}，形状: {scale_traj.shape}[/dim]")
        else:
            # (n_nodes, n_times, 3)
            scale_traj = trajectories
        
        node_idx = args.node
        
        # 1. 相空间图
        if do_all or not args.butterfly:
            print_header("生成相空间图")
            plot_phase_space(
                scale_traj,
                node_idx=node_idx,
                title=f"相空间 (节点 {node_idx})",
                save_path=str(output_dir / f"phase_space_node{node_idx}.png"),
            )
        
        # 2. 轨迹演化图
        if do_all or not args.butterfly:
            print_header("生成轨迹演化图")
            plot_trajectory_evolution(
                scale_traj,
                node_idx=node_idx,
                title=f"轨迹演化 (节点 {node_idx})",
                save_path=str(output_dir / f"trajectory_evolution_node{node_idx}.png"),
            )
        
        # 3. 3D轨迹图
        if do_all or not args.butterfly:
            print_header("生成3D轨迹图")
            plot_3d_trajectory(
                scale_traj,
                node_idx=node_idx,
                title=f"3D轨迹 (节点 {node_idx})",
                save_path=str(output_dir / f"trajectory_3d_node{node_idx}.png"),
            )
        
        # 4. 蝴蝶图序列
        if args.butterfly or do_all:
            print_header("生成蝴蝶图序列")
            viz_config = config.visualization.trajectory
            plot_butterfly_attractor(
                scale_traj,
                figures_dir=str(output_dir),
                node_idx=node_idx,
                step_interval=viz_config.uniform_interval,
                window_size=viz_config.window_size,
                interval_mode=viz_config.interval_mode,
            )
        
        # 5. 特征差异性分析
        if args.diversity or do_all:
            print_header("特征差异性分析")
            div_config = config.visualization.feature_diversity
            
            if trajectories.ndim == 4:
                analyze_all_scales(
                    trajectories,
                    figures_dir=str(output_dir),
                    data_dir=str(output_dir / "data"),
                    step_interval=div_config.step_interval,
                    method=div_config.method,
                )
            else:
                from src.visualization.diversity import analyze_feature_diversity
                analyze_feature_diversity(
                    scale_traj,
                    figures_dir=str(output_dir),
                    data_dir=str(output_dir / "data"),
                    step_interval=div_config.step_interval,
                    method=div_config.method,
                )
        
        print_success(f"可视化完成！结果已保存到: {output_dir}")
        
    except Exception as e:
        print_error(f"可视化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

