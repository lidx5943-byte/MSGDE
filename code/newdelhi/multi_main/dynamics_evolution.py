#!/usr/bin/env python3
"""
动力学演化脚本
==============

对单个耦合强度进行动力学演化分析。
这个脚本应该为每个耦合强度值单独提交一个SLURM作业。

使用方法
--------
python dynamics_evolution.py --config config.yaml --coupling-strength 0.5

参数说明
--------
--config : 配置文件路径（默认: config.yaml）
--coupling-strength : 耦合强度值（必需）
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import yaml
from typing import Dict, Any
from datetime import datetime
import shutil
import multiprocessing

import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合服务器环境
import matplotlib.pyplot as plt

# 添加参考代码路径到sys.path
sys.path.insert(0, str(Path("/mnt/home/jiangj33/eegcode/opt_code")))

from src.config import load_config
from src.dynamics.pipeline import DynamicsAnalysisPipeline
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


def get_limited_cpu_count(max_usage: float = 0.9) -> int:
    """
    获取限制后的 CPU 核心数
    
    参数
    ----
    max_usage : float, default=0.9
        CPU 最大使用率（0.0-1.0），默认 90%
        
    返回
    ----
    int
        限制后的 CPU 核心数（至少为 1）
    """
    total_cpus = multiprocessing.cpu_count()
    limited_cpus = max(1, int(total_cpus * max_usage))
    return limited_cpus


class ControlledDynamicsPipeline(DynamicsAnalysisPipeline):
    """可控制保存行为的动力学分析管道
    
    通过继承 DynamicsAnalysisPipeline 并重写 _save_results 方法，
    实现根据参数选择性保存文件，避免生成后删除的浪费。
    """
    
    def __init__(self, config=None, output_dir: str = None, 
                 save_trajectories: bool = True,
                 save_dynamics_data: bool = False,
                 save_features: bool = False):
        """
        初始化可控制的动力学分析管道
        
        参数
        ----
        config : Any, optional
            配置对象
        output_dir : str, optional
            输出目录
        save_trajectories : bool, default=True
            是否保存轨迹数据 (trajectories.npy)
        save_dynamics_data : bool, default=False
            是否保存动力学数据 (dynamics_data.npy)
        save_features : bool, default=False
            是否保存特征文件 (features_scale_*.npy)
        """
        super().__init__(config, output_dir)
        self.save_trajectories = save_trajectories
        self.save_dynamics_data = save_dynamics_data
        self.save_features = save_features
    
    def _save_results(self, dynamics_data, trajectories, all_features):
        """重写保存方法，根据配置选择性保存"""
        from pathlib import Path
        data_dir = self.output_dir / "data"
        
        if self.save_dynamics_data:
            save_numpy(dynamics_data, data_dir / "dynamics_data.npy")
        
        if self.save_trajectories:
            save_numpy(trajectories, data_dir / "trajectories.npy")
        
        if self.save_features:
            for i, features in enumerate(all_features):
                save_numpy(features, data_dir / f"features_scale_{i+1:02d}.npy")


def plot_cutoff_trajectories(cutoff_data: np.ndarray, output_path: Path, 
                            coupling_strength: float, cutoff_idx: int,
                            node_idx: int = 0):
    """
    绘制cutoff的蝴蝶轨迹图（Lorenz吸引子，X-Z平面）
    只绘制一张图，使用全部稳态步数
    
    参数
    ----
    cutoff_data : np.ndarray
        轨迹数据，形状为 (n_nodes, n_times, 3)
    output_path : Path
        图像保存路径
    coupling_strength : float
        耦合强度值
    cutoff_idx : int
        cutoff索引（从1开始）
    node_idx : int, default=0
        要绘制的节点索引（默认第0个节点）
    """
    n_nodes, n_times, n_dims = cutoff_data.shape
    
    # 确保节点索引有效
    node_idx = min(node_idx, n_nodes - 1)
    
    # 获取指定节点的全部稳态步数数据
    x = cutoff_data[node_idx, :, 0]  # X维度
    z = cutoff_data[node_idx, :, 2]  # Z维度
    
    # 如果数据点太多，进行下采样以提高绘图速度
    if len(x) > 5000:
        downsample = max(1, len(x) // 5000)
        x_plot = x[::downsample]
        z_plot = z[::downsample]
    else:
        x_plot = x
        z_plot = z
    
    # 创建2D图像（X-Z平面，经典蝴蝶图）
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制蝴蝶轨迹
    ax.plot(x_plot, z_plot, 'b-', linewidth=0.5, alpha=0.8)
    
    # 标记起点和终点
    if len(x_plot) > 0:
        ax.plot(x_plot[0], z_plot[0], 'go', markersize=8, label='Start', zorder=5)
        ax.plot(x_plot[-1], z_plot[-1], 'ro', markersize=8, label='End', zorder=5)
    
    # 设置标签和标题
    ax.set_xlabel('X', fontsize=14, fontweight='bold')
    ax.set_ylabel('Z', fontsize=14, fontweight='bold')
    ax.set_title(
        f'Lorenz Butterfly Attractor - Cutoff {cutoff_idx} (ε={coupling_strength:.1f})',
        fontsize=16,
        fontweight='bold'
    )
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right')
    
    # 添加信息文本
    info_text = f'Node {node_idx}\nSteps: 0-{n_times}\nPoints: {len(x_plot)}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_config_for_dynamics(custom_config: Dict[str, Any], coupling_strength: float, extract_features: bool = False) -> Any:
    """根据自定义配置和耦合强度创建动力学配置对象"""
    # 加载参考配置作为模板
    reference_config_path = "/mnt/gs21/scratch/jiangj33/ldx/case17/multi_main/config.yaml"
    config = load_config(reference_config_path)
    
    # 更新动力学参数
    if 'dynamics' in custom_config:
        dyn_config = custom_config['dynamics']
        
        # 更新振荡器参数
        if 'oscillator' in dyn_config:
            osc = dyn_config['oscillator']
            config.dynamics.oscillator.alpha = osc.get('alpha', config.dynamics.oscillator.alpha)
            config.dynamics.oscillator.beta = osc.get('beta', config.dynamics.oscillator.beta)
            config.dynamics.oscillator.gamma = osc.get('gamma', config.dynamics.oscillator.gamma)
            config.dynamics.oscillator.rk = osc.get('rk', config.dynamics.oscillator.rk)
        
        # 更新数值积分参数
        if 'numerical' in dyn_config:
            num = dyn_config['numerical']
            config.dynamics.numerical.method = num.get('method', config.dynamics.numerical.method)
            config.dynamics.numerical.time_step = num.get('time_step', config.dynamics.numerical.time_step)
            config.dynamics.numerical.coupling_strength = coupling_strength  # 关键：设置耦合强度
            config.dynamics.numerical.coupling_mode = num.get('coupling_mode', config.dynamics.numerical.coupling_mode)
            config.dynamics.numerical.use_periodic_boundary = num.get('use_periodic_boundary', config.dynamics.numerical.use_periodic_boundary)
        
        # 更新分析参数
        if 'analysis' in dyn_config:
            ana = dyn_config['analysis']
            config.dynamics.analysis.time_steps = ana.get('time_steps', config.dynamics.analysis.time_steps)
            config.dynamics.analysis.transient_steps = ana.get('transient_steps', config.dynamics.analysis.transient_steps)
            config.dynamics.analysis.max_nodes = ana.get('max_nodes', config.dynamics.analysis.max_nodes)
        
        # 更新采样参数
        if 'sampling' in dyn_config:
            samp = dyn_config['sampling']
            config.dynamics.analysis.sample_method = samp.get('method', 'hybrid')
            config.dynamics.analysis.hybrid_ratio = samp.get('hybrid_ratio', 0.3)
        
        # 更新特征提取参数
        if 'features' in dyn_config:
            feat = dyn_config['features']
            # 如果extract_features为False，禁用所有特征提取
            if not extract_features:
                config.dynamics.features.enable_chaos = False
                config.dynamics.features.enable_sync = False
            else:
                config.dynamics.features.enable_chaos = feat.get('enable_chaos', True)
                config.dynamics.features.enable_sync = feat.get('enable_sync', True)
        
        # 更新初始值范围
        if 'initial_values' in dyn_config:
            init = dyn_config['initial_values']
            config.dynamics.initial_values.x_min = init.get('x_min', config.dynamics.initial_values.x_min)
            config.dynamics.initial_values.x_max = init.get('x_max', config.dynamics.initial_values.x_max)
            config.dynamics.initial_values.y_min = init.get('y_min', config.dynamics.initial_values.y_min)
            config.dynamics.initial_values.y_max = init.get('y_max', config.dynamics.initial_values.y_max)
            config.dynamics.initial_values.z_min = init.get('z_min', config.dynamics.initial_values.z_min)
            config.dynamics.initial_values.z_max = init.get('z_max', config.dynamics.initial_values.z_max)
        
        # 更新稳定性参数
        if 'stability' in dyn_config:
            stab = dyn_config['stability']
            config.dynamics.stability.state_clip_min = stab.get('state_clip_min', config.dynamics.stability.state_clip_min)
            config.dynamics.stability.state_clip_max = stab.get('state_clip_max', config.dynamics.stability.state_clip_max)
            config.dynamics.stability.dynamics_clip_min = stab.get('dynamics_clip_min', config.dynamics.stability.dynamics_clip_min)
            config.dynamics.stability.dynamics_clip_max = stab.get('dynamics_clip_max', config.dynamics.stability.dynamics_clip_max)
    
    # 更新拉普拉斯参数
    if 'laplacian' in custom_config:
        lap_config = custom_config['laplacian']
        config.laplacian.sigma = lap_config.get('sigma', config.laplacian.sigma)
    
    return config


def check_experiment_completed(output_dir: Path, extract_features: bool) -> bool:
    """
    检查实验是否已完成
    
    参数
    ----
    output_dir : Path
        实验输出目录
    extract_features : bool
        是否提取特征
        
    返回
    ----
    bool
        如果实验已完成返回True，否则返回False
    """
    data_dir = output_dir / "data"
    
    # 检查trajectories.npy是否存在（这是必需的）
    trajectories_file = data_dir / "trajectories.npy"
    if not trajectories_file.exists():
        return False
    
    # 如果extract_features=True，还需要检查dynamics_data.npy
    if extract_features:
        dynamics_data_file = data_dir / "dynamics_data.npy"
        if not dynamics_data_file.exists():
            return False
    
    return True


def run_dynamics_evolution(config_path: str, coupling_strength: float):
    """运行单个耦合强度的动力学演化"""
    console = Console()
    
    console.print()
    console.rule(f"[bold magenta]动力学演化 - 耦合强度: {coupling_strength:.1f}[/bold magenta]")
    console.print()
    
    # 加载自定义配置
    console.print(f"[bold]加载配置文件: {config_path}[/bold]")
    custom_config = load_custom_config(config_path)
    
    # 获取配置参数
    dataset_name = custom_config['paths']['dataset_name']
    laplacian_output_path = custom_config['paths']['laplacian_output_path']
    output_base_path = custom_config['paths']['output_base_path']
    K = custom_config['experiment']['K']  # 拉普拉斯尺度数量
    sigma = custom_config['laplacian'].get('sigma', 3.0) if 'laplacian' in custom_config else 3.0
    
    # 实验控制参数
    extract_features = custom_config['experiment'].get('extract_features', False)
    save_cutoff = custom_config['experiment'].get('save_cutoff', False)
    plot_cutoff = custom_config['experiment'].get('plot_cutoff', True)
    skip_completed = custom_config['experiment'].get('skip_completed', True)
    
    # 计算CPU核心数
    total_cpus = multiprocessing.cpu_count()
    limited_cpus = get_limited_cpu_count(max_usage=0.9)
    n_jobs = limited_cpus
    
    console.print(f"\n[bold]配置信息:[/bold]")
    console.print(f"  数据集名称: {dataset_name}")
    console.print(f"  耦合强度: {coupling_strength:.1f}")
    console.print(f"  拉普拉斯矩阵族路径: {laplacian_output_path}")
    console.print(f"  输出基础路径: {output_base_path}")
    console.print(f"  拉普拉斯尺度数 K: {K}")
    console.print(f"  高斯核参数 σ: {sigma}")
    console.print(f"  CPU 核心数: {total_cpus} (使用: {n_jobs})")
    console.print(f"  提取特征: {extract_features}")
    console.print(f"  保存cutoff文件: {save_cutoff}")
    console.print(f"  绘制cutoff图像: {plot_cutoff}")
    
    # 获取拉普拉斯矩阵族路径
    laplacian_family_path = Path(laplacian_output_path) / f"laplacian_family_K{K}_sigma{sigma:.1f}.npy"
    
    # 检查拉普拉斯矩阵族是否存在
    if not laplacian_family_path.exists():
        console.print(f"\n[bold red]✗[/bold red] 错误: 拉普拉斯矩阵族不存在: {laplacian_family_path}")
        console.print(f"[dim]请先运行 multiscale_filter.py 生成拉普拉斯矩阵族[/dim]")
        sys.exit(1)
    
    # 加载拉普拉斯矩阵族
    console.print(f"\n[bold]加载拉普拉斯矩阵族: {laplacian_family_path.name}[/bold]")
    with Timer("加载拉普拉斯矩阵族"):
        laplacian_family = load_numpy(str(laplacian_family_path))
    console.print(f"[dim]拉普拉斯矩阵族形状: {laplacian_family.shape}[/dim]")
    
    # 创建输出目录
    coupling_str = f"{coupling_strength:.1f}"
    output_dir = Path(output_base_path) / f"{dataset_name}-{coupling_str}"
    ensure_dir(output_dir)
    
    # 检查是否已完成
    if skip_completed and check_experiment_completed(output_dir, extract_features):
        console.print(f"\n[bold yellow]⏭[/bold yellow] 实验已完成，跳过")
        console.print(f"[dim]输出目录: {output_dir}[/dim]")
        return
    
    try:
        # 创建动力学配置（包含当前耦合强度）
        console.print(f"\n[bold]创建动力学配置...[/bold]")
        dynamics_config = create_config_for_dynamics(custom_config, coupling_strength, extract_features)
        
        # 运行动力学分析
        console.print(f"[bold]运行动力学分析 (10个尺度并行演化)...[/bold]")
        dynamics_pipeline = ControlledDynamicsPipeline(
            dynamics_config,
            output_dir=str(output_dir),
            save_trajectories=True,  # 总是保存轨迹数据
            save_dynamics_data=extract_features,  # 根据参数决定
            save_features=extract_features  # 根据参数决定
        )
        
        with Timer(f"动力学分析 (ε={coupling_strength:.1f})"):
            dynamics_data, trajectories = dynamics_pipeline.run(
                laplacian_family,
                n_scales=K,
                n_jobs=n_jobs  # 使用分配的CPU核心数
            )
        
        console.print(f"[dim]轨迹数据形状: {trajectories.shape}[/dim]")
        
        # 保存cutoff文件和绘制图像
        saved_files = []
        plotted_files = []
        
        if save_cutoff or plot_cutoff:
            # 验证轨迹数据形状
            if trajectories.ndim != 4:
                raise ValueError(
                    f"轨迹数据应为4维 (n_scales, n_nodes, n_times, 3)，"
                    f"当前形状: {trajectories.shape}"
                )
            
            n_scales_actual = trajectories.shape[0]
            if n_scales_actual != K:
                console.print(
                    f"  [yellow]⚠[/yellow] 警告: 实际尺度数 {n_scales_actual} 与配置的K值 {K} 不一致"
                )
            
            # 创建figures目录（如果需要绘图）
            if plot_cutoff:
                figures_dir = output_dir / "figures"
                ensure_dir(figures_dir)
            
            # 处理每个尺度的cutoff
            console.print(f"\n[bold]处理cutoff文件和图像...[/bold]")
            for k in range(min(n_scales_actual, K)):
                cutoff_data = trajectories[k]  # 形状: (n_nodes, n_times, 3)
                
                # 保存cutoff文件
                if save_cutoff:
                    cutoff_file = output_dir / f"cutoff_{k+1}.npy"
                    save_numpy(cutoff_data, cutoff_file)
                    saved_files.append(f"cutoff_{k+1}.npy")
                
                # 绘制图像
                if plot_cutoff:
                    plot_file = figures_dir / f"cutoff_{k+1}_coupling_{coupling_str}.png"
                    plot_cutoff_trajectories(
                        cutoff_data, 
                        plot_file,
                        coupling_strength,
                        k + 1,
                        node_idx=0
                    )
                    plotted_files.append(plot_file.name)
        
        # 删除DynamicsAnalysisPipeline自动创建的logs目录（如果存在）
        logs_dir = output_dir / "logs"
        if logs_dir.exists():
            shutil.rmtree(logs_dir)
        
        console.print()
        console.print(f"[bold green]✓[/bold green] 动力学演化完成！")
        console.print(f"[dim]输出目录: {output_dir}[/dim]")
        if saved_files:
            console.print(f"[dim]已保存 {len(saved_files)} 个cutoff文件[/dim]")
        if plotted_files:
            console.print(f"[dim]已绘制 {len(plotted_files)} 个cutoff图像[/dim]")
        
    except Exception as e:
        import traceback
        console.print(f"\n[bold red]✗[/bold red] 动力学演化失败: {e}")
        traceback.print_exc()
        sys.exit(1)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="动力学演化 - 单个耦合强度分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="配置文件路径（默认: config.yaml）",
    )
    
    parser.add_argument(
        "--coupling-strength", "-e",
        type=float,
        required=True,
        help="耦合强度值（必需）",
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    try:
        run_dynamics_evolution(args.config, args.coupling_strength)
    except Exception as e:
        console = Console()
        console.print(f"[bold red]✗[/bold red] 动力学演化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
