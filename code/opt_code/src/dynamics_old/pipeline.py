"""
动力学分析流水线模块
====================

提供完整的动力学分析流程。

使用示例
--------
>>> from src.dynamics.pipeline import DynamicsAnalysisPipeline
>>> 
>>> pipeline = DynamicsAnalysisPipeline(config)
>>> features, trajectories = pipeline.run(laplacian_family)
"""

import warnings
import time
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from datetime import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from .oscillators import LorenzOscillator, generate_initial_conditions
from .solvers import create_solver, SolverConfig
from .features import extract_trajectory_features

from ..utils.logger import (
    get_logger, console, create_progress,
    print_header, print_step, print_success, print_warning, print_error,
    print_table, print_panel, format_time
)
from ..utils.timer import Timer, TimerGroup
from ..utils.io import save_numpy, ensure_dir, get_output_dir

from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn,
    MofNCompleteColumn
)
from rich.live import Live
from rich.table import Table
from rich.console import Group
from rich.panel import Panel


logger = get_logger(__name__)


def _hybrid_sampling(nodes, n_samples, ratio=0.3):
    """
    混合筛选方法：结合随机采样和均匀采样（与老版 dynamics_analysis.py 保持一致）
    
    参数
    ----
    nodes : np.ndarray
        节点索引数组
    n_samples : int
        需要采样的数量
    ratio : float
        随机采样的比例 (0-1)
        
    返回
    ----
    np.ndarray
        选中的节点索引
    """
    n_total = len(nodes)
    if n_samples >= n_total:
        return nodes
    
    n_random = int(n_samples * ratio)
    n_uniform = n_samples - n_random
    
    # 随机采样
    random_indices = np.random.choice(n_total, n_random, replace=False)
    random_nodes = nodes[random_indices]
    
    # 均匀采样
    uniform_indices = np.linspace(0, n_total-1, n_uniform, dtype=int)
    uniform_nodes = nodes[uniform_indices]
    
    # 合并并去重
    selected_nodes = np.unique(np.concatenate([random_nodes, uniform_nodes]))
    
    # 如果去重后数量不足，补充随机采样
    if len(selected_nodes) < n_samples:
        remaining_indices = np.setdiff1d(np.arange(n_total), 
                                       np.where(np.isin(nodes, selected_nodes))[0])
        if len(remaining_indices) > 0:
            additional_needed = n_samples - len(selected_nodes)
            additional_indices = np.random.choice(remaining_indices, 
                                               min(additional_needed, len(remaining_indices)), 
                                               replace=False)
            selected_nodes = np.concatenate([selected_nodes, nodes[additional_indices]])
    
    return selected_nodes[:n_samples]


def _evolve_single_scale(args) -> Tuple[int, np.ndarray, Dict[str, Any], bool]:
    """
    对单个尺度进行动力学演化（用于并行处理）
    
    参数
    ----
    args : tuple
        (scale_idx, L, config_dict)
        
    返回
    ----
    Tuple[int, np.ndarray, Dict, bool]
        (尺度索引, 轨迹数据, 统计信息, 成功标志)
    """
    scale_idx, L, config_dict = args
    
    # 抑制警告
    warnings.filterwarnings("ignore")
    
    stats = {
        'n_nodes': 0,
        'n_steps': 0,
        'x_range': (0, 0),
        'z_range': (0, 0),
    }
    
    try:
        # 重建配置
        solver_config = SolverConfig(
            time_step=config_dict['time_step'],
            coupling_strength=config_dict['coupling_strength'],
            coupling_mode=config_dict['coupling_mode'],
            use_periodic_boundary=config_dict['use_periodic_boundary'],
            rk=config_dict['rk'],
            state_clip_min=config_dict['state_clip_min'],
            state_clip_max=config_dict['state_clip_max'],
            dynamics_clip_min=config_dict['dynamics_clip_min'],
            dynamics_clip_max=config_dict['dynamics_clip_max'],
        )
        
        # 创建振荡器
        oscillator = LorenzOscillator(
            alpha=config_dict['alpha'],
            beta=config_dict['beta'],
            gamma=config_dict['gamma'],
            rk=config_dict['rk'],
        )
        
        # 采样节点（与老版 sample_nodes() 保持一致）
        # 注意：只有当节点数超过预设最大值时才启用采样，一般情况下不启用
        n_nodes = L.shape[0]
        max_nodes = config_dict['max_nodes']
        
        # 只有当节点数超过最大值时才进行采样（与老版逻辑一致）
        if n_nodes > max_nodes:
            # 使用配置中的采样方法
            sample_method = config_dict.get('sample_method', 'hybrid')
            hybrid_ratio = config_dict.get('hybrid_ratio', 0.3)
            
            if sample_method == "random":
                # 纯随机采样
                selected_indices = np.random.choice(n_nodes, max_nodes, replace=False)
            elif sample_method == "uniform":
                # 均匀采样
                selected_indices = np.linspace(0, n_nodes-1, max_nodes, dtype=int)
            elif sample_method == "hybrid":
                # 混合采样（使用与老版完全一致的逻辑）
                all_indices = np.arange(n_nodes)
                selected_indices = _hybrid_sampling(all_indices, max_nodes, hybrid_ratio)
            else:
                raise ValueError(f"不支持的采样方法: {sample_method}")
            
            # 提取对应的拉普拉斯矩阵子集
            L = L[np.ix_(selected_indices, selected_indices)]
            n_nodes = L.shape[0]
        # 如果 n_nodes <= max_nodes，则不采样，直接使用全部节点（与老版一致）
        
        stats['n_nodes'] = n_nodes
        
        # 生成初始条件
        initial_state = generate_initial_conditions(
            n_nodes,
            x_range=(config_dict['x_min'], config_dict['x_max']),
            y_range=(config_dict['y_min'], config_dict['y_max']),
            z_range=(config_dict['z_min'], config_dict['z_max']),
        )
        
        # 创建求解器
        from .solvers import EulerSolver, RK4Solver
        if config_dict['method'].lower() == 'euler':
            solver = EulerSolver(solver_config)
        else:
            solver = RK4Solver(solver_config)
        
        # 求解
        n_steps = config_dict['time_steps']
        trajectories = solver.solve(L, oscillator, initial_state, n_steps)
        
        # 去除暂态
        transient = config_dict['transient_steps']
        trajectories = trajectories[:, transient:, :]
        
        stats['n_steps'] = trajectories.shape[1]
        stats['x_range'] = (float(np.min(trajectories[:, :, 0])), float(np.max(trajectories[:, :, 0])))
        stats['z_range'] = (float(np.min(trajectories[:, :, 2])), float(np.max(trajectories[:, :, 2])))
        
        return (scale_idx, trajectories, stats, True)
        
    except Exception as e:
        n_nodes = L.shape[0]
        n_times = config_dict['time_steps'] - config_dict['transient_steps']
        trajectories = np.zeros((n_nodes, n_times, 3))
        stats['error'] = str(e)
        return (scale_idx, trajectories, stats, False)


class DynamicsAnalysisPipeline:
    """
    动力学分析流水线
    
    执行完整的动力学分析流程：
    1. 准备振荡器网络
    2. 多尺度动力学演化
    3. 特征提取
    4. 保存结果
    
    属性
    ----
    config : Config
        配置对象
    output_dir : Path
        输出目录
    """
    
    def __init__(self, config=None, output_dir: str = None):
        """
        初始化流水线
        
        参数
        ----
        config : Config, optional
            配置对象
        output_dir : str, optional
            输出目录，如果为None则自动生成
        """
        self.config = config
        self.logger = get_logger("dynamics")
        
        # 设置输出目录
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        elif config is not None:
            base_dir = config.output.base_dir
            if config.output.use_timestamp:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.output_dir = Path(base_dir) / f"dynamics_{timestamp}"
            else:
                self.output_dir = Path(base_dir) / "dynamics"
        else:
            self.output_dir = get_output_dir("dynamics")
        
        ensure_dir(self.output_dir)
        ensure_dir(self.output_dir / "data")
        ensure_dir(self.output_dir / "figures")
        ensure_dir(self.output_dir / "logs")
        ensure_dir(self.output_dir / "reports")
    
    def run(
        self,
        laplacian_family: np.ndarray,
        n_scales: int = None,
        n_jobs: int = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行动力学分析
        
        参数
        ----
        laplacian_family : np.ndarray
            拉普拉斯矩阵族，形状为 (n_scales, n, n)
        n_scales : int, optional
            分析的尺度数量
        n_jobs : int, optional
            并行作业数
            
        返回
        ----
        Tuple[np.ndarray, np.ndarray]
            (特征矩阵, 轨迹数据)
        """
        print_header("EEG动力学分析")
        
        timers = TimerGroup("动力学分析")
        timers.start_total()
        
        # 获取配置参数
        config_dict = self._get_config_dict()
        
        # 确定尺度数量
        if n_scales is None:
            n_scales = min(config_dict['n_scales'], laplacian_family.shape[0])
        else:
            n_scales = min(n_scales, laplacian_family.shape[0])
        
        # 确定并行数
        if n_jobs is None:
            n_jobs = min(multiprocessing.cpu_count() - 1, n_scales, 8)
        if n_jobs < 1:
            n_jobs = 1
        
        # 打印参数
        self._print_config(config_dict, n_scales, n_jobs)
        
        # =============================================
        # 阶段1：多尺度动力学演化
        # =============================================
        console.print()
        print_step(1, 2, "多尺度动力学演化")
        
        with timers.timer("动力学演化", verbose=False):
            all_trajectories, scale_stats = self._run_evolution(
                laplacian_family, n_scales, n_jobs, config_dict
            )
        
        # 打印演化结果统计
        self._print_evolution_summary(scale_stats, n_scales)
        
        # =============================================
        # 阶段2：特征提取
        # =============================================
        console.print()
        print_step(2, 2, "轨迹特征提取")
        
        with timers.timer("特征提取", verbose=False):
            all_features = self._run_feature_extraction(all_trajectories, n_scales, n_jobs)
        
        # 合并结果
        with timers.timer("结果合并", verbose=False):
            dynamics_data = np.concatenate(all_features, axis=1)
            trajectories_array = np.stack(all_trajectories)
        
        # 保存结果
        with timers.timer("保存结果", verbose=False):
            self._save_results(dynamics_data, trajectories_array, all_features)
        
        timers.stop_total()
        
        # 打印最终统计
        console.print()
        self._print_final_summary(dynamics_data, trajectories_array, n_scales, scale_stats)
        timers.report()
        
        # 生成报告
        self._generate_report(dynamics_data, trajectories_array, config_dict, scale_stats, timers)
        
        print_panel(
            f"分析完成！\n\n"
            f"📁 输出目录: {self.output_dir}\n"
            f"📊 特征矩阵: {dynamics_data.shape}\n"
            f"🌀 轨迹数据: {trajectories_array.shape}",
            title="动力学分析完成",
            style="bold green"
        )
        
        return dynamics_data, trajectories_array
    
    def _run_evolution(
        self,
        laplacian_family: np.ndarray,
        n_scales: int,
        n_jobs: int,
        config_dict: Dict,
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        运行多尺度动力学演化
        """
        # 估算内存需求
        n_nodes = min(laplacian_family.shape[1], config_dict['max_nodes'])
        n_steps = config_dict['time_steps'] - config_dict['transient_steps']
        mem_per_scale_gb = n_nodes * n_steps * 3 * 8 / (1024**3)  # float64 = 8 bytes
        
        # 如果内存需求过大，强制使用顺序处理
        if mem_per_scale_gb > 2.0:
            console.print(
                f"[yellow]⚠ 每尺度内存需求约 {mem_per_scale_gb:.1f}GB，"
                f"自动切换为顺序处理模式以避免内存溢出[/yellow]"
            )
            n_jobs = 1
        
        if n_jobs == 1:
            return self._run_evolution_sequential(laplacian_family, n_scales, config_dict)
        else:
            return self._run_evolution_parallel(laplacian_family, n_scales, n_jobs, config_dict)
    
    def _run_evolution_sequential(
        self,
        laplacian_family: np.ndarray,
        n_scales: int,
        config_dict: Dict,
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        顺序运行多尺度动力学演化（内存友好）
        """
        console.print(f"\n[bold cyan]🔄 顺序演化 {n_scales} 个尺度[/bold cyan]\n")
        
        all_trajectories = []
        scale_stats = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            expand=False,
        ) as progress:
            
            total_task = progress.add_task(
                f"[cyan]动力学演化",
                total=n_scales
            )
            
            for scale_idx in range(n_scales):
                task = (scale_idx, laplacian_family[scale_idx], config_dict)
                
                try:
                    idx, trajectories, stats, success = _evolve_single_scale(task)
                    all_trajectories.append(trajectories)
                    stats['success'] = success
                    scale_stats.append(stats)
                    
                    progress.update(total_task, advance=1)
                    
                    if success:
                        console.print(
                            f"  [green]✓[/green] 尺度 {idx + 1}/{n_scales} 完成 | "
                            f"节点: {stats['n_nodes']} | "
                            f"步数: {stats['n_steps']} | "
                            f"X∈[{stats['x_range'][0]:.1f}, {stats['x_range'][1]:.1f}] | "
                            f"Z∈[{stats['z_range'][0]:.1f}, {stats['z_range'][1]:.1f}]"
                        )
                    else:
                        console.print(
                            f"  [red]✗[/red] 尺度 {idx + 1}/{n_scales} 失败 | "
                            f"错误: {stats.get('error', '未知')}"
                        )
                except Exception as e:
                    console.print(f"  [red]✗[/red] 尺度 {scale_idx + 1} 异常: {e}")
                    n_nodes = laplacian_family.shape[1]
                    n_times = config_dict['time_steps'] - config_dict['transient_steps']
                    all_trajectories.append(np.zeros((min(n_nodes, config_dict['max_nodes']), n_times, 3)))
                    scale_stats.append({'success': False, 'error': str(e)})
        
        return all_trajectories, scale_stats
    
    def _run_evolution_parallel(
        self,
        laplacian_family: np.ndarray,
        n_scales: int,
        n_jobs: int,
        config_dict: Dict,
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        并行运行多尺度动力学演化
        """
        console.print(f"\n[bold cyan]🔄 并行演化 {n_scales} 个尺度 (进程数: {n_jobs})[/bold cyan]\n")
        
        # 准备任务
        tasks = [
            (scale_idx, laplacian_family[scale_idx], config_dict)
            for scale_idx in range(n_scales)
        ]
        
        all_trajectories = [None] * n_scales
        scale_stats = [None] * n_scales
        
        # 创建进度显示
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            expand=False,
        ) as progress:
            
            # 总进度
            total_task = progress.add_task(
                f"[cyan]动力学演化",
                total=n_scales
            )
            
            # 使用进程池并行执行
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = {
                    executor.submit(_evolve_single_scale, task): task[0]
                    for task in tasks
                }
                
                for future in as_completed(futures):
                    scale_idx = futures[future]
                    try:
                        idx, trajectories, stats, success = future.result()
                        all_trajectories[idx] = trajectories
                        scale_stats[idx] = stats
                        scale_stats[idx]['success'] = success
                        
                        # 更新进度
                        progress.update(total_task, advance=1)
                        
                        # 显示尺度完成信息
                        if success:
                            console.print(
                                f"  [green]✓[/green] 尺度 {idx + 1}/{n_scales} 完成 | "
                                f"节点: {stats['n_nodes']} | "
                                f"步数: {stats['n_steps']} | "
                                f"X∈[{stats['x_range'][0]:.1f}, {stats['x_range'][1]:.1f}] | "
                                f"Z∈[{stats['z_range'][0]:.1f}, {stats['z_range'][1]:.1f}]"
                            )
                        else:
                            console.print(
                                f"  [red]✗[/red] 尺度 {idx + 1}/{n_scales} 失败 | "
                                f"错误: {stats.get('error', '未知')}"
                            )
                    except Exception as e:
                        console.print(f"  [red]✗[/red] 尺度 {scale_idx + 1} 异常: {e}")
                        all_trajectories[scale_idx] = np.zeros((1, 1, 3))
                        scale_stats[scale_idx] = {'success': False, 'error': str(e)}
        
        return all_trajectories, scale_stats
    
    def _run_feature_extraction(
        self,
        all_trajectories: List[np.ndarray],
        n_scales: int,
        n_jobs: int = None,
    ) -> List[np.ndarray]:
        """
        从轨迹中提取特征（优化版本：使用向量化操作）
        """
        console.print(f"\n[bold cyan]📊 提取 {n_scales} 个尺度的轨迹特征（优化版本）[/bold cyan]\n")
        
        # 从配置中获取特征提取参数
        enable_chaos = True
        enable_sync = True
        if self.config is not None and hasattr(self.config.dynamics, 'features'):
            enable_chaos = self.config.dynamics.features.enable_chaos
            enable_sync = self.config.dynamics.features.enable_sync
        
        # 确定并行数
        if n_jobs is None:
            if self.config is not None and hasattr(self.config, 'parallel'):
                n_jobs = self.config.parallel.n_jobs
                if n_jobs == -1:
                    n_jobs = multiprocessing.cpu_count()
            else:
                n_jobs = 1
        
        # 计算特征数量
        n_features = 12
        if enable_chaos:
            n_features += 1
        if enable_sync:
            n_features += 1
        
        # 显示特征提取配置
        feature_info = []
        feature_info.append("基础时域统计特征: 12")
        if enable_chaos:
            feature_info.append("混沌特征: 1")
        if enable_sync:
            feature_info.append("同步特征: 1")
        console.print(f"[dim]特征配置: {', '.join(feature_info)} (共 {n_features} 个特征)[/dim]")
        if enable_chaos and n_jobs > 1:
            console.print(f"[dim]并行计算: {n_jobs} 个进程[/dim]\n")
        else:
            console.print()
        
        all_features = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TextColumn("•"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            expand=False,
        ) as progress:
            
            # 总进度（按尺度计算）
            total_task = progress.add_task(
                f"[cyan]总进度",
                total=n_scales
            )
            
            for scale_idx in range(n_scales):
                trajectories = all_trajectories[scale_idx]
                n_nodes = trajectories.shape[0]
                n_times = trajectories.shape[1]
                
                # 当前尺度的进度任务
                scale_task = progress.add_task(
                    f"[yellow]尺度 {scale_idx + 1}/{n_scales}",
                    total=1
                )
                
                # 使用优化后的向量化函数提取特征
                features = extract_trajectory_features(
                    trajectories,
                    center_node_idx=0,
                    verbose=False,
                    enable_chaos=enable_chaos,
                    enable_sync=enable_sync,
                    n_jobs=n_jobs if enable_chaos else 1,
                )
                
                all_features.append(features)
                
                # 更新进度
                progress.update(scale_task, advance=1)
                progress.update(total_task, advance=1)
                
                # 移除当前尺度的任务
                progress.remove_task(scale_task)
                
                # 显示尺度完成信息
                console.print(
                    f"  [green]✓[/green] 尺度 {scale_idx + 1}/{n_scales} 完成 | "
                    f"节点: {n_nodes} | "
                    f"时间步: {n_times} | "
                    f"特征: {features.shape} | "
                    f"均值: {np.mean(features):.4f} | "
                    f"标准差: {np.std(features):.4f}"
                )
        
        return all_features
    
    def _print_evolution_summary(self, scale_stats: List[Dict], n_scales: int):
        """打印演化结果摘要"""
        success_count = sum(1 for s in scale_stats if s and s.get('success', False))
        
        console.print()
        if success_count == n_scales:
            print_success(f"所有 {n_scales} 个尺度演化成功")
        else:
            print_warning(f"{success_count}/{n_scales} 个尺度演化成功")
    
    def _print_final_summary(
        self,
        dynamics_data: np.ndarray,
        trajectories: np.ndarray,
        n_scales: int,
        scale_stats: List[Dict],
    ):
        """打印最终统计摘要"""
        success_count = sum(1 for s in scale_stats if s and s.get('success', False))
        
        rows = [
            ["尺度数量", f"{success_count}/{n_scales} 成功"],
            ["特征矩阵形状", str(dynamics_data.shape)],
            ["轨迹数据形状", str(trajectories.shape)],
            ["特征维度", f"{dynamics_data.shape[1]} (每尺度 14 特征 × {n_scales} 尺度)"],
            ["特征均值", f"{np.mean(dynamics_data):.6f}"],
            ["特征标准差", f"{np.std(dynamics_data):.6f}"],
            ["特征范围", f"[{np.min(dynamics_data):.4f}, {np.max(dynamics_data):.4f}]"],
        ]
        print_table("分析结果统计", ["项目", "值"], rows)
    
    def _get_config_dict(self) -> Dict[str, Any]:
        """获取配置字典"""
        if self.config is not None:
            dynamics = self.config.dynamics
            return {
                # 振荡器参数
                'alpha': dynamics.oscillator.alpha,
                'beta': dynamics.oscillator.beta,
                'gamma': dynamics.oscillator.gamma,
                'rk': dynamics.oscillator.rk,
                # 数值参数
                'method': dynamics.numerical.method,
                'time_step': dynamics.numerical.time_step,
                'coupling_strength': dynamics.numerical.coupling_strength,
                'coupling_mode': dynamics.numerical.coupling_mode,
                'use_periodic_boundary': dynamics.numerical.use_periodic_boundary,
                # 分析参数
                'n_scales': dynamics.analysis.n_scales,
                'time_steps': dynamics.analysis.time_steps,
                'transient_steps': dynamics.analysis.transient_steps,
                'max_nodes': dynamics.analysis.max_nodes,
                # 采样参数（与老版保持一致）
                'sample_method': getattr(dynamics.analysis, 'sample_method', 'hybrid'),
                'hybrid_ratio': getattr(dynamics.analysis, 'hybrid_ratio', 0.3),
                # 初始值
                'x_min': dynamics.initial_values.x_min,
                'x_max': dynamics.initial_values.x_max,
                'y_min': dynamics.initial_values.y_min,
                'y_max': dynamics.initial_values.y_max,
                'z_min': dynamics.initial_values.z_min,
                'z_max': dynamics.initial_values.z_max,
                # 稳定性
                'state_clip_min': dynamics.stability.state_clip_min,
                'state_clip_max': dynamics.stability.state_clip_max,
                'dynamics_clip_min': dynamics.stability.dynamics_clip_min,
                'dynamics_clip_max': dynamics.stability.dynamics_clip_max,
            }
        else:
            # 默认配置
            return {
                'alpha': 10.0,
                'beta': 2.666666,
                'gamma': 60.0,
                'rk': 7.0,
                'method': 'Euler',
                'time_step': 1e-3,
                'coupling_strength': 0.42,
                'coupling_mode': 'x_only',
                'use_periodic_boundary': True,
                'n_scales': 5,
                'time_steps': 11000,
                'transient_steps': 1000,
                'max_nodes': 100000,  # 设置很大的值，默认不启用采样（与老版行为一致）
                'sample_method': 'hybrid',
                'hybrid_ratio': 0.3,
                'x_min': -10.0,
                'x_max': 10.0,
                'y_min': -10.0,
                'y_max': 10.0,
                'z_min': 0.0,
                'z_max': 50.0,
                'state_clip_min': -200.0,
                'state_clip_max': 200.0,
                'dynamics_clip_min': -5000.0,
                'dynamics_clip_max': 5000.0,
            }
    
    def _print_config(self, config_dict: Dict, n_scales: int, n_jobs: int):
        """打印配置信息"""
        rows = [
            ["振荡器类型", "Lorenz"],
            ["α (sigma)", config_dict['alpha']],
            ["β (beta)", f"{config_dict['beta']:.6f}"],
            ["γ (rho)", config_dict['gamma']],
            ["耦合强度 ε", config_dict['coupling_strength']],
            ["时间步长 h", config_dict['time_step']],
            ["数值方法", config_dict['method']],
            ["总步数", config_dict['time_steps']],
            ["暂态步数", config_dict['transient_steps']],
            ["有效步数", config_dict['time_steps'] - config_dict['transient_steps']],
            ["尺度数量", n_scales],
            ["最大节点数", config_dict['max_nodes']],
            ["并行作业数", n_jobs],
        ]
        print_table("Lorenz动力学参数", ["参数", "值"], rows)
    
    def _save_results(
        self,
        dynamics_data: np.ndarray,
        trajectories: np.ndarray,
        all_features: List[np.ndarray],
    ):
        """保存结果"""
        data_dir = self.output_dir / "data"
        
        save_numpy(dynamics_data, data_dir / "dynamics_data.npy")
        save_numpy(trajectories, data_dir / "trajectories.npy")
        
        # 保存每个尺度的特征
        for i, features in enumerate(all_features):
            save_numpy(features, data_dir / f"features_scale_{i+1:02d}.npy")
    
    def _generate_report(
        self,
        dynamics_data: np.ndarray,
        trajectories: np.ndarray,
        config_dict: Dict,
        scale_stats: List[Dict],
        timers: TimerGroup,
    ):
        """Generate comprehensive analysis report with full config parameters"""
        report_path = self.output_dir / "reports" / "analysis_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("EEG Dynamics Analysis Report\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            # ===== Lorenz Oscillator Parameters =====
            f.write("-" * 80 + "\n")
            f.write("1. LORENZ OSCILLATOR PARAMETERS\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Oscillator Type:     Lorenz\n")
            f.write(f"  alpha (sigma):       {config_dict['alpha']}\n")
            f.write(f"  beta:                {config_dict['beta']:.6f}\n")
            f.write(f"  gamma (rho):         {config_dict['gamma']}\n")
            f.write(f"  rk (periodic BC):    {config_dict['rk']}\n\n")
            
            # ===== Numerical Integration Parameters =====
            f.write("-" * 80 + "\n")
            f.write("2. NUMERICAL INTEGRATION PARAMETERS\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Method:              {config_dict['method']}\n")
            f.write(f"  Time Step (h):       {config_dict['time_step']}\n")
            f.write(f"  Coupling Strength:   {config_dict['coupling_strength']}\n")
            f.write(f"  Coupling Mode:       {config_dict['coupling_mode']}\n")
            f.write(f"  Periodic Boundary:   {config_dict['use_periodic_boundary']}\n\n")
            
            # ===== Analysis Parameters =====
            f.write("-" * 80 + "\n")
            f.write("3. ANALYSIS PARAMETERS\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Number of Scales:    {config_dict['n_scales']}\n")
            f.write(f"  Total Time Steps:    {config_dict['time_steps']}\n")
            f.write(f"  Transient Steps:     {config_dict['transient_steps']}\n")
            f.write(f"  Effective Steps:     {config_dict['time_steps'] - config_dict['transient_steps']}\n")
            f.write(f"  Max Nodes:           {config_dict['max_nodes']}\n\n")
            
            # ===== Initial Values =====
            f.write("-" * 80 + "\n")
            f.write("4. INITIAL VALUE RANGES\n")
            f.write("-" * 80 + "\n")
            f.write(f"  X range:             [{config_dict['x_min']}, {config_dict['x_max']}]\n")
            f.write(f"  Y range:             [{config_dict['y_min']}, {config_dict['y_max']}]\n")
            f.write(f"  Z range:             [{config_dict['z_min']}, {config_dict['z_max']}]\n\n")
            
            # ===== Stability Parameters =====
            f.write("-" * 80 + "\n")
            f.write("5. NUMERICAL STABILITY CONTROL\n")
            f.write("-" * 80 + "\n")
            f.write(f"  State Clip Range:    [{config_dict['state_clip_min']}, {config_dict['state_clip_max']}]\n")
            f.write(f"  Dynamics Clip Range: [{config_dict['dynamics_clip_min']}, {config_dict['dynamics_clip_max']}]\n\n")
            
            # ===== Scale Evolution Results =====
            f.write("-" * 80 + "\n")
            f.write("6. SCALE EVOLUTION RESULTS\n")
            f.write("-" * 80 + "\n")
            success_count = 0
            for i, stats in enumerate(scale_stats):
                if stats:
                    status = "SUCCESS" if stats.get('success') else "FAILED"
                    if stats.get('success'):
                        success_count += 1
                    f.write(f"  Scale {i+1}: {status}")
                    if stats.get('success'):
                        f.write(f" | Nodes: {stats['n_nodes']} | Steps: {stats['n_steps']}")
                        f.write(f" | X: [{stats['x_range'][0]:.2f}, {stats['x_range'][1]:.2f}]")
                        f.write(f" | Z: [{stats['z_range'][0]:.2f}, {stats['z_range'][1]:.2f}]")
                    elif stats.get('error'):
                        f.write(f" | Error: {stats['error']}")
                    f.write("\n")
            f.write(f"\n  Total: {success_count}/{len(scale_stats)} scales succeeded\n\n")
            
            # ===== Output Statistics =====
            f.write("-" * 80 + "\n")
            f.write("7. OUTPUT STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Feature Matrix Shape:     {dynamics_data.shape}\n")
            f.write(f"  Trajectory Data Shape:    {trajectories.shape}\n")
            f.write(f"  Features per Scale:       14\n")
            f.write(f"  Total Feature Dimensions: {dynamics_data.shape[1]}\n\n")
            f.write(f"  Feature Statistics:\n")
            f.write(f"    Mean:      {np.mean(dynamics_data):.6f}\n")
            f.write(f"    Std:       {np.std(dynamics_data):.6f}\n")
            f.write(f"    Min:       {np.min(dynamics_data):.6f}\n")
            f.write(f"    Max:       {np.max(dynamics_data):.6f}\n\n")
            
            # ===== Timing =====
            f.write("-" * 80 + "\n")
            f.write("8. TIMING INFORMATION\n")
            f.write("-" * 80 + "\n")
            for name, elapsed in timers.timers.items():
                f.write(f"  {name}: {elapsed:.2f}s\n")
            f.write(f"  Total: {timers.total_elapsed:.2f}s\n\n")
            
            f.write("=" * 80 + "\n")
        
        print_success(f"Report saved: {report_path}")
