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

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from datetime import datetime
import multiprocessing
from joblib import Parallel, delayed

from .oscillators import LorenzOscillator, generate_initial_conditions
from .solvers import create_solver, SolverConfig
from .features import extract_trajectory_features

from ..utils.logger import (
    get_logger, console, create_progress,
    print_header, print_step, print_success, print_warning,
    print_table, print_panel, format_time
)
from ..utils.timer import Timer, TimerGroup
from ..utils.io import save_numpy, ensure_dir, get_output_dir


logger = get_logger(__name__)


def _process_single_scale(args) -> Tuple[int, np.ndarray, np.ndarray, bool]:
    """
    处理单个尺度（用于并行处理）
    
    参数
    ----
    args : tuple
        (scale_idx, L, config_dict)
        
    返回
    ----
    Tuple[int, np.ndarray, np.ndarray, bool]
        (尺度索引, 特征, 轨迹, 成功标志)
    """
    scale_idx, L, config_dict = args
    
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
        
        # 采样节点
        n_nodes = L.shape[0]
        max_nodes = config_dict['max_nodes']
        
        if n_nodes > max_nodes:
            # 混合采样
            n_random = int(max_nodes * 0.3)
            n_uniform = max_nodes - n_random
            
            random_indices = np.random.choice(n_nodes, n_random, replace=False)
            uniform_indices = np.linspace(0, n_nodes-1, n_uniform, dtype=int)
            selected_indices = np.unique(np.concatenate([random_indices, uniform_indices]))[:max_nodes]
            
            L = L[np.ix_(selected_indices, selected_indices)]
            n_nodes = L.shape[0]
        
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
        
        # 提取特征
        features = extract_trajectory_features(trajectories)
        
        return (scale_idx, features, trajectories, True)
        
    except Exception as e:
        logger.error(f"尺度 {scale_idx + 1} 处理失败: {e}")
        n_nodes = L.shape[0]
        n_times = config_dict['time_steps'] - config_dict['transient_steps']
        features = np.zeros((n_nodes, 14))
        trajectories = np.zeros((n_nodes, n_times, 3))
        return (scale_idx, features, trajectories, False)


class DynamicsAnalysisPipeline:
    """
    动力学分析流水线
    
    执行完整的动力学分析流程：
    1. 加载拉普拉斯矩阵
    2. 创建振荡器网络
    3. 数值演化
    4. 特征提取
    5. 保存结果
    
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
        
        # 打印参数
        self._print_config(config_dict, n_scales, n_jobs)
        
        # 准备并行任务
        console.print(f"\n[bold cyan]开始处理 {n_scales} 个尺度...[/bold cyan]")
        
        tasks = [
            (scale_idx, laplacian_family[scale_idx], config_dict)
            for scale_idx in range(n_scales)
        ]
        
        # 并行处理
        with timers.timer("动力学演化", verbose=False):
            results = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(_process_single_scale)(task) for task in tasks
            )
        
        # 整理结果
        results.sort(key=lambda x: x[0])
        
        all_features = []
        all_trajectories = []
        success_count = 0
        
        for scale_idx, features, trajectories, success in results:
            all_features.append(features)
            all_trajectories.append(trajectories)
            if success:
                success_count += 1
                print_success(f"尺度 {scale_idx + 1} 处理完成")
            else:
                print_warning(f"尺度 {scale_idx + 1} 处理失败")
        
        # 合并特征
        with timers.timer("特征合并", verbose=False):
            dynamics_data = np.concatenate(all_features, axis=1)
            trajectories_array = np.stack(all_trajectories)
        
        # 保存结果
        with timers.timer("保存结果", verbose=False):
            self._save_results(dynamics_data, trajectories_array, all_features)
        
        timers.stop_total()
        
        # 打印统计
        self._print_summary(dynamics_data, trajectories_array, n_scales, success_count)
        timers.report()
        
        # 生成报告
        self._generate_report(dynamics_data, trajectories_array, config_dict, timers)
        
        print_panel(
            f"分析完成！\n输出目录: {self.output_dir}",
            title="动力学分析",
            style="green"
        )
        
        return dynamics_data, trajectories_array
    
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
                'max_nodes': 1840,
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
            ["尺度数量", n_scales],
            ["最大节点数", config_dict['max_nodes']],
            ["并行作业数", n_jobs],
        ]
        print_table("动力学参数", ["参数", "值"], rows)
    
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
        save_numpy(np.array(all_features, dtype=object), data_dir / "all_features.npy")
    
    def _print_summary(
        self,
        dynamics_data: np.ndarray,
        trajectories: np.ndarray,
        n_scales: int,
        success_count: int,
    ):
        """打印统计摘要"""
        rows = [
            ["特征矩阵形状", str(dynamics_data.shape)],
            ["轨迹数据形状", str(trajectories.shape)],
            ["成功尺度数", f"{success_count}/{n_scales}"],
            ["特征均值", f"{np.mean(dynamics_data):.4f}"],
            ["特征标准差", f"{np.std(dynamics_data):.4f}"],
            ["特征范围", f"[{np.min(dynamics_data):.4f}, {np.max(dynamics_data):.4f}]"],
        ]
        print_table("分析结果统计", ["项目", "值"], rows)
    
    def _generate_report(
        self,
        dynamics_data: np.ndarray,
        trajectories: np.ndarray,
        config_dict: Dict,
        timers: TimerGroup,
    ):
        """生成分析报告"""
        report_path = self.output_dir / "reports" / "analysis_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("EEG动力学分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输出目录: {self.output_dir}\n\n")
            
            f.write("参数设置\n")
            f.write("-" * 30 + "\n")
            for key, value in config_dict.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("结果统计\n")
            f.write("-" * 30 + "\n")
            f.write(f"特征矩阵形状: {dynamics_data.shape}\n")
            f.write(f"轨迹数据形状: {trajectories.shape}\n")
            f.write(f"特征均值: {np.mean(dynamics_data):.4f}\n")
            f.write(f"特征标准差: {np.std(dynamics_data):.4f}\n")
            f.write("\n")
            
            f.write("=" * 60 + "\n")
        
        print_success(f"报告已保存: {report_path}")
